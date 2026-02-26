"""Benchmark runners for 8 OCR evaluation tasks."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Allow loading very large images (OmniDocBench has 145M-pixel scans).
# client.py already resizes to 2048x2048 before sending to the model.
from PIL import Image as _PILImage
_PILImage.MAX_IMAGE_PIXELS = None

from tqdm import tqdm

from config import BENCHMARKS, PREPARED_DIR, RESULTS_DIR, BenchmarkConfig
from datasets_loader import LOADERS, BenchmarkSample
from metrics import (
    compute_bleu,
    compute_cdm,
    compute_cer,
    compute_kie_anls,
    compute_nid,
    compute_teds,
    compute_wer,
    normalized_edit_distance,
    ocrbench_score,
    omnidocbench_preprocess,
)
from prompts import get_prompt


@dataclass
class BenchmarkResult:
    benchmark_name: str
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample_results: list[dict] = field(default_factory=list)
    total_samples: int = 0
    elapsed_seconds: float = 0.0


def _load_prepared(prepared_dir: Path) -> list[BenchmarkSample]:
    """Load benchmark samples from prepared JPEG+JSONL directory (lazy: paths only)."""
    meta_path = prepared_dir / "metadata.jsonl"
    img_dir = prepared_dir / "images"
    samples = []
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            img_path = str(img_dir / f"{rec['idx']:05d}.jpg")
            samples.append(BenchmarkSample(
                image=None,
                ground_truth=rec["ground_truth"],
                metadata=rec.get("metadata", {}),
                sample_id=rec["sample_id"],
                image_path=img_path,
            ))
    return samples


_MAX_LOAD_PIXELS = 2048 * 2048  # match client.py MAX_IMAGE_PIXELS


def _load_image(sample: BenchmarkSample) -> _PILImage.Image:
    """Load image from sample — lazy path or already-loaded PIL Image.

    Resizes to ≤2048x2048 on load to prevent RAM blowup from huge scans.
    client.py does the same resize before base64-encoding, so this is safe.
    """
    if sample.image is not None:
        img = sample.image
    elif sample.image_path is not None:
        img = _PILImage.open(sample.image_path).convert("RGB")
    else:
        raise ValueError(f"Sample {sample.sample_id} has no image or image_path")
    w, h = img.size
    if w * h > _MAX_LOAD_PIXELS:
        scale = (_MAX_LOAD_PIXELS / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)
    return img


def run_benchmark(
    client,
    model_name: str,
    benchmark_key: str,
    max_samples_override: int | None = None,
    resume: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark against a model client.

    Loads dataset on-demand (freed after benchmark completes).
    Same seed ensures identical samples across model runs.
    """
    bench_cfg = BENCHMARKS[benchmark_key]
    max_samples = max_samples_override if max_samples_override is not None else bench_cfg.max_samples

    # Skip if final result already exists (resume mode)
    result_path = RESULTS_DIR / f"{_safe_name(model_name)}_{benchmark_key}.json"
    if resume and result_path.exists():
        print(f"[Benchmark] {bench_cfg.name} already completed, loading from {result_path.name}")
        with open(result_path) as f:
            data = json.load(f)
        return BenchmarkResult(**data)

    # Load from prepared directory (fast) or fall back to HuggingFace
    prepared_dir = PREPARED_DIR / benchmark_key
    meta_path = prepared_dir / "metadata.jsonl"
    if meta_path.exists():
        print(f"[Benchmark] Loading {bench_cfg.name} from prepared files...")
        samples = _load_prepared(prepared_dir)
        print(f"[Benchmark] Loaded {len(samples)} prepared samples for {bench_cfg.name}")
    else:
        loader = LOADERS[benchmark_key]
        print(f"[Benchmark] Loading {bench_cfg.name} from HuggingFace (no prepared dir)...")
        samples = loader(max_samples)
        print(f"[Benchmark] Loaded {len(samples)} samples for {bench_cfg.name}")

    # Resume from checkpoint if requested
    checkpoint_path = RESULTS_DIR / f"{_safe_name(model_name)}_{benchmark_key}_checkpoint.json"
    completed_ids: set[str] = set()
    per_sample_results: list[dict] = []

    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        per_sample_results = checkpoint.get("per_sample_results", [])
        completed_ids = {r["sample_id"] for r in per_sample_results}
        print(f"[Benchmark] Resuming: {len(completed_ids)} samples already done")

    # Get evaluator
    evaluator = _get_evaluator(bench_cfg)

    start_time = time.time()
    desc = f"{model_name} | {bench_cfg.name}"

    # Filter to remaining samples
    remaining = [s for s in samples if s.sample_id not in completed_ids]

    BATCH_SIZE = 1  # sequential: 1 request at a time to avoid vLLM deadlock on large images
    CONSECUTIVE_ERROR_CHECK = 3  # check server health after this many consecutive empty results
    MAX_TOTAL_TIMEOUTS = 20  # abort benchmark after this many total timeouts

    # Quick pre-check: skip immediately if server is already dead
    if remaining and hasattr(client, "client") and not _check_server_health(client):
        print(f"[Benchmark] Server unreachable — skipping {bench_cfg.name}")
        return BenchmarkResult(
            benchmark_name=bench_cfg.name, model_name=model_name,
            metrics={}, per_sample_results=per_sample_results,
            total_samples=len(per_sample_results), elapsed_seconds=0.0,
        )

    pbar = tqdm(total=len(remaining), desc=desc, unit="sample", initial=0)
    consecutive_errors = 0
    total_timeouts = 0
    server_dead = False

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start : batch_start + BATCH_SIZE]

        # Prepare prompts and lazy-load images for this batch only
        batch_prompts = []
        batch_images = []
        for sample in batch:
            question = sample.metadata.get("question")
            prompt = get_prompt(bench_cfg.prompt_key, question, model_name=model_name)
            batch_prompts.append(prompt)
            batch_images.append(_load_image(sample))

        # Batch inference (concurrent for vLLM, sequential for MinerU)
        # All clients now return (text, latency_ms) tuples
        if hasattr(client, "batch_infer"):
            infer_results = client.batch_infer(batch_images, batch_prompts)
        else:
            infer_results = [client.infer(img, p) for img, p in zip(batch_images, batch_prompts)]

        # Free batch images immediately after inference
        del batch_images

        # Evaluate each sample
        for sample, (prediction, latency_ms) in zip(batch, infer_results):
            # Track consecutive empty predictions (server hang indicator)
            if not prediction.strip():
                consecutive_errors += 1
                total_timeouts += 1

                if consecutive_errors >= CONSECUTIVE_ERROR_CHECK:
                    if hasattr(client, "client") and not _check_server_health(client):
                        print(f"\n[Benchmark] Server dead after {consecutive_errors} consecutive errors. Aborting {bench_cfg.name}.")
                        server_dead = True
                        break
                    else:
                        # Server alive but these samples timed out — skip and continue
                        print(f"\n[Benchmark] {consecutive_errors} consecutive timeouts but server alive — skipping problematic samples")
                        consecutive_errors = 0

                if total_timeouts >= MAX_TOTAL_TIMEOUTS:
                    print(f"\n[Benchmark] {total_timeouts} total timeouts — aborting {bench_cfg.name}")
                    server_dead = True
                    break
            else:
                consecutive_errors = 0

            try:
                scores = evaluator(prediction, sample.ground_truth, sample.metadata)
                per_sample_results.append({
                    "sample_id": sample.sample_id,
                    "prediction": prediction,
                    "scores": scores,
                    "latency_ms": latency_ms,
                    "error": None,
                })
            except Exception as e:
                per_sample_results.append({
                    "sample_id": sample.sample_id,
                    "prediction": prediction,
                    "scores": {},
                    "latency_ms": latency_ms,
                    "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                })
            completed_ids.add(sample.sample_id)

        if server_dead:
            break

        pbar.update(len(batch))

        # Checkpoint after each batch
        _save_checkpoint(checkpoint_path, model_name, benchmark_key, per_sample_results)

    pbar.close()

    elapsed = time.time() - start_time

    # If server died, keep checkpoint for resume — do NOT save final result
    if server_dead:
        _save_checkpoint(checkpoint_path, model_name, benchmark_key, per_sample_results)
        aggregate = _aggregate_metrics(per_sample_results, bench_cfg.metric_type)
        print(f"[Benchmark] {bench_cfg.name} ABORTED (server dead): {len(per_sample_results)} samples completed")
        return BenchmarkResult(
            benchmark_name=bench_cfg.name,
            model_name=model_name,
            metrics=aggregate,
            per_sample_results=per_sample_results,
            total_samples=len(per_sample_results),
            elapsed_seconds=elapsed,
        )

    # Compute aggregate metrics
    aggregate = _aggregate_metrics(per_sample_results, bench_cfg.metric_type)

    result = BenchmarkResult(
        benchmark_name=bench_cfg.name,
        model_name=model_name,
        metrics=aggregate,
        per_sample_results=per_sample_results,
        total_samples=len(per_sample_results),
        elapsed_seconds=elapsed,
    )

    # Save final result
    result_path = RESULTS_DIR / f"{_safe_name(model_name)}_{benchmark_key}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    print(f"[Benchmark] {bench_cfg.name} done: {aggregate}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return result


def _get_evaluator(bench_cfg: BenchmarkConfig):
    """Return the evaluation function for a benchmark type."""
    evaluators = {
        "document_parse": _eval_document_parse,
        "document_parse_dp": _eval_dp_bench,
        "text_recognition": _eval_ocrbench,
        "formula_recognition": _eval_formula,
        "table_recognition": _eval_table,
        "kie_extraction": _eval_kie,
        "handwritten": _eval_handwritten,
    }
    return evaluators[bench_cfg.metric_type]


# ─── OmniDocBench official evaluation helpers ─────────────────────────

_OMNIDOCBENCH_ROOT = "/home/ubuntu/OmniDocBench"
_omnidoc_path_added = False


def _ensure_omnidocbench_imports():
    """Add OmniDocBench root to sys.path for its internal relative imports.

    Uses append (not insert) to avoid shadowing our own metrics.py module
    with OmniDocBench's metrics/ package.
    """
    global _omnidoc_path_added
    if not _omnidoc_path_added:
        import sys
        if _OMNIDOCBENCH_ROOT not in sys.path:
            sys.path.append(_OMNIDOCBENCH_ROOT)
        _omnidoc_path_added = True


def _extract_omnidoc_gt_elements(anno: dict) -> dict[str, list]:
    """Extract GT elements from OmniDocBench annotation, handling truncated text.

    Replicates OmniDocBench/dataset/end2end_dataset.py:get_page_elements().
    Returns dict mapping category_type → list of element dicts.
    """
    from collections import defaultdict

    saved_element_dict: dict[str, list] = defaultdict(list)
    related_truncated: list[list] = []
    truncated_all: dict = {}

    # Handle truncated text relations
    extra = anno.get("extra", {})
    relations = extra.get("relation", [])
    for relation in relations:
        if relation.get("relation_type") == "truncated":
            src = relation["source_anno_id"]
            tgt = relation["target_anno_id"]
            truncated_all[src] = ""
            truncated_all[tgt] = ""
            exist_flag = False
            for merge_list in related_truncated:
                if src in merge_list or tgt in merge_list:
                    merge_list.append(src)
                    merge_list.append(tgt)
                    exist_flag = True
                    break
            if not exist_flag:
                related_truncated.append([src, tgt])

    # Organize elements by category, separating truncated ones
    for item in anno.get("layout_dets", []):
        if item.get("anno_id") not in truncated_all:
            saved_element_dict[item["category_type"]].append(item)
        else:
            truncated_all[item["anno_id"]] = item

    # Merge truncated text blocks
    for merge_list in related_truncated:
        text_block_list = [truncated_all[key] for key in merge_list
                          if isinstance(truncated_all.get(key), dict)]
        if not text_block_list:
            continue
        sorted_block = sorted(text_block_list, key=lambda x: x.get("order", 0))
        text = "".join(block.get("text", "") for block in sorted_block)
        merged_block = {
            "category_type": sorted_block[0]["category_type"],
            "order": sorted_block[0].get("order", 0),
            "anno_id": sorted_block[0].get("anno_id", ""),
            "text": text,
            "merge_list": sorted_block,
        }
        saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)

    return dict(saved_element_dict)


# ─── Per-benchmark evaluators ─────────────────────────────────────────

def _eval_document_parse(pred: str, gt, metadata: dict) -> dict:
    """OmniDocBench: Official element-wise evaluation protocol.

    Parses prediction markdown with md_tex_filter(), matches against
    GT annotations with match_gt2pred_quick(), and computes:
    - Text: Edit Distance (normalized, 0-1, lower=better)
    - Table: TEDS (0-100, higher=better)
    - Formula: Edit Distance (normalized, 0-1, lower=better)
    - Overall: ((1-text_ED)*100 + table_TEDS + (1-formula_ED)*100) / 3

    Falls back to legacy evaluator for old-format ground truth (plain string).
    """
    # Handle legacy format (plain string GT)
    if not isinstance(gt, dict) or "annotation" not in gt:
        return _eval_document_parse_legacy(pred, gt, metadata)

    _ensure_omnidocbench_imports()
    from utils.extract import md_tex_filter
    from utils.match_quick import match_gt2pred_quick
    from utils.match import match_gt2pred_simple

    anno = gt["annotation"]
    img_name = metadata.get("image_filename", str(gt.get("annotation_index", "unknown")))

    # 1. Parse prediction markdown into structured elements
    pred_dataset = md_tex_filter(pred) if pred.strip() else {}

    # 2. Extract GT elements from annotation
    gt_elements = _extract_omnidoc_gt_elements(anno)

    # 3. Gather GT text+formula elements (same categories as official eval)
    text_formula_categories = [
        'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference',
        'equation_caption', 'figure_caption', 'figure_footnote',
        'table_caption', 'table_footnote', 'code_algorithm',
        'code_algorithm_caption', 'header', 'footer', 'page_footnote',
        'page_number', 'equation_isolated',
    ]
    gt_mix = []
    for cat in text_formula_categories:
        if gt_elements.get(cat):
            gt_mix.extend(gt_elements[cat])
    gt_mix = sorted(gt_mix, key=lambda x: x.get('order') or 0)

    # 4. Gather pred elements (non-table)
    pred_mix = []
    for cat in pred_dataset:
        if cat not in ('html_table', 'latex_table', 'md2html_table'):
            pred_mix.extend(pred_dataset[cat])

    # 5. Match tables separately (official protocol)
    table_teds_scores = []
    if gt_elements.get('table'):
        gt_tables = sorted(gt_elements['table'], key=lambda x: x.get('order') or 0)
        latex_n = len(pred_dataset.get('latex_table', []))
        html_n = len(pred_dataset.get('html_table', []))

        if latex_n == 0 and html_n == 0:
            # No pred tables — score all GT tables as 0
            table_matches, unmatch_table_pred = match_gt2pred_simple(
                gt_tables, [], 'html_table', img_name)
        elif latex_n > html_n:
            table_matches, unmatch_table_pred = match_gt2pred_simple(
                gt_tables, pred_dataset['latex_table'], 'latex_table', img_name)
        else:
            table_matches, unmatch_table_pred = match_gt2pred_simple(
                gt_tables, pred_dataset['html_table'], 'html_table', img_name)

        # Move unmatched table predictions to text matching pool
        if unmatch_table_pred:
            pred_mix.extend(unmatch_table_pred)

        # Compute TEDS for each matched table
        for m in table_matches:
            if m.get('gt_idx') == [""] or not m.get('gt', ''):
                continue  # Skip extra pred-only matches
            gt_table = m.get('gt', '')
            pred_table = m.get('pred', '')
            if gt_table:
                teds_val = compute_teds(pred_table, gt_table) * 100  # 0-100 scale
                table_teds_scores.append(teds_val)

    # 6. Match text+formula elements using quick_match (with timeout fallback)
    text_edit_scores = []
    formula_edit_scores = []

    ignore_categories = {
        'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote',
        'code_algorithm', 'code_algorithm_caption', 'header', 'footer',
        'page_footnote', 'page_number', 'equation_caption',
    }

    if gt_mix or pred_mix:
        try:
            from func_timeout import func_timeout, FunctionTimedOut
            try:
                matches = func_timeout(
                    30, match_gt2pred_quick,
                    args=(gt_mix, pred_mix, 'text_all', img_name))
            except FunctionTimedOut:
                matches, _ = match_gt2pred_simple(
                    gt_mix, pred_mix, 'text_all', img_name)
        except ImportError:
            # func_timeout not installed — run without timeout
            matches = match_gt2pred_quick(gt_mix, pred_mix, 'text_all', img_name)

        for m in matches:
            gt_cat = m.get('gt_category_type', '')
            edit = m.get('edit', 1.0)

            if gt_cat in ignore_categories:
                continue  # Matched but not scored (official protocol)

            if gt_cat == 'equation_isolated':
                formula_edit_scores.append(edit)
            elif gt_cat:
                text_edit_scores.append(edit)

    # 7. Compute per-type averages
    avg_text_ed = (sum(text_edit_scores) / len(text_edit_scores)
                   if text_edit_scores else 1.0)
    text_score = (1.0 - avg_text_ed) * 100

    avg_table_teds = (sum(table_teds_scores) / len(table_teds_scores)
                      if table_teds_scores else 0.0)

    avg_formula_ed = (sum(formula_edit_scores) / len(formula_edit_scores)
                      if formula_edit_scores else 1.0)
    formula_score = (1.0 - avg_formula_ed) * 100

    # 8. Overall: average only over element types that exist on this page
    components = []
    if text_edit_scores:
        components.append(text_score)
    if table_teds_scores:
        components.append(avg_table_teds)
    if formula_edit_scores:
        components.append(formula_score)
    overall = sum(components) / len(components) if components else 0.0

    return {
        "text_edit_dist": avg_text_ed,
        "text_score": text_score,
        "table_teds": avg_table_teds,
        "formula_edit_dist": avg_formula_ed,
        "formula_score": formula_score,
        "overall": overall,
        "n_text_elements": len(text_edit_scores),
        "n_table_elements": len(table_teds_scores),
        "n_formula_elements": len(formula_edit_scores),
    }


def _eval_document_parse_legacy(pred: str, gt, metadata: dict) -> dict:
    """Legacy OmniDocBench evaluator: Edit Distance + BLEU on concatenated text.

    Preserved for backward compatibility with old-format ground truth.
    """
    gt_str = gt if isinstance(gt, str) else str(gt)
    norm_pred = omnidocbench_preprocess(pred)
    norm_gt = omnidocbench_preprocess(gt_str)
    return {
        "edit_dist": normalized_edit_distance(norm_pred, norm_gt),
        "bleu": compute_bleu(norm_pred, norm_gt),
    }


def _strip_markdown_for_nid(text: str) -> str:
    """Strip markdown artifacts from prediction text for NID comparison.

    Matches official DP-Bench eval which works with plain text elements.
    """
    import re as _re
    # Remove markdown image references: ![...](...)
    text = _re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # Remove HTML tables (evaluated separately via TEDS)
    # But if removing tables would leave text empty, extract table text content instead
    text_without_tables = _re.sub(r'<table[^>]*>.*?</table>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
    if _re.sub(r'\s+', '', text_without_tables):
        # Non-empty after stripping tables — use the stripped version
        text = text_without_tables
    else:
        # Entire prediction was table(s) — extract cell text for NID comparison
        text = _re.sub(r'<[^>]+>', ' ', text)  # strip all HTML tags, replace with space
    # Remove markdown heading markers: # ## ### etc.
    text = _re.sub(r'^#{1,6}\s+', '', text, flags=_re.MULTILINE)
    # Remove markdown table blocks (lines with pipes that form tables)
    lines = text.splitlines()
    filtered = []
    in_display_math = False
    for line in lines:
        stripped = line.strip()
        # Track display math blocks ($$...$$) — don't strip lines inside
        if stripped == '$$':
            in_display_math = not in_display_math
            filtered.append(line)
            continue
        if in_display_math:
            filtered.append(line)
            continue
        # Skip separator lines: |---|---|
        if _re.match(r'^[\s|:-]+$', stripped) and '|' in stripped:
            continue
        # Skip table data lines (start with | and end with |)
        if stripped.startswith('|') and stripped.endswith('|') and '|' in stripped[1:-1]:
            continue
        # Skip non-leading-pipe table lines only if pipes are NOT LaTeX
        if stripped.count('|') >= 2 and not stripped.startswith('|'):
            # Remove LaTeX $...$ content and LaTeX pipe commands
            no_latex = _re.sub(r'\$[^$]+\$', '', stripped)
            no_latex = _re.sub(r'\\(?:left|right|l|r)\|', '', no_latex)
            no_latex = _re.sub(r'\\\|', '', no_latex)
            if no_latex.count('|') >= 2 and _re.match(r'^[^|]+\|.+\|', no_latex):
                continue
        filtered.append(line)
    text = '\n'.join(filtered)
    # Remove newlines (official eval removes \n)
    text = text.replace('\n', ' ')
    # Collapse multiple spaces
    text = _re.sub(r' +', ' ', text).strip()
    return text


def _eval_dp_bench(pred: str, gt, metadata: dict) -> dict:
    """Upstage DP-Bench: NID for text, TEDS for tables."""
    if not isinstance(gt, dict):
        return {"nid": compute_nid(pred, str(gt))}

    elements = gt.get("elements", [])
    if not elements:
        raw = gt.get("raw", {})
        gt_text = raw.get("text", raw.get("markdown", str(gt)))
        return {"nid": compute_nid(pred, str(gt_text))}

    # Official DP-Bench ignores figure/table/chart for NID text comparison
    _NID_IGNORE_CATS = {"figure", "table", "chart"}

    # Compute NID on text content (excluding figure/table/chart per official eval)
    gt_texts = []
    gt_tables_html = []
    for elem in elements:
        if isinstance(elem, dict):
            content = elem.get("content", {})
            if isinstance(content, dict):
                cat = elem.get("category", "")
                if cat == "Table":
                    html = content.get("html", "")
                    if html:
                        gt_tables_html.append(html)
                # Skip figure/table/chart for NID (matching official eval)
                if cat.lower() in _NID_IGNORE_CATS:
                    continue
                text = content.get("text", content.get("markdown", ""))
                if text:
                    gt_texts.append(str(text))
            elif isinstance(content, str):
                gt_texts.append(content)

    # Official eval: concatenate with space, remove newlines
    import re as _re_dp
    gt_full_text = ' '.join(gt_texts).replace('\n', '')
    gt_full_text = _re_dp.sub(r' +', ' ', gt_full_text).strip()
    # Strip markdown artifacts from prediction
    pred_text = _strip_markdown_for_nid(pred)
    scores = {"nid": compute_nid(pred_text, gt_full_text)}

    # TEDS for tables if present — extract tables from prediction first
    if gt_tables_html:
        pred_tables = _extract_tables_from_pred(pred)
        teds_scores = []
        teds_s_scores = []
        for gt_html in gt_tables_html:
            # Ensure GT HTML is wrapped in <table> (DP-Bench GT may omit it)
            gt_table = gt_html.strip()
            if not gt_table.lower().startswith("<table"):
                gt_table = f"<table>{gt_table}</table>"

            best_teds = 0.0
            best_teds_s = 0.0
            if pred_tables:
                # Match each GT table against the best-matching predicted table
                for pt in pred_tables:
                    t = compute_teds(pt, gt_table)
                    ts = compute_teds(pt, gt_table, structure_only=True)
                    if t > best_teds:
                        best_teds = t
                        best_teds_s = ts
            teds_scores.append(best_teds)
            teds_s_scores.append(best_teds_s)
        scores["teds"] = sum(teds_scores) / len(teds_scores)
        scores["teds_structure"] = sum(teds_s_scores) / len(teds_s_scores)

    return scores


def _eval_ocrbench(pred: str, gt, metadata: dict) -> dict:
    """OCRBench: Official VQA scoring from MultimodalOCR repo.

    Two-tier approach:
    - Short answers (<5 words): substring match (gt in pred) → 0 or 1
    - Long answers (≥5 words): ANLS with ≥0.5 threshold → [0.5, 1]
    Returns max score across all valid answers.

    Special case: HME100k dataset does NOT lowercase and removes all spaces
    (from MultimodalOCR/OCRBench/example.py).
    """
    if isinstance(gt, list):
        answers = gt  # preserve original types for int/float conversion
    else:
        answers = [gt]

    dataset_name = metadata.get("dataset", "")
    return {
        "accuracy": ocrbench_score(pred, answers, dataset_name=dataset_name),
    }


def _eval_formula(pred: str, gt, metadata: dict) -> dict:
    """UniMERNet: Normalized Edit Distance + CDM F1 (per-sample).

    Official UniMERNet evaluation (test.py:190-196) normalizes BOTH
    pred and gt with normalize_text() BEFORE computing metrics.

    CDM F1 is the official metric — renders LaTeX to images and matches
    characters visually. KaTeX exit() bug + missing environments patched.

    Note: BLEU is computed at corpus-level in eval_bench.py, not per-sample.
    """
    from metrics import _normalize_latex
    gt_str = gt if isinstance(gt, str) else str(gt)
    norm_pred = _normalize_latex(pred)
    norm_gt = _normalize_latex(gt_str)

    # CDM F1 with timeout to prevent hangs
    sample_id = metadata.get("sample_id", str(id(pred)))
    try:
        cdm_result = compute_cdm(pred, gt_str, str(sample_id))
        cdm_f1 = cdm_result.get("F1_score", 0.0)
    except Exception:
        cdm_f1 = 0.0

    return {
        "edit_distance": normalized_edit_distance(norm_pred, norm_gt),
        "cdm_f1": cdm_f1,
        "norm_pred": norm_pred,  # Stored for corpus-level BLEU in eval_bench.py
        "norm_gt": norm_gt,
    }


def _eval_table(pred: str, gt, metadata: dict) -> dict:
    """PubTabNet / TEDS_TEST: TEDS + TEDS-Structure."""
    gt_str = gt if isinstance(gt, str) else str(gt)
    # Convert model-specific table formats to HTML
    if "<fcel>" in pred:
        pred_html = _convert_paddle_table_to_html(pred)
    elif "|" in pred and "<table" not in pred.lower():
        pred_html = _convert_markdown_table_to_html(pred)
    elif "<table" not in pred.lower():
        # Fallback: space-separated table (4+ spaces between columns)
        pred_html = _convert_space_table_to_html(pred)
    else:
        pred_html = pred
    return {
        "teds": compute_teds(pred_html, gt_str),
        "teds_structure": compute_teds(pred_html, gt_str, structure_only=True),
    }


def _extract_kie_fields_from_text(text: str) -> dict:
    """Extract KIE fields from raw OCR text using regex patterns.

    Fallback for when models output raw text instead of structured JSON.
    Targets the 8 Nanonets-KIE fields from Malaysian receipts/invoices.
    Handles both raw OCR text and pipeline markdown output (# headings,
    HTML tables, full-width Unicode colons).
    """
    import re as _re
    fields: dict[str, str] = {}
    lines = text.strip().splitlines()

    # Normalize full-width colons (U+FF1A) to ASCII for regex matching
    # Also strip markdown bold markers (**) that interfere with regex
    normalized = text.replace('\uff1a', ':')
    normalized = normalized.replace('**', '')

    # Extract text from HTML table cells for amount searching
    # Pipeline models wrap financial data in <table> tags
    # Convert <tr> to newlines so each row is a separate line for pattern matching
    plain_with_tables = _re.sub(r'<\s*/?\s*tr[^>]*>', '\n', normalized, flags=_re.IGNORECASE)
    plain_with_tables = _re.sub(r'<\s*/?\s*td[^>]*>', ' ', plain_with_tables, flags=_re.IGNORECASE)
    plain_with_tables = _re.sub(r'<[^>]+>', ' ', plain_with_tables)

    # date: multiple format support
    # Formats: DD/MM/YYYY, DD-MM-YYYY, DD Mon YYYY, DD/Mon/YYYY,
    #          DD - Mon - YYYY, Mon DD YYYY, DD / MM / YYYY, DD.MM.YY
    _MONTHS = (r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
               r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
               r'Nov(?:ember)?|Dec(?:ember)?)')
    _DATE_PATTERNS = [
        # P1: DD/MM/YYYY or DD-MM-YYYY (compact)
        r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        # P2: DD Mon YYYY (e.g., "22 Jun 2018", "03 Jun 18")
        rf'(\d{{1,2}}\s+{_MONTHS}\.?\s+\d{{2,4}})',
        # P3: DD/Mon/YYYY (e.g., "05/Apr/2018", "20/Jan/2017")
        rf'(\d{{1,2}}/{_MONTHS}/\d{{2,4}})',
        # P4: DD - Mon - YYYY (e.g., "05 - Jan - 2017", "28 - Feb - 2018")
        rf'(\d{{1,2}}\s*-\s*{_MONTHS}\s*-\s*\d{{2,4}})',
        # P5: Mon DD, YYYY (e.g., "Oct 3 , 2015", "March 21, 2018")
        rf'({_MONTHS}\s+\d{{1,2}}\s*,?\s*\d{{4}})',
        # P6: DD / Mon / YYYY (spaced, e.g., "15 / Apr / 2017")
        rf'(\d{{1,2}}\s*/\s*{_MONTHS}\s*/\s*\d{{2,4}})',
        # P7: DD / MM / YYYY (spaced slashes, e.g., "01 / 05 / 2016")
        r'(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})',
        # P8: DD.MM.YY (dotted, e.g., "23.03.18")
        r'(\d{1,2}\.\d{1,2}\.\d{2,4})',
    ]
    # First try: preceded by "Date" label
    for search_text in [normalized, plain_with_tables]:
        for pat in _DATE_PATTERNS:
            date_m = _re.search(
                r'(?:Date\s*[:\.]?\s*\|?\s*)' + pat,
                search_text, _re.IGNORECASE,
            )
            if date_m:
                fields["date"] = date_m.group(1).strip()
                break
        if "date" in fields:
            break
    if "date" not in fields:
        # Fallback: any date pattern in either text (first match wins)
        for search_text in [normalized, plain_with_tables]:
            for pat in _DATE_PATTERNS:
                date_m = _re.search(pat, search_text, _re.IGNORECASE)
                if date_m:
                    fields["date"] = date_m.group(1).strip()
                    break
            if "date" in fields:
                break

    # doc_no_receipt_no: receipt/invoice/doc number (single line only)
    # Search in both normalized text and HTML-stripped text for table cells
    for search_text in [normalized, plain_with_tables]:
        doc_m = _re.search(
            r'(?:Doc(?:ument)?\s*No\.?|Receipt\s*(?:No\.?|N:?|H:?|#:?)|'
            r'Invoice\.?\s*(?:No\.?|#:?|number)|INVO\s*(?:No\.?|#:?)|'
            r'Bill\s*(?:No\.?|#:?)|(?:Cash\s+)?\b(?:INV|TNV|IN)\.?\s*(?:No\.?|#:?)|'
            r'Order\s*No\.?|Slip\s*(?:No\.?|:)|'
            r'Trans(?:action)?\s*No\.?|Ref(?:erence)?\s*(?:No\.?|#:?)|'
            r'Invoice\s*:|Rcpt\s*#:?|CB\s*#:?|(?:Check|Chk)\s*#?\s*:?|'
            r'(?<![A-Za-z])(?:ce|ice|oice|voice|price)\s*#\s*:?|'
            r'(?:Tax\s*)?[1I]?nv(?:oice)?\s*(?:No\.?|#:?)?[:/\s]*(?=\d)|'
            r'C/N\s*No\.?\s*:?)'
            r'\s*[#:\.]?\s*\|?\s*'
            r'([A-Z0-9][A-Z0-9/\-]{0,}(?:\s[A-Z0-9/\-]+)*)',
            search_text, _re.IGNORECASE,
        )
        if doc_m:
            val = doc_m.group(1).strip()
            # Take only first line (stop at newline)
            val = val.split('\n')[0].strip()
            # Remove trailing context words
            val = _re.sub(r'\s+(?:DATE|CASHIER|SALESPERSON|TIME|ORDER|POS|TABLE|COVER|ITEM|QTY|GST|TAX|PRINTED|STATUS).*',
                          '', val, flags=_re.IGNORECASE)
            fields["doc_no_receipt_no"] = val.strip()
            break

    # seller_gst_id: GST registration number (typically 12-15 digits)
    # Handle OCR errors: RST/CST for GST, various label formats
    for search_text in [normalized, plain_with_tables]:
        gst_m = _re.search(
            r'(?:(?:GST|RST|CST|TAX)\s*(?:ID|NO|REG|Registration)\.?\s*(?:NO\.?|ID|#)?\s*[:\.]?\s*'
            r'|NO\.?\s+ID\s+GST\s*[:\.]?\s*)'
            r'(?:CBP\s+)?(\d[\d\s\-]{8,})',
            search_text, _re.IGNORECASE,
        )
        if gst_m:
            val = _re.sub(r'[\s\-]', '', gst_m.group(1)).strip()
            if len(val) >= 9:  # GST IDs are typically 12+ digits
                fields["seller_gst_id"] = val
                break

    # seller_phone: phone number near TEL/PHONE (support full-width colon, Tel No., Tel/Fax)
    for search_text in [normalized, plain_with_tables]:
        phone_m = _re.search(
            r'(?:TEL(?:EPHONE)?(?:\s*(?:No\.?|/\s*Fax))?|PHONE|HP|H/P|'
            r'Mobile(?:\s*(?:No\.?|/\s*Whatsapps?))?|Contact|'
            r'Customer\s+Service(?:\s+Hotline)?|Careline|Hotline)\s*[:\.]*\s*'
            r'(\+?\s*[\d\- ]{7,})',
            search_text, _re.IGNORECASE,
        )
        if phone_m:
            fields["seller_phone"] = phone_m.group(1).strip()
            break
    # Fallback: standalone Malaysian phone on its own line (e.g., "+603-9130 2672")
    if "seller_phone" not in fields:
        for search_text in [normalized, plain_with_tables]:
            for line in search_text.split('\n')[:15]:
                stripped = line.strip().rstrip(',')
                # Match standalone phone: +6xx-xxx xxxx or 0xx-xxx xxxx
                ph_m = _re.match(r'^(\+?6?\d{2,3}[-\s]\d{3,4}[-\s]?\d{3,4})\s*$', stripped)
                if ph_m:
                    fields["seller_phone"] = ph_m.group(1).strip()
                    break
            if "seller_phone" in fields:
                break

    # seller_name: first meaningful line + company name continuation lines
    # For HTML predictions, use plain_with_tables lines as fallback
    name_lines = lines
    first_non_empty = next((l.strip() for l in lines if l.strip()), '')
    if first_non_empty.startswith('<') or first_non_empty.startswith('```'):
        name_lines = plain_with_tables.split('\n')
    name_start_idx = -1
    for i, line in enumerate(name_lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Remove markdown heading markers and bold
        cleaned = _re.sub(r'^#+\s*', '', stripped)
        cleaned = _re.sub(r'\*\*', '', cleaned).strip()
        if not cleaned or len(cleaned) <= 2:
            continue
        # Skip lines that are clearly not company names
        if _re.match(r'^\([\dA-Z\-]+\)$', cleaned):
            continue
        # Skip bare numbers/short codes (receipt/store codes)
        if _re.match(r'^[A-Z0-9/\-]{2,10}$', cleaned):
            continue
        # Skip markdown code block markers (```markdown, ```)
        if cleaned.startswith('```'):
            continue
        # Skip HTML tags/elements that leaked through
        if _re.match(r'^<[^>]+>$', cleaned) or cleaned.startswith('<table'):
            continue
        # Skip lines that are purely digits (store/transaction IDs)
        if _re.match(r'^\d+$', cleaned):
            continue
        # If line is very long and contains company suffix + address markers,
        # split: take only the company name portion
        if len(cleaned) > 50:
            # Strategy 1: Split after company suffix (SDN BHD, ENTERPRISE, etc.)
            # Match the longest company suffix: prioritize multi-word suffixes
            suffix_m = _re.search(
                r'(?:SDN\s*\.?\s*BHD|S/B|ENTERPRISE|TRADING\s+SDN|'
                r'CO\.\s*LTD|PTE\s*\.?\s*LTD|COMPANY)\s*',
                cleaned, _re.IGNORECASE,
            )
            if suffix_m:
                rest = cleaned[suffix_m.end():]
                if _re.search(r'(?:\([A-Z0-9\-]+\)|LOT|NO\.?\s*\d|JALAN|JLN|\d{5})', rest, _re.IGNORECASE):
                    cleaned = cleaned[:suffix_m.end()].strip()
            else:
                # Strategy 2: No company suffix found — split at registration
                # number in parens: "(XXX-X)" followed by address
                reg_m = _re.search(r'\s+\([A-Z0-9\-]+\)\s+', cleaned)
                if reg_m:
                    rest = cleaned[reg_m.end():]
                    if _re.search(r'(?:LOT|NO\.?\s*\d|JALAN|JLN|\d{5})', rest, _re.IGNORECASE):
                        cleaned = cleaned[:reg_m.start()].strip()
        fields["seller_name"] = cleaned
        name_start_idx = i
        break
    # Extend name with continuation lines (SDN BHD, S/B, ENTERPRISE, etc.)
    if name_start_idx >= 0:
        for j in range(name_start_idx + 1, min(name_start_idx + 4, len(name_lines))):
            next_line = _re.sub(r'\*\*', '', name_lines[j]).strip()
            if not next_line:
                continue
            # Stop at address/section markers
            if _re.match(
                r'(?:TEL|PHONE|GST|TAX|NO\b|ADDR|LOT|JLN|JALAN|\d{5}|Owned|COMPANY\s*NO)',
                next_line, _re.IGNORECASE,
            ):
                break
            # Continue if looks like company suffix but NOT if it also has registration number
            if _re.search(
                r'SDN|BHD|S/B|ENTERPRISE|GROUP|TRADING|CO\.\s|COMPANY',
                next_line, _re.IGNORECASE,
            ) and not _re.search(r'\([A-Z0-9\-]+\)', next_line):
                fields["seller_name"] += "\n" + next_line
            else:
                break

    # seller_address: lines after company name (including continuation lines)
    # Use name_start_idx to skip past the seller_name lines
    _name_first_line = fields.get("seller_name", "").split('\n')[0] if "seller_name" in fields else ""
    _name_line_count = len(fields.get("seller_name", "").split('\n')) if "seller_name" in fields else 0
    addr_lines = []
    # Handle one-line headers: if seller_name was split from a long line,
    # extract address from the remainder of the same line
    if name_start_idx >= 0 and "seller_name" in fields:
        orig_line = _re.sub(r'^#+\s*', '', name_lines[name_start_idx].strip())
        orig_line = _re.sub(r'\*\*', '', orig_line).strip()
        if len(orig_line) > len(_name_first_line) + 5:
            remainder = orig_line[len(_name_first_line):].strip()
            # Strip registration numbers in parens at start
            remainder = _re.sub(r'^\([\dA-Z\-]+\)\s*', '', remainder)
            # Strip "COMPANY NO.: ..." prefixes
            remainder = _re.sub(r'^(?:COMPANY\s*NO\.?\s*:?\s*[\dA-Z\-]+\s*)', '', remainder, flags=_re.IGNORECASE)
            # Stop at GST/TAX/TEL markers
            marker = _re.search(r'(?:GST|TAX|TEL|PHONE|www\.)', remainder, _re.IGNORECASE)
            if marker:
                remainder = remainder[:marker.start()].strip()
            if remainder:
                addr_lines.append(remainder)
    found_name = False
    name_lines_skipped = 0
    for line in name_lines:
        stripped = line.strip()
        cleaned = _re.sub(r'^#+\s*', '', stripped)
        cleaned = _re.sub(r'\*\*', '', cleaned).strip()
        if not found_name:
            if cleaned == _name_first_line or cleaned.startswith(_name_first_line):
                found_name = True
                name_lines_skipped = 1
            continue
        # Skip name continuation lines
        if name_lines_skipped < _name_line_count:
            if cleaned:
                name_lines_skipped += 1
            continue
        # Stop at known section markers (with full-width colon support)
        if _re.match(
            r'(?:TEL|PHONE|GST|TAX\s+INVOICE|BILL\s+TO|CASH|RECEIPT|SIMPLIFIED)',
            cleaned.replace('\uff1a', ':'), _re.IGNORECASE,
        ):
            break
        # Skip registration numbers (in parens, or "CO REG:", "Co. No.", "GST Reg No" etc.)
        if _re.match(r'^[-*]?\s*\([\dA-Z\-\s:\.]+\)$', cleaned):
            continue
        if _re.match(
            r'(?:CO[.\-]?\s*REG|Co\.?\s*No|REG\s*NO|Company\s*(?:Reg|No)|'
            r'GST\s*(?:Reg|ID|Registration)\s*(?:No)?|'
            r'Licensee\s|Warning:|'
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            cleaned, _re.IGNORECASE,
        ):
            continue
        # Skip lines that are just registration/ID numbers in parens with labels
        if _re.match(r'^[-*]?\s*\(.*(?:REG|Co\.?\s*No|GST).*\)$', cleaned, _re.IGNORECASE):
            continue
        # Stop at URLs, telephone labels, or invoice labels within address
        if _re.match(r'(?:www\.|http|Tel\s*(?:No)?[\s:.]|Fax\s*[\s:.])', cleaned, _re.IGNORECASE):
            break
        if cleaned:
            addr_lines.append(cleaned)
    if addr_lines:
        fields["seller_address"] = "\n".join(addr_lines)

    # total_tax: GST/tax amount — extract FIRST so we can exclude it from total
    for search_text in [plain_with_tables, normalized]:
        # P1: "TAX TOTAL:" or "TAX TOTAL:**" followed by amount (very common)
        tax_m = _re.search(
            r'TAX\s+TOTAL\s*[:\*]*\s*\|?\s*\*?\*?\s*(?:RM|MYR)?\s*([\d,]+\.\d{2})',
            search_text, _re.IGNORECASE,
        )
        if not tax_m:
            # P2: "Total GST", "GST payable (N%)", "Total Tax/SST", "GST AMT/AMOUNT"
            # Exclude "ZERO TAX AMT" (zero-rated amount, not a tax value)
            tax_m = _re.search(
                r'(?:TOTAL\s+(?:GST|TAX|SST)|(?<!ZERO\s)(?:GST|TAX|SST)\s*(?:payable|AMT|AMOUNT))'
                r'\s*(?:\([^)]*\))?\s*[:\.]?\s*\|?\s*'
                r'(?:RM|MYR)?\s*([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P2b: GST Summary/Analysis section — robust parser using line-by-line extraction
            # Matches: "GST Summary", "GST Analysis", "ST Summary" (OCR), "GSI - Rate" (OCR)
            # Also matches: "CODE AMOUNT % TAX/AMT" header pattern
            # Takes LAST decimal number from first tax code row (tax column)
            gst_pos = _re.search(
                r'(?:GST|GSI|ST)\s+(?:Summary|Analysis)|'
                r'(?:GST|GSI)\s*-?\s*Rate|'
                r'CODE\s+AMOUNT\s+%\s+TAX',
                search_text, _re.IGNORECASE,
            )
            if gst_pos:
                after = search_text[gst_pos.end():]
                summary_lines = after.split('\n')[:12]
                for line in summary_lines:
                    if _re.search(r'(?:SR|ZR|ZRL|CR|E|OS|NS|SP|T|S|Z)\s*[^a-zA-Z]', line, _re.IGNORECASE):
                        nums = _re.findall(r'(\d[\d,]*\.\d{2})\b', line)
                        if len(nums) >= 2:
                            # Last number is the tax column
                            fields["total_tax"] = nums[-1].replace(',', '')
                            break
                # Fallback: look for Tax(RM) > Total > Amount(RM) in GST Summary
                # Priority order prevents Amount(RM) (pre-tax total) from overriding Tax(RM)
                if "total_tax" not in fields:
                    _fb_tax = _fb_total = _fb_amount = None
                    for line in summary_lines:
                        m_t = _re.search(r'Tax\s*\(?\s*RM\s*\)?\s*[:\.]?\s*(\d[\d,]*\.\d{2})\b', line, _re.IGNORECASE)
                        if m_t and not _fb_tax:
                            _fb_tax = m_t.group(1).replace(',', '')
                        m_tot = _re.search(r'\*{0,2}Total\*{0,2}\s*[:\.]?\s*(\d[\d,]*\.\d{2})\b', line, _re.IGNORECASE)
                        if m_tot and not _fb_total:
                            _fb_total = m_tot.group(1).replace(',', '')
                        m_a = _re.search(r'Amount\s*\(?\s*RM\s*\)?\s*[:\.]?\s*(\d[\d,]*\.\d{2})\b', line, _re.IGNORECASE)
                        if m_a and not _fb_amount:
                            _fb_amount = m_a.group(1).replace(',', '')
                    chosen = _fb_tax or _fb_total or _fb_amount
                    if chosen:
                        fields["total_tax"] = chosen
                if "total_tax" in fields:
                    break
        if not tax_m:
            # P3: "GST (N.N%)" or "GST @N%:" — standalone, NOT after Inclusive/Sales/Total
            # Allow pipe separators between percentage and amount
            tax_m = _re.search(
                r'(?<!Inclusive\s)(?<!Incl\.\s)(?<!Include\s)(?<!Sales\s)'
                r'(?:GST|TAX|SST)\s*(?:\([\d.]+%\)|@[\d.]+%?)\s*[:\.]?[\s|]*'
                r'(?:RM|MYR|\$)?\s*([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P3b: "GST @6%: $0.46" — with $ prefix
            tax_m = _re.search(
                r'(?<!Inclusive\s)(?<!Incl\.\s)(?<!Include\s)(?<!Sales\s)'
                r'(?:GST|TAX|SST)\s*(?:\([\d.]+%\)|@[\d.]+%?)\s*[:\.]?[\s|]*'
                r'\$\s*([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P4: "GST INCLUSIVE    0.00" or "GST N% INCLUSIVE/INCLUDED    0.00"
            tax_m = _re.search(
                r'(?:GST|TAX|SST)\s+(?:\d+%?\s+)?(?:INCLUSIVE|INCLUDED)\s+(?:RM|MYR)?\s*-?([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P4b: "N% GST Included N.NN" or "GST Included N.NN" or "GST N% Included N.NN"
            tax_m = _re.search(
                r'(?:\d+%\s+)?(?:GST|TAX|SST)\s+(?:\d+%?\s+)?Included\s+(?:RM|MYR|\$)?\s*-?([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P5: "TOTAL INCLUDES N% GST    0.00"
            tax_m = _re.search(
                r'TOTAL\s+INCLUDES?\s+[\d.]+%\s*(?:GST|TAX|SST)\s*[:\.]?[\s|]*'
                r'(?:RM|MYR)?\s*([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P6: "Tax(RM) N.NN" or "Tax (RM) N.NN" — GST Summary blocks
            tax_m = _re.search(
                r'Tax\s*\(?\s*RM\s*\)?\s*[:\.]?[\s|]*([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P6b: Markdown table with "Tax (RM)" column header
            # e.g. | Item Count | Amount (RM) | Tax (RM) |
            #      |    | 9.34    | 0.56    |
            header_m = _re.search(r'Tax\s*\(RM\)\s*\|', search_text, _re.IGNORECASE)
            if header_m:
                # Find column index of Tax (RM) in header
                hdr_line_start = search_text.rfind('\n', 0, header_m.start()) + 1
                hdr_line_end = search_text.find('\n', header_m.end())
                hdr_line = search_text[hdr_line_start:hdr_line_end] if hdr_line_end > 0 else search_text[hdr_line_start:]
                hdr_cols = [c.strip() for c in hdr_line.split('|')]
                tax_col_idx = -1
                for ci, col in enumerate(hdr_cols):
                    if _re.search(r'Tax\s*\(RM\)', col, _re.IGNORECASE):
                        tax_col_idx = ci
                        break
                if tax_col_idx >= 0:
                    # Skip separator row(s), then extract from data row
                    remaining = search_text[hdr_line_end+1:] if hdr_line_end > 0 else ''
                    for data_line in remaining.split('\n')[:4]:
                        if _re.match(r'\s*\|[-\s|]+\|\s*$', data_line):
                            continue  # separator row
                        data_cols = [c.strip() for c in data_line.split('|')]
                        if tax_col_idx < len(data_cols):
                            amt_m = _re.search(r'(\d[\d,]*\.\d{2})', data_cols[tax_col_idx])
                            if amt_m:
                                tax_m = amt_m  # reuse match object for group(1)
                                break
        if not tax_m:
            # P7: bare "GST: N.NN" or "GST: -N.NN" (NOT GST ID/NO/REG/Reg/Summary)
            # Also handles "GST: N.NN RM" (amount before currency suffix)
            # Pipe tolerance for "GST | N.NN" patterns
            tax_m = _re.search(
                r'(?<!\w)(?:GST|GSI|SST)\s*(?!ID|No|NO|REG|Reg|Summary|Summ|Incl|Include)\s*'
                r'[:\.]?[\s|]*(?:RM|MYR|\$)?\s*-?([\d,]+\.\d{2})\b',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P8: "GST N%: RM0.00" or "GST N%   0.00 RM" (without @ or parens)
            # Pipe tolerance for "GST N% | 0.00" patterns
            tax_m = _re.search(
                r'(?<!\w)(?:GST|SST)\s+[\d.]+%\s*[:\.]?[\s|]*(?:RM|MYR|\$)?\s*-?([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if not tax_m:
            # P9: "GST @N% included in total  RMN.NN"
            tax_m = _re.search(
                r'(?:GST|SST)\s*@?\s*[\d.]+%?\s*(?:included?\s+in\s+total)\s*'
                r'(?:RM|MYR|\$)?\s*([\d,]+\.\d{2})',
                search_text, _re.IGNORECASE,
            )
        if tax_m:
            fields["total_tax"] = tax_m.group(1).replace(',', '').strip()
            break

    # total_amount: search in both plain text AND HTML-stripped text
    # Strip markdown bold markers for cleaner matching (** in markdown tables)
    tax_val = fields.get("total_tax", "")
    _TOTAL_PATS = [
        # P1: GRAND TOTAL
        r'GRAND\s+TOTAL',
        # P2: Total Sales/Amount (Inclusive of GST @N%) — NOT Excluding
        (r'TOTAL\s+(?:SALES|AMOUNT|AMT|PAYABLE)\s*'
         r'(?!\s*\(?\s*(?:Excl|Excluding))'
         r'(?:(?:Inclusive|Inc|Incl|Including)?\.?\s*(?:of\s*)?(?:GST|SST|Tax)?\s*(?:@?\s*[\d.]+%?)?)?'),
        # P3: Total (Inclusive of GST) or Total Incl. GST@N%
        (r'TOTAL\s*(?:\(?(?:Inclusive|Inc|Incl|Including)\.?\s*(?:of\s*)?'
         r'(?:GST|SST|Tax)?\s*(?:@?\s*[\d.]+%?)?\)?)'),
        # P4: Total (MYR/RM)
        r'TOTAL\s*\(?\s*(?:MYR|RM)\s*\)?',
        # P5: Gross Amt/Amount/Total
        r'GROSS\s+(?:AMT|AMOUNT|TOTAL)',
        # P6: Rounded Total or Net Total
        r'(?:Rounded|Net)\s+TOTAL\s*(?:\(RM\))?',
        # P7: bare TOTAL (not followed by column-header words or GST/Tax)
        r'(?<!\w)TOTAL(?!\s*(?:GST|TAX|SST|QTY|Qty|Quantity|Excl|Items?|Includes?|Description|Price|Unit))',
        # P8: Sub-total / Subtotal (fallback, some receipts only have subtotal)
        r'SUB[\s\-]*TOTAL',
        # P9: Total (Excluding GST) — use when no inclusive total exists
        r'TOTAL\s*\(?\s*(?:Excl(?:uding)?\.?\s*(?:GST|SST|Tax)?\s*)\)?',
    ]
    for search_text in [plain_with_tables, normalized]:
        # Strip markdown bold/italic markers
        cleaned = _re.sub(r'\*{1,2}', '', search_text)
        for pat in _TOTAL_PATS:
            total_m = _re.search(pat, cleaned, _re.IGNORECASE)
            if total_m:
                # Get rest of line after match + next line (amount may be on next line)
                after = cleaned[total_m.end():]
                rest_lines = after.split('\n')
                rest = rest_lines[0]
                amounts = _re.findall(r'(?:RM|MYR)?\s*([\d,]+\.\d{2})\b', rest)
                # If no amounts on same line, check next non-empty line
                if not amounts and len(rest_lines) > 1:
                    for next_line in rest_lines[1:3]:
                        next_stripped = next_line.strip()
                        if next_stripped:
                            rest = rest + '\n' + next_stripped
                            amounts = _re.findall(r'(?:RM|MYR)?\s*([\d,]+\.\d{2})\b', rest)
                            break
                if amounts:
                    # Table rows (with |): take last number (rightmost column)
                    # Plain text: take first number (immediately after label)
                    is_table = '|' in rest
                    val = (amounts[-1] if is_table else amounts[0]).replace(',', '').strip()
                    # Don't use tax value as total (P7 fallback guard)
                    if val != tax_val or pat != _TOTAL_PATS[-1]:
                        fields["total_amount"] = val
                        break
        if "total_amount" in fields:
            break

    return fields


def _eval_kie(pred: str, gt, metadata: dict) -> dict:
    """Nanonets-KIE: Average ANLS (official NanoNets/docext evaluation).

    NanoNets uses simple average of per-field NLS scores:
    NLS = 1 - edit_distance / max(len(pred), len(gt))
    No threshold, no F1/precision/recall — just mean ANLS.

    Fallback: when JSON parsing fails (model outputs raw OCR text instead of
    structured JSON), extract fields via regex patterns.
    """
    pred_fields = _parse_json_fields(pred)
    gt_fields = gt if isinstance(gt, dict) else {}
    # Fallback: JSON parsing failed OR parsed keys don't overlap with GT fields
    # (e.g. json_repair produces garbage keys like 'Adjustment' from raw text)
    has_overlap = bool(set(pred_fields.keys()) & set(gt_fields.keys())) if pred_fields and gt_fields else False
    if (not pred_fields or not has_overlap) and pred.strip() and gt_fields:
        regex_fields = _extract_kie_fields_from_text(pred)
        if not has_overlap or len(set(regex_fields.keys()) & set(gt_fields.keys())) > len(set(pred_fields.keys()) & set(gt_fields.keys())):
            pred_fields = regex_fields
    # Normalize whitespace around punctuation for fairer comparison
    # GT annotations have inconsistent spacing (e.g., "1851 - A" vs "1851-A")
    import re as _re_kie
    def _norm_ws(s: str) -> str:
        s = _re_kie.sub(r'\s*,\s*', ', ', s)
        s = _re_kie.sub(r'\s*-\s*', '-', s)
        s = _re_kie.sub(r'\s*\.\s*', '.', s)
        return _re_kie.sub(r'\s+', ' ', s).strip()
    pred_fields = {k: _norm_ws(v) if isinstance(v, str) else v for k, v in pred_fields.items()}
    gt_fields = {k: _norm_ws(v) if isinstance(v, str) else v for k, v in gt_fields.items()}
    return {"anls": compute_kie_anls(pred_fields, gt_fields)}


def _normalize_iam_spacing(text: str) -> str:
    """Normalize IAM tokenization artifacts.

    The IAM dataset inserts spaces before punctuation (e.g. "word ." instead of
    "word."). Modern OCR models produce standard punctuation without these
    spaces. Normalizing both sides prevents unfair penalization.
    """
    import re as _re_iam
    # Remove spaces before common punctuation
    text = _re_iam.sub(r' ([.,!?;:])', r'\1', text)
    # Normalize spaces around quotes
    text = _re_iam.sub(r'" ', '"', text)
    text = _re_iam.sub(r' "', '"', text)
    # Normalize spaces inside parens
    text = _re_iam.sub(r'\( ', '(', text)
    text = _re_iam.sub(r' \)', ')', text)
    return text


def _eval_handwritten(pred: str, gt, metadata: dict) -> dict:
    """Handwritten-Forms: CER + WER (standard IAM evaluation).

    Standard IAM metrics normalize by ground truth length:
    CER = edit_distance(pred, gt) / len(gt)
    WER = word_edit_distance(pred, gt) / num_gt_words
    Lower = better.

    Applies IAM spacing normalization to remove tokenization artifacts
    (spaces before punctuation) from both GT and prediction.
    """
    gt_str = gt if isinstance(gt, str) else str(gt)
    pred_n = _normalize_iam_spacing(pred)
    gt_n = _normalize_iam_spacing(gt_str)
    return {
        "cer": compute_cer(pred_n, gt_n),
        "wer": compute_wer(pred_n, gt_n),
    }


# ─── Aggregation ──────────────────────────────────────────────────────

def _aggregate_metrics(per_sample: list[dict], metric_type: str) -> dict[str, float]:
    """Compute mean of each score across all samples + latency stats."""
    if not per_sample:
        return {}

    all_scores = [r["scores"] for r in per_sample if r["scores"]]
    if not all_scores:
        return {}

    # Collect ALL keys across all samples (not just the first one)
    keys: set[str] = set()
    for s in all_scores:
        keys.update(s.keys())

    aggregate = {}
    for key in sorted(keys):
        vals = [s[key] for s in all_scores if key in s and isinstance(s[key], (int, float))]
        if vals:
            aggregate[f"mean_{key}"] = sum(vals) / len(vals)

    # Add error rate
    error_count = sum(1 for r in per_sample if r.get("error"))
    aggregate["error_rate"] = error_count / len(per_sample)

    # Latency stats (ms)
    latencies = sorted(
        r["latency_ms"] for r in per_sample if isinstance(r.get("latency_ms"), (int, float))
    )
    if latencies:
        n = len(latencies)
        aggregate["latency_mean_ms"] = sum(latencies) / n
        aggregate["latency_median_ms"] = latencies[n // 2]
        aggregate["latency_p90_ms"] = latencies[int(n * 0.9)]
        aggregate["latency_p99_ms"] = latencies[int(n * 0.99)]
        aggregate["latency_min_ms"] = latencies[0]
        aggregate["latency_max_ms"] = latencies[-1]
        aggregate["throughput_samples_per_sec"] = n / (sum(latencies) / 1000) if sum(latencies) > 0 else 0.0

    return aggregate


# ─── Helpers ──────────────────────────────────────────────────────────


def _check_server_health(client) -> bool:
    """Quick check if vLLM server is still responding."""
    import urllib.request
    try:
        base = client.client.base_url
        url = f"{base.scheme}://{base.host}:{base.port}/v1/models"
        req = urllib.request.Request(url)
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False


def _safe_name(name: str) -> str:
    """Convert model name to filesystem-safe string."""
    return name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _save_checkpoint(path: Path, model_name: str, benchmark_key: str, results: list[dict]) -> None:
    """Save intermediate checkpoint."""
    data = {
        "model_name": model_name,
        "benchmark_key": benchmark_key,
        "per_sample_results": results,
        "n_completed": len(results),
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def _parse_json_fields(text: str) -> dict:
    """Extract JSON dict from model output using json_repair (fault-tolerant).

    Matches docext/benchmark/benchmark.py _parse_response() which uses
    json_repair.repair_json() for robust JSON extraction.
    """
    import json_repair

    text = text.strip()
    try:
        parsed = json_repair.repair_json(text, ensure_ascii=False, return_objects=True)
    except RecursionError:
        return {}

    # Handle list case: merge dicts (same as docext benchmark.py:638-651)
    if isinstance(parsed, list):
        merged = {}
        for item in parsed:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key not in merged:
                        merged[key] = value
                    else:
                        if isinstance(merged[key], list):
                            merged[key].append(value)
                        else:
                            merged[key] = [merged[key], value]
        return merged
    if isinstance(parsed, dict):
        return parsed
    if parsed == "":
        return {}
    return {}


def _extract_tables_from_pred(pred: str) -> list[str]:
    """Extract table HTML from prediction text.

    Handles three formats:
    1. HTML tables: <table>...</table>
    2. Markdown pipe tables: | col1 | col2 |
    3. PaddleOCR format: <fcel>...<nl>...
    Returns list of HTML table strings.
    """
    import re as _re
    tables = []

    # 1. Extract HTML tables directly
    html_tables = _re.findall(r'<table[^>]*>.*?</table>', pred, _re.DOTALL | _re.IGNORECASE)
    tables.extend(html_tables)

    # 2. Extract Markdown pipe tables (with or without leading pipes)
    lines = pred.splitlines()
    md_block: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Standard: | col1 | col2 |  OR  col1 | col2 | col3
        is_pipe_line = (
            (stripped.startswith("|") and "|" in stripped[1:])
            or (_re.match(r'^[^|]+\|.+\|', stripped) and stripped.count("|") >= 2)
            or _re.match(r'^[\s|:-]+$', stripped) and "|" in stripped  # separator row
        )
        if is_pipe_line:
            md_block.append(stripped)
        else:
            if len(md_block) >= 2:  # At least header + 1 row
                html = _convert_markdown_table_to_html("\n".join(md_block))
                # Filter degenerate tables (e.g. "||" + "| :- |")
                if html.count('<td>') >= 2:
                    tables.append(html)
            md_block = []
    if len(md_block) >= 2:
        html = _convert_markdown_table_to_html("\n".join(md_block))
        if html.count('<td>') >= 2:
            tables.append(html)

    # 3. LaTeX array/tabular tables: \begin{array}...\end{array}
    for env in ('array', 'tabular'):
        pattern = _re.compile(
            r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}',
            _re.DOTALL
        )
        for m in pattern.finditer(pred):
            html = _convert_latex_table_to_html(m.group())
            if html:
                tables.append(html)

    # 4. PaddleOCR format
    if "<fcel>" in pred:
        tables.append(_convert_paddle_table_to_html(pred))

    return tables


def _convert_latex_table_to_html(latex: str) -> str:
    """Convert LaTeX array/tabular table to HTML.

    Handles: \\begin{array}{|l|c|c|}...\\end{array}
    Rows separated by \\\\, cells by &, \\hline ignored.
    \\text{...} unwrapped to plain text.
    """
    import re as _re
    # Remove \begin{array/tabular}{...} and \end{...}
    body = _re.sub(r'\\begin\{(?:array|tabular)\}\{[^}]*\}', '', latex)
    body = _re.sub(r'\\end\{(?:array|tabular)\}', '', body)
    # Split into rows by \\
    raw_rows = _re.split(r'\\\\', body)
    html_rows = []
    for row in raw_rows:
        row = row.strip()
        if not row or row == r'\hline':
            continue
        # Remove \hline at start/end of row
        row = row.replace(r'\hline', '').strip()
        if not row:
            continue
        # Split by & to get cells
        cells = row.split('&')
        cell_htmls = []
        for cell in cells:
            cell = cell.strip()
            # Unwrap \text{...}
            cell = _re.sub(r'\\text\{([^}]*)\}', r'\1', cell)
            # Unwrap \textbf{...}
            cell = _re.sub(r'\\textbf\{([^}]*)\}', r'\1', cell)
            # Strip markdown bold markers **...**
            cell = cell.replace('**', '')
            # Convert LaTeX escapes: \% → %, \& → &, \$ → $
            cell = cell.replace('\\%', '%').replace('\\&', '&').replace('\\$', '$')
            cell = cell.strip()
            cell_htmls.append(f"<td>{cell}</td>")
        html_rows.append("<tr>" + "".join(cell_htmls) + "</tr>")
    if not html_rows:
        return ""
    return "<table>" + "".join(html_rows) + "</table>"


def _convert_paddle_table_to_html(text: str) -> str:
    """Convert PaddleOCR-VL table format (<fcel>, <ecel>, <lcel>, <ucel>, <nl>) to HTML.

    Markers:
    - <fcel>: filled cell (has content)
    - <ecel>: empty cell
    - <lcel>: left-merge (colspan with the cell to the left)
    - <ucel>: up-merge (rowspan with the cell above)
    - <nl>: new row
    """
    import re as _re
    rows = text.split("<nl>")
    # First pass: parse into a grid of (type, content) tuples
    grid: list[list[tuple[str, str]]] = []
    for row in rows:
        row_cells: list[tuple[str, str]] = []
        # Split on all marker types, capturing the marker
        tokens = _re.split(r"(<fcel>|<ecel>|<lcel>|<ucel>)", row)
        marker = None
        for token in tokens:
            if token in ("<fcel>", "<ecel>", "<lcel>", "<ucel>"):
                marker = token
            elif marker is not None:
                row_cells.append((marker, token.strip()))
                marker = None
        if row_cells:
            grid.append(row_cells)

    if not grid:
        return text

    # Second pass: build HTML with colspan/rowspan
    # Track rowspan carry-overs: col_idx → remaining_rowspan
    num_cols = max(len(r) for r in grid)
    rowspan_left: list[int] = [0] * num_cols  # remaining rowspan per column

    html_rows = []
    for row_cells in grid:
        cells_html: list[str] = []
        col_idx = 0
        cell_idx = 0
        while cell_idx < len(row_cells):
            # Skip columns occupied by rowspan from above
            while col_idx < num_cols and rowspan_left[col_idx] > 0:
                rowspan_left[col_idx] -= 1
                col_idx += 1

            marker, content = row_cells[cell_idx]
            cell_idx += 1

            if marker == "<ucel>":
                # This cell is merged upward — skip it (already covered by rowspan)
                col_idx += 1
                continue

            if marker == "<lcel>":
                # This cell merges left — should have been consumed by colspan counting
                col_idx += 1
                continue

            # Count consecutive <lcel> after this cell for colspan
            colspan = 1
            while cell_idx < len(row_cells) and row_cells[cell_idx][0] == "<lcel>":
                colspan += 1
                cell_idx += 1

            # Count consecutive <ucel> in same column below for rowspan
            # (We scan the grid vertically from the current row)
            rowspan = 1
            cur_row_idx = grid.index(row_cells)
            for below_row in grid[cur_row_idx + 1:]:
                if col_idx < len(below_row) and below_row[col_idx][0] == "<ucel>":
                    rowspan += 1
                else:
                    break

            # Set rowspan tracking for spanned columns
            if rowspan > 1:
                for c in range(col_idx, min(col_idx + colspan, num_cols)):
                    rowspan_left[c] = rowspan - 1

            attrs = ""
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            cells_html.append(f"<td{attrs}>{content}</td>")
            col_idx += colspan

        if cells_html:
            html_rows.append("<tr>" + "".join(cells_html) + "</tr>")

    return "<table>" + "".join(html_rows) + "</table>"


def _convert_space_table_to_html(text: str) -> str:
    """Convert space-separated table text to HTML.

    Fallback for predictions that use 4+ spaces between columns instead of
    pipes or HTML tags. Skips markdown headings and bullet points.
    """
    import re as _re
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        cells = _re.split(r'\s{4,}', line)
        cells = [c.strip() for c in cells if c.strip()]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return text
    html = '<table>'
    for row in rows:
        html += '<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>'
    html += '</table>'
    return html


def _convert_markdown_table_to_html(text: str) -> str:
    """Convert Markdown pipe-table to HTML <table>.

    Handles both standard (| col | col |) and bare (col | col | col) formats.
    Skips separator rows (e.g. |---|---|).
    """
    import re as _re
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    # Keep lines with pipes (table rows)
    pipe_lines = [l for l in lines if "|" in l]
    # Filter out separator rows (e.g. |---|---| or ---|---|---)
    data_lines = [l for l in pipe_lines if not _re.match(r"^[\s|:-]+$", l)]
    if not data_lines:
        return text
    html_rows = []
    for line in data_lines:
        # Strip leading/trailing pipe
        line = line.strip("|")
        cells = [c.strip() for c in line.split("|")]
        tag = "td"
        row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
        html_rows.append(f"<tr>{row}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"
