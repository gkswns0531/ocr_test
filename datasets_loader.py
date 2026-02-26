"""Dataset loaders for 8 OCR benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from config import DATA_CACHE_DIR


@dataclass
class BenchmarkSample:
    image: Image.Image | None
    ground_truth: str | dict | list
    metadata: dict = field(default_factory=dict)
    sample_id: str = ""
    image_path: str | None = None  # lazy loading: store path, load on demand


# ─── 1. OmniDocBench ──────────────────────────────────────────────────

# Official OmniDocBench category types for plain text evaluation
# From OmniDocBench/dataset/end2end_dataset.py:process_get_matched_elements()
_OMNIDOC_PLAIN_TEXT_CATEGORIES = {
    'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference',
    'equation_caption', 'figure_caption', 'figure_footnote', 'table_caption',
    'table_footnote', 'code_algorithm', 'code_algorithm_caption',
    'header', 'footer', 'page_footnote', 'page_number',
}
# Categories filtered out during text metric evaluation (matched but ignored)
# From OmniDocBench/dataset/end2end_dataset.py:filtered_out_ignore()
_OMNIDOC_IGNORE_CATEGORIES = {
    'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote',
    'code_algorithm', 'code_algorithm_caption', 'header', 'footer',
    'page_footnote', 'page_number', 'equation_caption',
}
_OMNIDOC_EVAL_TEXT_CATEGORIES = _OMNIDOC_PLAIN_TEXT_CATEGORIES - _OMNIDOC_IGNORE_CATEGORIES


def load_omnidocbench(max_samples: int | None = None) -> list[BenchmarkSample]:
    """Load OmniDocBench v1.5.

    Images from HF dataset, GT annotations from OmniDocBench.json.
    Stores full annotation dict as ground_truth for element-wise evaluation
    using the official OmniDocBench protocol (md_tex_filter + match_gt2pred_quick).
    """
    # Download annotation JSON
    anno_path = hf_hub_download(
        "opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset"
    )
    with open(anno_path) as f:
        annotations = json.load(f)

    # Load images
    ds = load_dataset("opendatalab/OmniDocBench", split="train", cache_dir=str(DATA_CACHE_DIR))

    # Build filename-stem → HF dataset index mapping.
    # HF dataset has 1358 images (including 3 meta images like data_diversity.png)
    # while annotations JSON has 1355 entries. Order differs between the two,
    # and HF uses .png while annotations use .jpg, so we align by stem (no extension).
    hf_stem_to_idx: dict[str, int] = {}
    for idx in range(len(ds)):
        img = ds[idx]["image"]
        if hasattr(img, "filename") and img.filename:
            stem = Path(img.filename).stem
            hf_stem_to_idx[stem] = idx

    n = len(annotations)
    if max_samples is not None:
        n = min(n, max_samples)

    samples = []
    for i in range(n):
        anno = annotations[i]
        # Look up the matching HF image by annotation's image_path stem
        anno_image_path = anno.get("page_info", {}).get("image_path", "")
        anno_stem = Path(anno_image_path).stem if anno_image_path else ""

        if anno_stem and hf_stem_to_idx:
            hf_idx = hf_stem_to_idx.get(anno_stem)
            if hf_idx is None:
                continue  # No matching image in HF dataset (e.g. meta images)
        else:
            hf_idx = i  # Fallback to sequential alignment
            if hf_idx >= len(ds):
                break

        image = _ensure_pil(ds[hf_idx].get("image"))
        if image is None:
            continue
        # Store full annotation as ground_truth for official element-wise evaluation
        ground_truth = {
            "annotation": anno,
            "annotation_index": i,
        }
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=ground_truth,
            metadata={
                "index": i,
                "hf_index": hf_idx,
                "image_filename": anno_stem,
                "page_info": anno.get("page_info", {}),
            },
            sample_id=f"omnidocbench_{i}",
        ))
    return samples


# ─── 2. Upstage DP-Bench ──────────────────────────────────────────────

def load_upstage_dp_bench(max_samples: int | None = None) -> list[BenchmarkSample]:
    """Load Upstage DP-Bench dataset.

    PDFs + reference.json with element-level annotations.
    Converts PDF pages to images for OCR evaluation.
    """
    import fitz  # PyMuPDF

    # Download reference JSON
    ref_path = hf_hub_download(
        "upstage/dp-bench", "dataset/reference.json", repo_type="dataset"
    )
    with open(ref_path) as f:
        reference = json.load(f)

    samples = []
    pdf_names = sorted(reference.keys())
    if max_samples is not None:
        pdf_names = pdf_names[:max_samples]

    for pdf_name in pdf_names:
        # Download PDF
        try:
            pdf_path = hf_hub_download(
                "upstage/dp-bench", f"dataset/pdfs/{pdf_name}", repo_type="dataset"
            )
        except Exception:
            continue

        # Convert first page to image
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        except Exception:
            continue

        gt = reference[pdf_name]
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=gt,
            metadata={"pdf_name": pdf_name},
            sample_id=f"dpbench_{pdf_name}",
        ))
    return samples


# ─── 3. OCRBench ──────────────────────────────────────────────────────

def load_ocrbench(max_samples: int | None = 100) -> list[BenchmarkSample]:
    """Load OCRBench with its own question/answer pairs."""
    ds = load_dataset("echo840/OCRBench", split="test", cache_dir=str(DATA_CACHE_DIR))
    samples = []
    for i, row in enumerate(_take_ds(ds, max_samples)):
        image = _ensure_pil(row.get("image"))
        if image is None:
            continue
        question = row.get("question", "Read the text in this image.")
        answer = row.get("answer", "")
        if isinstance(answer, str):
            answer = [answer]
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=answer,
            metadata={
                "index": i,
                "question": question,
                "dataset": row.get("dataset", ""),
                "type": row.get("question_type", ""),
            },
            sample_id=f"ocrbench_{i}",
        ))
    return samples


# ─── 4. UniMERNet ─────────────────────────────────────────────────────

def load_unimernet(max_samples: int | None = 100) -> list[BenchmarkSample]:
    """Load UniMER Dataset for formula recognition (streaming to avoid 2.1GB full load)."""
    ds = load_dataset("deepcopy/UniMER", split="test", streaming=True)
    samples = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        image = _ensure_pil(row.get("image"))
        if image is None:
            continue
        gt = row.get("text", "")
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=gt,
            metadata={"index": i},
            sample_id=f"unimernet_{i}",
        ))
    return samples


# ─── 5. PubTabNet ─────────────────────────────────────────────────────

def load_pubtabnet(max_samples: int | None = 100) -> list[BenchmarkSample]:
    """Load PubTabNet for table recognition (streaming to avoid 12.7GB full load)."""
    ds = load_dataset("apoidea/pubtabnet-html", split="validation", streaming=True)
    samples = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        image = _ensure_pil(row.get("image"))
        if image is None:
            continue
        gt = row.get("html_table", row.get("html", ""))
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=gt,
            metadata={"index": i, "imgid": row.get("imgid", "")},
            sample_id=f"pubtabnet_{i}",
        ))
    return samples


# ─── 6. TEDS TEST ─────────────────────────────────────────────────────

def load_teds_test(max_samples: int | None = 100) -> list[BenchmarkSample]:
    """Load PubTabNet validation subset for TEDS evaluation (streaming).

    Skips first 4000 to avoid overlap with pubtabnet benchmark.
    """
    ds = load_dataset("apoidea/pubtabnet-html", split="validation", streaming=True)
    offset = 4000
    samples = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if max_samples is not None and len(samples) >= max_samples:
            break
        image = _ensure_pil(row.get("image"))
        if image is None:
            continue
        gt = row.get("html_table", row.get("html", ""))
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=gt,
            metadata={"index": i, "offset": offset, "imgid": row.get("imgid", "")},
            sample_id=f"teds_test_{i}",
        ))
    return samples


# ─── 7. Nanonets KIE ──────────────────────────────────────────────────

def load_nanonets_kie(max_samples: int | None = 100) -> list[BenchmarkSample]:
    """Load Nanonets KIE benchmark for key information extraction.

    Matches docext/benchmark/vlm_datasets/nanonets_kie.py:
    - Dataset: nanonets/key_information_extraction (test split)
    - GT fields are in 'annotations' dict with flat string values
    """
    ds = load_dataset("nanonets/key_information_extraction", split="test", cache_dir=str(DATA_CACHE_DIR))
    samples = []
    for i, row in enumerate(_take_ds(ds, max_samples)):
        image = _ensure_pil(row.get("image"))
        if image is None:
            continue
        # GT is directly in 'annotations' dict (flat key→string mapping)
        gt_fields = row.get("annotations", {})
        if not isinstance(gt_fields, dict):
            gt_fields = {}
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=gt_fields,
            metadata={"index": i},
            sample_id=f"nanonets_kie_{i}",
        ))
    return samples


# ─── 8. Handwritten Forms (IAM-line) ──────────────────────────────────

def load_handwritten_forms(max_samples: int | None = 100) -> list[BenchmarkSample]:
    """Load IAM-line dataset for handwritten text recognition."""
    ds = load_dataset("Teklia/IAM-line", split="test", cache_dir=str(DATA_CACHE_DIR))
    samples = []
    for i, row in enumerate(_take_ds(ds, max_samples)):
        image = _ensure_pil(row.get("image"))
        if image is None:
            continue
        gt = row.get("text", "")
        samples.append(BenchmarkSample(
            image=image,
            ground_truth=gt,
            metadata={"index": i},
            sample_id=f"handwritten_{i}",
        ))
    return samples


# ─── Helpers ───────────────────────────────────────────────────────────

DEFAULT_SEED = 42


def _take_ds(ds, n: int | None, seed: int | None = None):
    """Take n items from a HuggingFace dataset with reproducible random sampling.

    Uses DEFAULT_SEED (module-level, settable via CLI) for reproducibility.
    The same subset is selected across all models for fair comparison.
    """
    if n is None:
        return ds
    total = len(ds)
    n = min(n, total)
    if n == total:
        return ds
    # Reproducible random indices
    import random
    actual_seed = seed if seed is not None else DEFAULT_SEED
    rng = random.Random(actual_seed)
    indices = sorted(rng.sample(range(total), n))
    return ds.select(indices)


def _ensure_pil(img) -> Image.Image | None:
    """Ensure the value is a PIL Image, converting if necessary."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        import io
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:
            return None
    return None


# ─── Loader Registry ──────────────────────────────────────────────────

LOADERS = {
    "omnidocbench": load_omnidocbench,
    "upstage_dp_bench": load_upstage_dp_bench,
    "ocrbench": load_ocrbench,
    "unimernet": load_unimernet,
    "pubtabnet": load_pubtabnet,
    "teds_test": load_teds_test,
    "nanonets_kie": load_nanonets_kie,
    "handwritten_forms": load_handwritten_forms,
}
