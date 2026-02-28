#!/usr/bin/env python3
"""Compute additional embeddings for full comparison matrix.

New embeddings:
  - pure_text: OCR text WITHOUT figure_title/vision_footnote regions
  - caption_text: GPT-5-mini generated captions for image/chart regions

Computes for both 2B and 8B models, then evaluates all combinations.

Usage:
    python3 run_comparison_embeddings.py [--batch-text 256]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_b200_pipeline import (
    VLEmbeddingModel,
    EmbeddingModelConfig,
    linearize_html_table,
    _release_memory,
    QUERY_INSTRUCTION,
    OCR_RESULTS_FILE,
    OUTPUT_DIR,
    MAX_TOKENS,
    CHARS_PER_TOKEN,
)

CAPTIONS_FILE = OUTPUT_DIR / "region_captions.jsonl"
EMBEDDINGS_2B_DIR = OUTPUT_DIR / "embeddings"
EMBEDDINGS_8B_DIR = OUTPUT_DIR / "embeddings_8b"

MODEL_2B_ID = "Forturne/Qwen3-VL-Embedding-2B-FP8"
MODEL_8B_ID = "Forturne/Qwen3-VL-Embedding-8B-FP8"


def extract_pure_texts(ocr_results_path: Path) -> list[tuple[str, str]]:
    """Extract page texts WITHOUT figure_title/vision_footnote regions.

    Returns: [(page_id, pure_text), ...]
    """
    results: list[tuple[str, str]] = []
    max_chars = int(MAX_TOKENS * CHARS_PER_TOKEN)

    with open(ocr_results_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_id = rec["page_id"]
            regions = rec.get("regions", [])
            parts: list[str] = []

            for r in regions:
                native_label = r.get("native_label", "")
                if native_label in ("figure_title", "vision_footnote"):
                    continue

                content = r.get("content")
                if not content:
                    continue

                label = r.get("label", "")
                if label == "table":
                    linearized = linearize_html_table(content)
                    if linearized.strip():
                        parts.append(linearized)
                elif label in ("display_formula", "inline_formula", "formula"):
                    parts.append(content.strip())
                else:
                    text = content.strip()
                    if text:
                        parts.append(text)

            merged = "\n\n".join(parts)
            if len(merged) > max_chars:
                merged = merged[:max_chars]
                last_break = max(merged.rfind("\n\n"), merged.rfind("\n"), merged.rfind(". "))
                if last_break > max_chars * 0.8:
                    merged = merged[:last_break]

            results.append((page_id, merged))

    return results


def load_caption_texts() -> list[tuple[str, str]]:
    """Load GPT-5-mini captions grouped by page_id.

    Returns: [(page_id, concatenated_captions), ...] for pages that have captions.
    """
    page_captions: dict[str, list[str]] = {}

    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            caption = rec.get("gpt5_caption", "").strip()
            if not caption:
                continue
            page_id = rec["page_id"]
            if page_id not in page_captions:
                page_captions[page_id] = []
            page_captions[page_id].append(caption)

    results: list[tuple[str, str]] = []
    for page_id, captions in page_captions.items():
        results.append((page_id, "\n\n".join(captions)))

    return results


def compute_embeddings_for_model(
    model_name: str,
    model_id: str,
    emb_dir: Path,
    pure_texts: list[tuple[str, str]],
    caption_items: list[tuple[str, str]],
    batch_text: int,
    gpu_mem: float,
) -> None:
    """Compute pure_text and caption_text embeddings for a given model."""

    pure_text_path = emb_dir / "corpus_pure_text.npy"
    pure_text_pids_path = emb_dir / "pure_text_page_ids.json"
    caption_emb_path = emb_dir / "corpus_caption_text.npy"
    caption_pids_path = emb_dir / "caption_page_ids.json"

    need_pure = not pure_text_path.exists()
    need_caption = not caption_emb_path.exists()

    if not need_pure and not need_caption:
        print(f"  [{model_name}] All embeddings exist, skipping")
        return

    model_cfg = EmbeddingModelConfig(
        name=model_name,
        model_id=model_id,
        batch_size_image=128,
        batch_size_text=batch_text,
        quantization=None,
        gpu_memory_utilization=gpu_mem,
    )
    model = VLEmbeddingModel(model_cfg)

    # 1. Pure text embeddings
    if need_pure:
        print(f"  [{model_name}] Computing pure text embeddings ({len(pure_texts)} pages)...")
        texts = [t for _, t in pure_texts]
        pids = [pid for pid, _ in pure_texts]
        embs = model.encode_texts(texts, batch_size=batch_text, instruction="")
        np.save(str(pure_text_path), embs)
        with open(pure_text_pids_path, "w") as f:
            json.dump(pids, f)
        print(f"  [{model_name}] Saved: {pure_text_path} (shape={embs.shape})")
        del embs
        _release_memory()
    else:
        print(f"  [{model_name}] Skipping pure text (exists)")

    # 2. Caption text embeddings
    if need_caption:
        print(f"  [{model_name}] Computing caption text embeddings ({len(caption_items)} pages)...")
        texts = [t for _, t in caption_items]
        pids = [pid for pid, _ in caption_items]
        embs = model.encode_texts(texts, batch_size=batch_text, instruction="")
        np.save(str(caption_emb_path), embs)
        with open(caption_pids_path, "w") as f:
            json.dump(pids, f)
        print(f"  [{model_name}] Saved: {caption_emb_path} (shape={embs.shape})")
        del embs
        _release_memory()
    else:
        print(f"  [{model_name}] Skipping caption text (exists)")

    del model
    _release_memory()


def evaluate_all() -> None:
    """Run full comparison matrix evaluation."""
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("Full Comparison Matrix: R@K Evaluation")
    print("=" * 70)

    qa = load_dataset("SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-QA", split="test", cache_dir="data")
    ann = load_dataset("SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-annotations", split="test", cache_dir="data")

    idx_to_info: dict[int, tuple[str, int]] = {}
    for row in ann:
        for offset, ci in enumerate(row["page_indices"]):
            idx_to_info[ci] = (row["file_id"], offset)

    jsonl_pids: list[str] = []
    with open(OCR_RESULTS_FILE) as f:
        for line in f:
            jsonl_pids.append(json.loads(line)["page_id"])

    configs = [
        ("2B", EMBEDDINGS_2B_DIR),
        ("8B", EMBEDDINGS_8B_DIR),
    ]

    for model_label, emb_dir in configs:
        queries = np.load(str(emb_dir / "queries.npy"))
        print(f"\n{'='*70}")
        print(f"  {model_label} Model (dim={queries.shape[1]})")
        print(f"{'='*70}")

        # Load all corpus embeddings
        corpus_types: dict[str, tuple[np.ndarray, list[str]]] = {}

        # OCR text (with captions)
        ct_path = emb_dir / "corpus_ocr_text.npy"
        if ct_path.exists():
            corpus_types["text"] = (np.load(str(ct_path)), jsonl_pids)

        # Pure text (without captions)
        pt_path = emb_dir / "corpus_pure_text.npy"
        if pt_path.exists():
            with open(emb_dir / "pure_text_page_ids.json") as f:
                pt_pids = json.load(f)
            corpus_types["pure_text"] = (np.load(str(pt_path)), pt_pids)

        # Region multimodal
        rg_path = emb_dir / "corpus_regions.npy"
        rg_meta_path = emb_dir / "region_metadata.jsonl"
        if rg_path.exists() and rg_meta_path.exists():
            rg_meta = []
            with open(rg_meta_path) as f:
                for line in f:
                    rg_meta.append(json.loads(line))
            rg_pids = [r["page_id"] for r in rg_meta]
            corpus_types["region_mm"] = (np.load(str(rg_path)), rg_pids)

        # Caption text
        cap_path = emb_dir / "corpus_caption_text.npy"
        if cap_path.exists():
            with open(emb_dir / "caption_page_ids.json") as f:
                cap_pids = json.load(f)
            corpus_types["caption_text"] = (np.load(str(cap_path)), cap_pids)

        # Define retrieval configurations
        retrieval_configs: list[tuple[str, list[str]]] = [
            ("text-only", ["text"]),
            ("pure-text-only", ["pure_text"]),
            ("text+region", ["text", "region_mm"]),
            ("pure+region", ["pure_text", "region_mm"]),
            ("text+caption", ["text", "caption_text"]),
            ("pure+caption", ["pure_text", "caption_text"]),
            ("text+region+caption", ["text", "region_mm", "caption_text"]),
            ("pure+region+caption", ["pure_text", "region_mm", "caption_text"]),
        ]

        # Filter to available configs
        retrieval_configs = [(name, keys) for name, keys in retrieval_configs if all(k in corpus_types for k in keys)]

        # R@K for each config
        header = f"{'K':>4}"
        for name, _ in retrieval_configs:
            header += f" | {name:>20}"
        print(f"\n{header}")
        print("-" * len(header))

        for K in [1, 5, 10, 20]:
            row = f"R@{K:>2}"
            for name, keys in retrieval_configs:
                # Build combined corpus
                combined_emb = np.vstack([corpus_types[k][0] for k in keys])
                combined_pids = []
                for k in keys:
                    combined_pids.extend(corpus_types[k][1])

                sim = queries @ combined_emb.T
                correct = 0

                for qi in range(len(qa)):
                    gt = qa[qi]["ground_truth"]
                    expected = set()
                    for gi in gt:
                        if gi in idx_to_info:
                            fid, off = idx_to_info[gi]
                            expected.add(f"{fid}_{off+1}")

                    sorted_idx = np.argsort(sim[qi])[::-1]
                    seen: set[str] = set()
                    top_pids: set[str] = set()
                    for idx in sorted_idx:
                        pid = combined_pids[idx]
                        if pid not in seen:
                            seen.add(pid)
                            top_pids.add(pid)
                            if len(top_pids) >= K:
                                break
                    if expected & top_pids:
                        correct += 1

                n = len(qa)
                row += f" | {100*correct/n:>18.1f}%"
            print(row)

        # Per-type R@10 for key configs
        print(f"\n  Per-type R@10 ({model_label}):")
        key_configs = [(name, keys) for name, keys in retrieval_configs
                       if name in ("text-only", "pure-text-only", "text+region", "pure+region",
                                   "text+region+caption", "pure+region+caption")]

        type_header = f"  {'type':>10}"
        for name, _ in key_configs:
            type_header += f" | {name:>20}"
        print(type_header)
        print("  " + "-" * (len(type_header) - 2))

        for qtype in sorted(set(qa[qi].get("type", "unknown") for qi in range(len(qa)))):
            type_row = f"  {qtype:>10}"
            for name, keys in key_configs:
                combined_emb = np.vstack([corpus_types[k][0] for k in keys])
                combined_pids = []
                for k in keys:
                    combined_pids.extend(corpus_types[k][1])

                sim = queries @ combined_emb.T
                correct = 0
                total = 0

                for qi in range(len(qa)):
                    if qa[qi].get("type", "unknown") != qtype:
                        continue
                    total += 1
                    gt = qa[qi]["ground_truth"]
                    expected = set()
                    for gi in gt:
                        if gi in idx_to_info:
                            fid, off = idx_to_info[gi]
                            expected.add(f"{fid}_{off+1}")

                    sorted_idx = np.argsort(sim[qi])[::-1]
                    seen: set[str] = set()
                    top_pids: set[str] = set()
                    for idx in sorted_idx:
                        pid = combined_pids[idx]
                        if pid not in seen:
                            seen.add(pid)
                            top_pids.add(pid)
                            if len(top_pids) >= 10:
                                break
                    if expected & top_pids:
                        correct += 1

                type_row += f" | {100*correct/total:>18.1f}%"
            print(type_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-text", type=int, default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("Comparison Embeddings: Pure Text + Caption Text")
    print("=" * 70)

    # 1. Extract pure texts
    print("\n  Extracting pure texts (no figure_title/vision_footnote)...")
    pure_texts = extract_pure_texts(OCR_RESULTS_FILE)
    print(f"  Pure texts: {len(pure_texts)} pages")

    # 2. Load caption texts
    print("  Loading GPT-5-mini captions...")
    caption_items = load_caption_texts()
    print(f"  Caption texts: {len(caption_items)} pages with captions")

    # 3. Compute 2B embeddings
    print(f"\n{'='*70}")
    print("Computing 2B embeddings")
    print(f"{'='*70}")
    compute_embeddings_for_model(
        model_name="qwen3-vl-2b",
        model_id=MODEL_2B_ID,
        emb_dir=EMBEDDINGS_2B_DIR,
        pure_texts=pure_texts,
        caption_items=caption_items,
        batch_text=args.batch_text,
        gpu_mem=0.85,
    )

    # 4. Compute 8B embeddings
    print(f"\n{'='*70}")
    print("Computing 8B embeddings")
    print(f"{'='*70}")
    compute_embeddings_for_model(
        model_name="qwen3-vl-8b",
        model_id=MODEL_8B_ID,
        emb_dir=EMBEDDINGS_8B_DIR,
        pure_texts=pure_texts,
        caption_items=caption_items,
        batch_text=args.batch_text,
        gpu_mem=0.90,
    )

    # 5. Evaluate
    evaluate_all()


if __name__ == "__main__":
    main()
