#!/usr/bin/env python3
"""Compute embeddings using Qwen3-VL-Embedding-8B-FP8 and compare with 2B.

Reuses the pipeline's VLEmbeddingModel but with the 8B model.
Saves to output/embeddings_8b/ directory.

Usage:
    python3 run_8b_embeddings.py [--batch-text 256] [--batch-image 128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Import pipeline components
sys.path.insert(0, str(Path(__file__).parent))
from run_b200_pipeline import (
    VLEmbeddingModel,
    EmbeddingModelConfig,
    extract_region_items,
    resize_to_max_pixels,
    _release_memory,
    QUERY_INSTRUCTION,
    IMAGE_INSTRUCTION,
    OCR_RESULTS_FILE,
    PARSED_TEXTS_FILE,
    EMBEDDINGS_DIR,
    OUTPUT_DIR,
)
from PIL import Image


EMBEDDINGS_8B_DIR = OUTPUT_DIR / "embeddings_8b"
MODEL_8B_ID = "Forturne/Qwen3-VL-Embedding-8B-FP8"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-text", type=int, default=256)
    parser.add_argument("--batch-image", type=int, default=128)
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-VL-Embedding-8B Embeddings")
    print("=" * 60)

    EMBEDDINGS_8B_DIR.mkdir(parents=True, exist_ok=True)

    # Load parsed texts
    page_ids: list[str] = []
    page_texts: list[str] = []
    with open(PARSED_TEXTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                page_ids.append(obj["page_id"])
                page_texts.append(obj.get("parsed_text", ""))
    print(f"  Loaded {len(page_ids)} parsed texts")

    # Load model
    model_cfg = EmbeddingModelConfig(
        name="qwen3-vl-embedding-8b",
        model_id=MODEL_8B_ID,
        batch_size_image=args.batch_image,
        batch_size_text=args.batch_text,
        quantization=None,  # compressed-tensors FP8
        gpu_memory_utilization=0.90,
    )
    model = VLEmbeddingModel(model_cfg)

    # 1. Region multimodal embeddings
    corpus_regions_path = EMBEDDINGS_8B_DIR / "corpus_regions.npy"
    region_metadata_path = EMBEDDINGS_8B_DIR / "region_metadata.jsonl"

    if corpus_regions_path.exists():
        print(f"  Skipping region embeddings (exists: {corpus_regions_path})")
    else:
        print("  Extracting image/chart regions...")
        region_items = extract_region_items(OCR_RESULTS_FILE)
        print(f"  Found {len(region_items)} regions")

        multimodal_pairs: list[tuple[Image.Image, str]] = []
        valid_items: list[dict] = []
        for item in region_items:
            crop_path = item["crop_path"]
            if not crop_path or not Path(crop_path).exists():
                continue
            img = resize_to_max_pixels(Image.open(crop_path).convert("RGB"))
            multimodal_pairs.append((img, item["caption_text"]))
            valid_items.append(item)

        if multimodal_pairs:
            print(f"  Computing 8B multimodal embeddings for {len(multimodal_pairs)} regions...")
            region_embs = model.encode_multimodal(multimodal_pairs, batch_size=args.batch_image)
            np.save(str(corpus_regions_path), region_embs)
            print(f"  Saved: {corpus_regions_path} (shape={region_embs.shape})")

            with open(region_metadata_path, "w", encoding="utf-8") as f:
                for item in valid_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  Saved: {region_metadata_path}")

            del region_embs, multimodal_pairs
            _release_memory()

    # 2. Text embeddings
    corpus_text_path = EMBEDDINGS_8B_DIR / "corpus_ocr_text.npy"
    if corpus_text_path.exists():
        print(f"  Skipping text embeddings (exists: {corpus_text_path})")
    else:
        print("  Computing 8B text embeddings...")
        text_embs = model.encode_texts(page_texts, batch_size=args.batch_text, instruction="")
        np.save(str(corpus_text_path), text_embs)
        print(f"  Saved: {corpus_text_path} (shape={text_embs.shape})")
        del text_embs
        _release_memory()

    # 3. Query embeddings
    queries_path = EMBEDDINGS_8B_DIR / "queries.npy"
    if queries_path.exists():
        print(f"  Skipping query embeddings (exists: {queries_path})")
    else:
        print("  Computing 8B query embeddings...")
        from datasets import load_dataset
        qa_ds = load_dataset(
            "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark",
            name="SDS-KoPub-QA",
            cache_dir="data",
            split="test",
        )
        query_texts = [item["query"] for item in qa_ds]
        query_embs = model.encode_texts(query_texts, batch_size=args.batch_text, instruction=QUERY_INSTRUCTION)
        np.save(str(queries_path), query_embs)
        print(f"  Saved: {queries_path} (shape={query_embs.shape})")
        del query_embs, qa_ds
        _release_memory()

    del model
    _release_memory()

    print("\n  8B embedding computation complete!")

    # 4. Evaluate
    print("\n" + "=" * 60)
    print("R@K Evaluation: 2B vs 8B")
    print("=" * 60)
    evaluate_comparison()


def evaluate_comparison() -> None:
    """Compare 2B vs 8B embeddings."""
    from datasets import load_dataset

    qa = load_dataset("SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-QA", split="test", cache_dir="data")
    ann = load_dataset("SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-annotations", split="test", cache_dir="data")

    idx_to_info = {}
    for row in ann:
        for offset, ci in enumerate(row['page_indices']):
            idx_to_info[ci] = (row['file_id'], offset)

    jsonl_pids = []
    with open(OCR_RESULTS_FILE) as f:
        for line in f:
            jsonl_pids.append(json.loads(line)["page_id"])

    configs = {
        "2B": EMBEDDINGS_DIR,
        "8B": EMBEDDINGS_8B_DIR,
    }

    for label, emb_dir in configs.items():
        corpus_text = np.load(str(emb_dir / "corpus_ocr_text.npy"))
        queries = np.load(str(emb_dir / "queries.npy"))

        region_meta = []
        region_meta_path = emb_dir / "region_metadata.jsonl"
        if region_meta_path.exists():
            with open(region_meta_path) as f:
                for line in f:
                    region_meta.append(json.loads(line))

        region_pids = [r["page_id"] for r in region_meta]
        corpus_regions_path = emb_dir / "corpus_regions.npy"
        has_regions = corpus_regions_path.exists()

        sim_text = queries @ corpus_text.T

        if has_regions:
            corpus_regions = np.load(str(corpus_regions_path))
            combined_emb = np.vstack([corpus_text, corpus_regions])
            combined_pids = jsonl_pids + region_pids
            sim_combined = queries @ combined_emb.T

        print(f"\n--- {label} ---")
        print(f"{'K':>4} | {'Text-only':>12} | {'Combined':>12}")
        print("-" * 40)

        for K in [1, 5, 10, 20]:
            ct = cc = 0
            for qi in range(len(qa)):
                gt = qa[qi]['ground_truth']
                expected = set()
                for gi in gt:
                    if gi in idx_to_info:
                        fid, off = idx_to_info[gi]
                        expected.add(f"{fid}_{off+1}")

                top_t = np.argsort(sim_text[qi])[::-1][:K]
                if expected & set(jsonl_pids[i] for i in top_t):
                    ct += 1

                if has_regions:
                    sorted_idx = np.argsort(sim_combined[qi])[::-1]
                    seen = set()
                    top_c = set()
                    for idx in sorted_idx:
                        pid = combined_pids[idx]
                        if pid not in seen:
                            seen.add(pid)
                            top_c.add(pid)
                            if len(top_c) >= K:
                                break
                    if expected & top_c:
                        cc += 1

                n = len(qa)
            comb_str = f"{100*cc/n:.1f}%" if has_regions else "N/A"
            print(f"R@{K:>2} | {100*ct/n:>10.1f}% | {comb_str:>10s}")

        # Per-type R@10
        print(f"\n  Per-type R@10 ({label}):")
        type_counts: dict[str, int] = {}
        type_correct: dict[str, int] = {}
        for qi in range(len(qa)):
            qtype = qa[qi].get("type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            gt = qa[qi]['ground_truth']
            expected = set()
            for gi in gt:
                if gi in idx_to_info:
                    fid, off = idx_to_info[gi]
                    expected.add(f"{fid}_{off+1}")

            if has_regions:
                sorted_idx = np.argsort(sim_combined[qi])[::-1]
                seen = set()
                top_c = set()
                for idx in sorted_idx:
                    pid = combined_pids[idx]
                    if pid not in seen:
                        seen.add(pid)
                        top_c.add(pid)
                        if len(top_c) >= 10:
                            break
                if expected & top_c:
                    type_correct[qtype] = type_correct.get(qtype, 0) + 1
            else:
                top_t = np.argsort(sim_text[qi])[::-1][:10]
                if expected & set(jsonl_pids[i] for i in top_t):
                    type_correct[qtype] = type_correct.get(qtype, 0) + 1

        for qtype in sorted(type_counts.keys()):
            n = type_counts[qtype]
            c = type_correct.get(qtype, 0)
            print(f"    {qtype:10s}: {100*c/n:.1f}% ({c}/{n})")


if __name__ == "__main__":
    main()
