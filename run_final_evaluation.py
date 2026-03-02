#!/usr/bin/env python3
"""Run full R@K evaluation with corrected region embeddings.

Computes:
1. Full 600-query R@K for all configurations (text-only, text+region, pure+region)
2. Per-type R@K breakdown
3. True visual 40-case analysis
4. Caption experiments EXCLUDED (captions generated from wrong crops)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATA_DIR = Path("output_dl")
EMBEDDINGS_2B_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_8B_DIR = DATA_DIR / "embeddings_8b"
OCR_RESULTS_FILE = DATA_DIR / "ocr_results.jsonl"


def load_qa_and_annotations():
    """Load QA and annotation datasets."""
    qa = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-QA",
        split="test", cache_dir="data",
    )
    ann = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-annotations",
        split="test", cache_dir="data",
    )

    idx_to_info: dict[int, tuple[str, int]] = {}
    for row in ann:
        for offset, ci in enumerate(row["page_indices"]):
            idx_to_info[ci] = (row["file_id"], offset)

    jsonl_pids: list[str] = []
    with open(OCR_RESULTS_FILE) as f:
        for line in f:
            jsonl_pids.append(json.loads(line)["page_id"])

    return qa, idx_to_info, jsonl_pids


def compute_recall_at_k(
    queries: np.ndarray,
    combined_emb: np.ndarray,
    combined_pids: list[str],
    qa,
    idx_to_info: dict,
    K: int,
    query_indices: list[int] | None = None,
) -> float:
    """Compute R@K for given queries vs combined corpus."""
    sim = queries @ combined_emb.T
    correct = 0
    total = 0
    indices = query_indices or list(range(len(qa)))

    for qi in indices:
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
        total += 1

    return correct / total if total > 0 else 0.0


def get_true_visual_40_indices(qa) -> list[int]:
    """Get indices of 40 true_visual queries (58 visual minus 18 without images on GT pages)."""
    # Load from saved file if available
    detail_file = Path("/tmp/true_visual_40_detail.json")
    if detail_file.exists():
        with open(detail_file) as f:
            data = json.load(f)
        return [case["qi"] for case in data]

    # Otherwise, recompute
    # True visual = type "visual" AND has image/chart regions on GT pages
    region_pages = set()
    with open(OCR_RESULTS_FILE) as f:
        for line in f:
            rec = json.loads(line)
            for crop in rec.get("image_crops", []):
                if crop.get("saved"):
                    region_pages.add(rec["page_id"])

    ann = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-annotations",
        split="test", cache_dir="data",
    )
    idx_to_info: dict[int, tuple[str, int]] = {}
    for row in ann:
        for offset, ci in enumerate(row["page_indices"]):
            idx_to_info[ci] = (row["file_id"], offset)

    indices = []
    for qi in range(len(qa)):
        if qa[qi].get("type", "") != "visual":
            continue
        gt = qa[qi]["ground_truth"]
        has_image_on_gt = False
        for gi in gt:
            if gi in idx_to_info:
                fid, off = idx_to_info[gi]
                pid = f"{fid}_{off+1}"
                if pid in region_pages:
                    has_image_on_gt = True
                    break
        if has_image_on_gt:
            indices.append(qi)

    return indices


def main() -> None:
    print("=" * 70)
    print("Final R@K Evaluation (Corrected Region Embeddings)")
    print("=" * 70)

    qa, idx_to_info, jsonl_pids = load_qa_and_annotations()
    print(f"QA: {len(qa)} queries")

    true_visual_indices = get_true_visual_40_indices(qa)
    print(f"True visual queries: {len(true_visual_indices)}")

    configs = [
        ("2B", EMBEDDINGS_2B_DIR),
        ("8B", EMBEDDINGS_8B_DIR),
    ]

    results_all = {}

    for model_label, emb_dir in configs:
        queries = np.load(str(emb_dir / "queries.npy"))
        print(f"\n{'='*70}")
        print(f"  {model_label} Model (dim={queries.shape[1]})")
        print(f"{'='*70}")

        # Load corpus embeddings
        corpus_types: dict[str, tuple[np.ndarray, list[str]]] = {}

        # OCR text
        ct_path = emb_dir / "corpus_ocr_text.npy"
        if ct_path.exists():
            corpus_types["text"] = (np.load(str(ct_path)), jsonl_pids)

        # Pure text
        pt_path = emb_dir / "corpus_pure_text.npy"
        if pt_path.exists():
            with open(emb_dir / "pure_text_page_ids.json") as f:
                pt_pids = json.load(f)
            corpus_types["pure_text"] = (np.load(str(pt_path)), pt_pids)

        # Region multimodal (corrected)
        rg_path = emb_dir / "corpus_regions.npy"
        rg_meta_path = emb_dir / "region_metadata.jsonl"
        if rg_path.exists() and rg_meta_path.exists():
            rg_meta = []
            with open(rg_meta_path) as f:
                for line in f:
                    rg_meta.append(json.loads(line))
            rg_pids = [r["page_id"] for r in rg_meta]
            corpus_types["region_mm"] = (np.load(str(rg_path)), rg_pids)
            print(f"  Region embeddings: {len(rg_pids)} regions")
        else:
            print(f"  WARNING: Region embeddings not found at {rg_path}")

        # Caption text embeddings (v2 corrected)
        cap_path = emb_dir / "corpus_caption_text.npy"
        if cap_path.exists():
            with open(emb_dir / "caption_page_ids.json") as f:
                cap_pids = json.load(f)
            corpus_types["caption_text"] = (np.load(str(cap_path)), cap_pids)
            print(f"  Caption embeddings: {len(cap_pids)} pages")

        # Retrieval configs
        retrieval_configs: list[tuple[str, list[str]]] = [
            ("text-only", ["text"]),
            ("pure-text-only", ["pure_text"]),
            ("text+region", ["text", "region_mm"]),
            ("pure+region", ["pure_text", "region_mm"]),
            ("text+caption", ["text", "caption_text"]),
            ("pure+caption", ["pure_text", "caption_text"]),
            ("text+rgn+cap", ["text", "region_mm", "caption_text"]),
            ("pure+rgn+cap", ["pure_text", "region_mm", "caption_text"]),
        ]
        retrieval_configs = [
            (name, keys) for name, keys in retrieval_configs
            if all(k in corpus_types for k in keys)
        ]

        # === All 600 queries ===
        print(f"\n  All {len(qa)} queries:")
        header = f"{'K':>4}"
        for name, _ in retrieval_configs:
            header += f" | {name:>15}"
        print(f"  {header}")
        print(f"  {'-' * len(header)}")

        model_results = {}
        for K in [1, 5, 10, 20]:
            row = f"  R@{K:>2}"
            for name, keys in retrieval_configs:
                combined_emb = np.vstack([corpus_types[k][0] for k in keys])
                combined_pids = []
                for k in keys:
                    combined_pids.extend(corpus_types[k][1])

                r = compute_recall_at_k(queries, combined_emb, combined_pids, qa, idx_to_info, K)
                row += f" | {100*r:>13.1f}%"
                model_results[f"{name}_R@{K}"] = r
            print(row)

        # === Per-type R@10 ===
        print(f"\n  Per-type R@10:")
        types = sorted(set(qa[qi].get("type", "unknown") for qi in range(len(qa))))
        type_header = f"  {'type':>10}"
        for name, _ in retrieval_configs:
            type_header += f" | {name:>15}"
        print(type_header)
        print(f"  {'-' * (len(type_header) - 2)}")

        for qtype in types:
            type_indices = [qi for qi in range(len(qa)) if qa[qi].get("type", "unknown") == qtype]
            type_row = f"  {qtype:>10}"
            for name, keys in retrieval_configs:
                combined_emb = np.vstack([corpus_types[k][0] for k in keys])
                combined_pids = []
                for k in keys:
                    combined_pids.extend(corpus_types[k][1])
                r = compute_recall_at_k(queries, combined_emb, combined_pids, qa, idx_to_info, 10, type_indices)
                type_row += f" | {100*r:>13.1f}%"
                model_results[f"{name}_{qtype}_R@10"] = r
            print(f"{type_row} (n={len(type_indices)})")

        # === True visual 40 queries ===
        if true_visual_indices:
            print(f"\n  True Visual ({len(true_visual_indices)} queries):")
            tv_header = f"  {'K':>4}"
            for name, _ in retrieval_configs:
                tv_header += f" | {name:>15}"
            print(tv_header)
            print(f"  {'-' * len(tv_header)}")

            for K in [1, 5, 10, 20]:
                tv_row = f"  R@{K:>2}"
                for name, keys in retrieval_configs:
                    combined_emb = np.vstack([corpus_types[k][0] for k in keys])
                    combined_pids = []
                    for k in keys:
                        combined_pids.extend(corpus_types[k][1])
                    r = compute_recall_at_k(queries, combined_emb, combined_pids, qa, idx_to_info, K, true_visual_indices)
                    tv_row += f" | {100*r:>13.1f}%"
                    model_results[f"tv_{name}_R@{K}"] = r
                print(tv_row)

        results_all[model_label] = model_results

    # Save results
    with open(DATA_DIR / "evaluation_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
