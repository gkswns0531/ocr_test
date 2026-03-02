#!/usr/bin/env python3
"""Final comparison: pure_text, region, and v11 caption (4o-mini production) on true_visual 40.

Tests across 3 embedding models: Qwen3-VL 2B, 8B, and BGE-M3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATA_DIR = Path("output_dl")


def load_qa_and_annotations():
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
    return qa, idx_to_info


def compute_recall_at_k(queries, emb, pids, qa, idx_to_info, K, query_indices):
    sim = queries @ emb.T
    correct = 0
    for qi in query_indices:
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
            pid = pids[idx]
            if pid not in seen:
                seen.add(pid)
                top_pids.add(pid)
                if len(top_pids) >= K:
                    break
        if expected & top_pids:
            correct += 1
    return correct / len(query_indices)


def get_true_visual_40() -> list[int]:
    f = Path("/tmp/true_visual_40_detail.json")
    if f.exists():
        with open(f) as fh:
            return [c["qi"] for c in json.load(fh)]
    return []


def load_configs_for_model(emb_dir: Path) -> list[tuple[str, np.ndarray, list[str]]]:
    """Load pure_text, region, and v11 caption embeddings from a given directory."""
    configs: list[tuple[str, np.ndarray, list[str]]] = []

    # pure_text (two naming conventions)
    pt_path = emb_dir / "corpus_pure_text.npy"
    pt_pids_path = emb_dir / "pure_text_page_ids.json"
    pt_meta_path = emb_dir / "text_metadata.jsonl"
    alt_pt_path = emb_dir / "corpus_text.npy"

    if pt_path.exists() and pt_pids_path.exists():
        pt = np.load(str(pt_path))
        with open(pt_pids_path) as f:
            pt_pids = json.load(f)
        configs.append(("pure_text", pt, pt_pids))
    elif alt_pt_path.exists() and pt_meta_path.exists():
        pt = np.load(str(alt_pt_path))
        pt_meta = []
        with open(pt_meta_path) as f:
            for line in f:
                pt_meta.append(json.loads(line))
        configs.append(("pure_text", pt, [r["page_id"] for r in pt_meta]))

    # region
    rg_path = emb_dir / "corpus_regions.npy"
    rg_meta_path = emb_dir / "region_metadata.jsonl"
    if rg_path.exists() and rg_meta_path.exists():
        rg = np.load(str(rg_path))
        rg_meta = []
        with open(rg_meta_path) as f:
            for line in f:
                rg_meta.append(json.loads(line))
        configs.append(("region", rg, [r["page_id"] for r in rg_meta]))

    # v11 caption (4o-mini production)
    cap_path = emb_dir / "corpus_caption.npy"
    pids_path = emb_dir / "caption_page_ids.json"
    if cap_path.exists() and pids_path.exists():
        cap = np.load(str(cap_path))
        with open(pids_path) as f:
            cap_pids = json.load(f)
        configs.append(("caption", cap, cap_pids))

    return configs


def main() -> None:
    qa, idx_to_info = load_qa_and_annotations()
    tv = get_true_visual_40()
    print(f"True visual queries: {len(tv)}\n")

    models = [
        ("Qwen3-VL-2B", DATA_DIR / "embeddings"),
        ("Qwen3-VL-8B", DATA_DIR / "embeddings_8b"),
        ("BGE-M3", DATA_DIR / "embeddings_bge_m3"),
    ]

    for label, emb_dir in models:
        queries_path = emb_dir / "queries.npy"
        if not queries_path.exists():
            print(f"[{label}] queries.npy not found, skipping")
            continue

        queries = np.load(str(queries_path))
        configs = load_configs_for_model(emb_dir)

        if not configs:
            print(f"[{label}] No corpus embeddings found, skipping")
            continue

        print(f"{'='*80}")
        print(f"  {label} — True Visual {len(tv)} queries")
        print(f"{'='*80}")

        header = f"  {'K':>4}"
        for name, _, _ in configs:
            header += f" | {name:>16}"
        print(header)
        print(f"  {'-' * len(header)}")

        for K in [1, 5, 10, 20]:
            row = f"  R@{K:>2}"
            for name, emb, pids in configs:
                r = compute_recall_at_k(queries, emb, pids, qa, idx_to_info, K, tv)
                row += f" | {100*r:>14.1f}%"
            print(row)
        print()


if __name__ == "__main__":
    main()
