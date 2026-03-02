#!/usr/bin/env python3
"""Compute BGE-M3 embeddings for queries, pure_text, regions (text-only), and captions.

BGE-M3 is text-only (no multimodal), so region embeddings use OCR caption text only.
Uses vLLM pooling runner for efficient batched inference.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from run_b200_pipeline import extract_page_text, _release_memory

DATA_DIR = Path("output_dl")
EMB_DIR = DATA_DIR / "embeddings_bge_m3"

MODEL_ID = "BAAI/bge-m3"
BATCH_SIZE = 256  # BGE-M3 is smaller, can batch more


def load_bge_m3():
    from vllm import LLM

    print(f"Loading BGE-M3 via vLLM...")
    model = LLM(
        model=MODEL_ID,
        runner="pooling",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
    )
    print("BGE-M3 loaded.")
    return model


def encode_texts(model, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode texts using vLLM embed() and L2-normalize."""
    all_embs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = model.embed(batch)
        embs = np.array([out.outputs.embedding for out in outputs])
        all_embs.append(embs)
    embs = np.concatenate(all_embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return embs / norms


def compute_query_embeddings(model) -> None:
    out_path = EMB_DIR / "queries.npy"
    if out_path.exists():
        print("[queries] Skipping (exists)")
        return

    qa = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-QA",
        split="test", cache_dir="data",
    )
    query_texts = [item["query"] for item in qa]
    print(f"[queries] Encoding {len(query_texts)} queries...")
    t0 = time.time()
    embs = encode_texts(model, query_texts)
    print(f"[queries] Done in {time.time()-t0:.1f}s, shape={embs.shape}")
    np.save(str(out_path), embs)


def compute_pure_text_embeddings(model) -> None:
    out_path = EMB_DIR / "corpus_text.npy"
    meta_path = EMB_DIR / "text_metadata.jsonl"
    if out_path.exists():
        print("[pure_text] Skipping (exists)")
        return

    # Load OCR results and extract page texts (same logic as 2B/8B)
    ocr_file = DATA_DIR / "ocr_results.jsonl"
    page_texts: list[tuple[str, str]] = []
    max_chars = int(3800 * 1.2)

    with open(ocr_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_id = rec["page_id"]
            text = extract_page_text(rec.get("regions", []))
            if text.strip():
                if len(text) > max_chars:
                    text = text[:max_chars]
                page_texts.append((page_id, text))

    texts = [t for _, t in page_texts]
    pids = [pid for pid, _ in page_texts]

    print(f"[pure_text] Encoding {len(texts)} pages...")
    t0 = time.time()
    embs = encode_texts(model, texts)
    print(f"[pure_text] Done in {time.time()-t0:.1f}s, shape={embs.shape}")

    np.save(str(out_path), embs)
    with open(meta_path, "w", encoding="utf-8") as f:
        for pid, text in page_texts:
            f.write(json.dumps({"page_id": pid, "text_preview": text[:100]}, ensure_ascii=False) + "\n")


def compute_region_text_embeddings(model) -> None:
    """Compute region embeddings using OCR caption text only (no images for BGE-M3)."""
    out_path = EMB_DIR / "corpus_regions.npy"
    meta_path = EMB_DIR / "region_metadata.jsonl"
    if out_path.exists():
        print("[region_text] Skipping (exists)")
        return

    # Load same region metadata as 2B/8B but use text-only embedding
    ref_meta_path = DATA_DIR / "embeddings" / "region_metadata.jsonl"
    if not ref_meta_path.exists():
        print("[region_text] No region_metadata.jsonl found, skipping")
        return

    records: list[dict] = []
    with open(ref_meta_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # BGE-M3 can only use the caption text portion
    texts = [r.get("caption_text", "") or "" for r in records]
    pids = [r["page_id"] for r in records]

    # Filter out empty captions
    valid = [(t, pid, r) for t, pid, r in zip(texts, pids, records) if t.strip()]
    if not valid:
        print("[region_text] No captions found, skipping")
        return

    texts_valid = [v[0] for v in valid]
    records_valid = [v[2] for v in valid]

    print(f"[region_text] Encoding {len(texts_valid)} regions...")
    t0 = time.time()
    embs = encode_texts(model, texts_valid)
    print(f"[region_text] Done in {time.time()-t0:.1f}s, shape={embs.shape}")

    np.save(str(out_path), embs)
    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records_valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_caption_embeddings(model) -> None:
    out_path = EMB_DIR / "corpus_caption.npy"
    pids_path = EMB_DIR / "caption_page_ids.json"
    if out_path.exists():
        print("[caption] Skipping (exists)")
        return

    captions_file = DATA_DIR / "region_captions.jsonl"
    if not captions_file.exists():
        print("[caption] No captions file, skipping")
        return

    # Aggregate captions per page (same logic as compute_caption_embeddings_v11.py)
    page_captions: dict[str, list[str]] = {}
    with open(captions_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            caption = rec.get("caption", "").strip()
            if not caption:
                continue
            page_id = rec["page_id"]
            if page_id not in page_captions:
                page_captions[page_id] = []
            page_captions[page_id].append(caption)

    max_chars = int(3800 * 1.2)
    items: list[tuple[str, str]] = []
    for pid, caps in page_captions.items():
        text = "\n\n".join(caps)
        if len(text) > max_chars:
            text = text[:max_chars]
        items.append((pid, text))

    texts = [t for _, t in items]
    pids = [pid for pid, _ in items]

    print(f"[caption] Encoding {len(texts)} pages...")
    t0 = time.time()
    embs = encode_texts(model, texts)
    print(f"[caption] Done in {time.time()-t0:.1f}s, shape={embs.shape}")

    np.save(str(out_path), embs)
    with open(pids_path, "w") as f:
        json.dump(pids, f)


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    model = load_bge_m3()

    compute_query_embeddings(model)
    compute_pure_text_embeddings(model)
    compute_region_text_embeddings(model)
    compute_caption_embeddings(model)

    del model
    _release_memory()
    print("\nAll BGE-M3 embeddings done!")


if __name__ == "__main__":
    main()
