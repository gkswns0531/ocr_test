#!/usr/bin/env python3
"""Compute caption text embeddings from production captions (gpt-4o-mini)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_b200_pipeline import VLEmbeddingModel, EmbeddingModelConfig, _release_memory

DATA_DIR = Path("output_dl")
CAPTIONS_FILE = DATA_DIR / "region_captions.jsonl"

MODEL_CONFIGS = [
    ("qwen3-vl-2b", "Forturne/Qwen3-VL-Embedding-2B-FP8", DATA_DIR / "embeddings", 256, 0.85),
    ("qwen3-vl-8b", "Forturne/Qwen3-VL-Embedding-8B-FP8", DATA_DIR / "embeddings_8b", 256, 0.90),
]


def load_caption_texts() -> list[tuple[str, str]]:
    page_captions: dict[str, list[str]] = {}
    with open(CAPTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            caption = rec.get("caption", "").strip()
            if not caption:
                continue
            page_id = rec["page_id"]
            if page_id not in page_captions:
                page_captions[page_id] = []
            page_captions[page_id].append(caption)

    max_chars = int(3800 * 1.2)  # Korean → ~1.2 chars/token
    results: list[tuple[str, str]] = []
    for pid, caps in page_captions.items():
        text = "\n\n".join(caps)
        if len(text) > max_chars:
            text = text[:max_chars]
        results.append((pid, text))
    return results


def main() -> None:
    caption_items = load_caption_texts()
    print(f"Caption texts: {len(caption_items)} pages")

    for model_name, model_id, emb_dir, batch_text, gpu_mem in MODEL_CONFIGS:
        emb_path = emb_dir / "corpus_caption.npy"
        pids_path = emb_dir / "caption_page_ids.json"

        if emb_path.exists():
            print(f"[{model_name}] Skipping (exists)")
            continue

        model = VLEmbeddingModel(EmbeddingModelConfig(
            name=model_name, model_id=model_id,
            batch_size_image=128, batch_size_text=batch_text,
            quantization=None, gpu_memory_utilization=gpu_mem,
        ))

        texts = [t for _, t in caption_items]
        pids = [pid for pid, _ in caption_items]

        print(f"[{model_name}] Computing ({len(texts)} pages)...")
        t0 = time.time()
        embs = model.encode_texts(texts, batch_size=batch_text, instruction="")
        print(f"[{model_name}] Done in {time.time()-t0:.1f}s, shape={embs.shape}")

        np.save(str(emb_path), embs)
        with open(pids_path, "w") as f:
            json.dump(pids, f)

        del model, embs
        _release_memory()

    print("All done!")


if __name__ == "__main__":
    main()
