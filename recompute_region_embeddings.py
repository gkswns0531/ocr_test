#!/usr/bin/env python3
"""Recompute region multimodal embeddings with corrected crop images.

Processes both 2B and 8B models sequentially.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from run_b200_pipeline import (
    VLEmbeddingModel,
    EmbeddingModelConfig,
    extract_region_items,
    resize_to_max_pixels,
    _release_memory,
)

# Use output_dl/ (downloaded data), not output/ (pipeline default)
DATA_DIR = Path("output_dl")
OCR_RESULTS_FILE = DATA_DIR / "ocr_results.jsonl"
EMBEDDINGS_2B_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_8B_DIR = DATA_DIR / "embeddings_8b"

MODEL_2B_ID = "Forturne/Qwen3-VL-Embedding-2B-FP8"
MODEL_8B_ID = "Forturne/Qwen3-VL-Embedding-8B-FP8"


def compute_region_embeddings(
    model_name: str,
    model_id: str,
    emb_dir: Path,
    region_items: list[dict],
    batch_size_image: int,
    gpu_mem: float,
) -> None:
    """Compute region multimodal embeddings for a given model."""
    corpus_regions_path = emb_dir / "corpus_regions.npy"
    region_metadata_path = emb_dir / "region_metadata.jsonl"

    if corpus_regions_path.exists():
        print(f"  [{model_name}] Skipping (already exists)")
        return

    model_cfg = EmbeddingModelConfig(
        name=model_name,
        model_id=model_id,
        batch_size_image=batch_size_image,
        batch_size_text=256,
        quantization=None,
        gpu_memory_utilization=gpu_mem,
    )
    model = VLEmbeddingModel(model_cfg)

    # Load crop images and pair with caption text
    print(f"  [{model_name}] Loading {len(region_items)} crop images...")
    multimodal_pairs: list[tuple[Image.Image, str]] = []
    valid_items: list[dict] = []
    skipped = 0

    for item in tqdm(region_items, desc="Loading crops"):
        crop_path = item["crop_path"]
        if not crop_path or not Path(crop_path).exists():
            skipped += 1
            continue
        img = resize_to_max_pixels(Image.open(crop_path).convert("RGB"))
        multimodal_pairs.append((img, item["caption_text"]))
        valid_items.append(item)

    print(f"  [{model_name}] Valid: {len(valid_items)}, Skipped: {skipped}")

    if not multimodal_pairs:
        print(f"  [{model_name}] No valid crops. Aborting.")
        del model
        _release_memory()
        return

    print(f"  [{model_name}] Computing multimodal embeddings...")
    t0 = time.time()
    region_embs = model.encode_multimodal(multimodal_pairs, batch_size=batch_size_image)
    elapsed = time.time() - t0
    print(f"  [{model_name}] Done in {elapsed:.1f}s, shape={region_embs.shape}")

    np.save(str(corpus_regions_path), region_embs)
    print(f"  [{model_name}] Saved: {corpus_regions_path}")

    with open(region_metadata_path, "w", encoding="utf-8") as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  [{model_name}] Saved: {region_metadata_path} ({len(valid_items)} entries)")

    del model, region_embs, multimodal_pairs
    _release_memory()


def main() -> None:
    print("=" * 70)
    print("Recomputing Region Embeddings with Corrected Crops")
    print("=" * 70)

    # Extract region items (uses updated ocr_results.jsonl with correct crop paths)
    print("\nExtracting region items from OCR results...")
    region_items = extract_region_items(OCR_RESULTS_FILE)
    print(f"Found {len(region_items)} image/chart regions")

    # Verify crop paths are correct
    n_ok = sum(1 for item in region_items if Path(item["crop_path"]).exists())
    print(f"Crops found: {n_ok}/{len(region_items)}")

    # 2B model
    print(f"\n{'='*70}")
    print("2B Model")
    print(f"{'='*70}")
    compute_region_embeddings(
        model_name="qwen3-vl-2b",
        model_id=MODEL_2B_ID,
        emb_dir=EMBEDDINGS_2B_DIR,
        region_items=region_items,
        batch_size_image=128,
        gpu_mem=0.85,
    )

    # 8B model
    print(f"\n{'='*70}")
    print("8B Model")
    print(f"{'='*70}")
    compute_region_embeddings(
        model_name="qwen3-vl-8b",
        model_id=MODEL_8B_ID,
        emb_dir=EMBEDDINGS_8B_DIR,
        region_items=region_items,
        batch_size_image=64,
        gpu_mem=0.90,
    )

    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
