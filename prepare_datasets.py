#!/usr/bin/env python3
"""Prepare and save benchmark datasets to disk.

Saves images as JPEG (compressed) + metadata as JSONL.
Each benchmark → prepared_datasets/{key}/images/*.jpg + metadata.jsonl

Usage:
    python prepare_datasets.py [--seed 42] [--benchmarks all]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

from config import BENCHMARKS, PREPARED_DIR
from datasets_loader import LOADERS


def prepare_one(benchmark_key: str, seed: int) -> None:
    """Download, sample, and save one benchmark dataset."""
    import datasets_loader
    datasets_loader.DEFAULT_SEED = seed

    bench_cfg = BENCHMARKS[benchmark_key]
    out_dir = PREPARED_DIR / benchmark_key
    meta_path = out_dir / "metadata.jsonl"

    if meta_path.exists():
        print(f"  [Skip] {benchmark_key} already exists")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"  [Load] {bench_cfg.name} (max_samples={bench_cfg.max_samples}, seed={seed})")
    _show_mem("  Before")

    loader = LOADERS[benchmark_key]
    samples = loader(bench_cfg.max_samples)
    _show_mem("  Loaded")

    # Save images as JPEG + metadata as JSONL
    with open(meta_path, "w") as mf:
        for i, sample in enumerate(samples):
            # Save image
            img_path = img_dir / f"{i:05d}.jpg"
            sample.image.save(str(img_path), format="JPEG", quality=95)

            # Save metadata (everything except image)
            meta = {
                "idx": i,
                "sample_id": sample.sample_id,
                "ground_truth": sample.ground_truth,
                "metadata": sample.metadata,
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Calculate total size
    total_bytes = sum(
        f.stat().st_size for f in out_dir.rglob("*") if f.is_file()
    )
    print(f"  [Done] {benchmark_key}: {len(samples)} samples, {total_bytes / 1e6:.1f} MB")

    # Free memory
    del samples
    gc.collect()
    _show_mem("  After GC")


def _show_mem(prefix: str = "") -> None:
    try:
        import psutil
        m = psutil.virtual_memory()
        print(f"{prefix} RAM: {m.used / 1e9:.1f}/{m.total / 1e9:.1f} GB ({m.percent}%)")
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", nargs="+", default=["all"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    args = parser.parse_args()

    keys = list(BENCHMARKS.keys()) if "all" in args.benchmarks else args.benchmarks

    if args.force:
        import shutil
        for k in keys:
            d = PREPARED_DIR / k
            if d.exists():
                shutil.rmtree(d)
                print(f"  [Del] Removed {k}/")

    print(f"Preparing {len(keys)} datasets (seed={args.seed})...")
    print(f"Output dir: {PREPARED_DIR}\n")

    for i, key in enumerate(keys):
        print(f"[{i + 1}/{len(keys)}] {key}")
        try:
            prepare_one(key, args.seed)
        except Exception as e:
            print(f"  [ERROR] {key}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        print()

    # Summary
    print("=" * 50)
    print("PREPARED DATASETS:")
    for key in keys:
        d = PREPARED_DIR / key
        meta = d / "metadata.jsonl"
        if meta.exists():
            n = sum(1 for _ in open(meta))
            total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {key}: {n} samples, {total_bytes / 1e6:.1f} MB")
        else:
            print(f"  {key}: MISSING")


if __name__ == "__main__":
    main()
