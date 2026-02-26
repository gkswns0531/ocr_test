#!/usr/bin/env python3
"""Prepare datasets one-by-one with memory guard (abort at 90%)."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time

import psutil

from config import BENCHMARKS, PREPARED_DIR
from datasets_loader import LOADERS

MEM_LIMIT_PCT = 90.0
SEED = 42

# Skip the two large datasets
SKIP = {"omnidocbench", "unimernet"}

TARGETS = [
    "upstage_dp_bench",
    "ocrbench",
    "pubtabnet",
    "teds_test",
    "nanonets_kie",
    "handwritten_forms",
]


def mem_pct() -> float:
    return psutil.virtual_memory().percent


def mem_info() -> str:
    m = psutil.virtual_memory()
    return f"RAM {m.used / 1e9:.1f}/{m.total / 1e9:.1f} GB ({m.percent:.0f}%)"


def check_mem(label: str = "") -> None:
    pct = mem_pct()
    print(f"  [{label}] {mem_info()}")
    if pct >= MEM_LIMIT_PCT:
        print(f"\n  *** ABORT: Memory {pct:.0f}% >= {MEM_LIMIT_PCT}% ***")
        sys.exit(1)


def prepare_one(key: str) -> None:
    import datasets_loader
    datasets_loader.DEFAULT_SEED = SEED

    bench_cfg = BENCHMARKS[key]
    out_dir = PREPARED_DIR / key
    meta_path = out_dir / "metadata.jsonl"

    # Remove stale data if exists (e.g. nanonets_kie dataset changed)
    if meta_path.exists():
        shutil.rmtree(out_dir)
        print(f"  [Removed stale] {key}/")

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    check_mem("Before load")

    print(f"  Loading {bench_cfg.name} (max_samples={bench_cfg.max_samples}) ...")
    loader = LOADERS[key]
    samples = loader(bench_cfg.max_samples)

    check_mem("After load")
    print(f"  Loaded {len(samples)} samples")

    with open(meta_path, "w") as mf:
        for i, sample in enumerate(samples):
            # Memory check every 50 samples
            if i > 0 and i % 50 == 0:
                pct = mem_pct()
                if pct >= MEM_LIMIT_PCT:
                    print(f"\n  *** ABORT mid-save: Memory {pct:.0f}% at sample {i} ***")
                    # Clean up partial output
                    shutil.rmtree(out_dir)
                    sys.exit(1)

            img_path = img_dir / f"{i:05d}.jpg"
            sample.image.save(str(img_path), format="JPEG", quality=95)

            meta = {
                "idx": i,
                "sample_id": sample.sample_id,
                "ground_truth": sample.ground_truth,
                "metadata": sample.metadata,
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  [Done] {len(samples)} samples, {total_bytes / 1e6:.1f} MB")

    del samples
    gc.collect()
    check_mem("After GC")


def main() -> None:
    print(f"Targets: {TARGETS}")
    print(f"Skip: {SKIP}")
    print(f"Memory limit: {MEM_LIMIT_PCT}%")
    print(f"Seed: {SEED}")
    print(f"Output: {PREPARED_DIR}\n")
    check_mem("Start")

    for i, key in enumerate(TARGETS):
        print(f"\n[{i+1}/{len(TARGETS)}] {key}")
        t0 = time.time()
        try:
            prepare_one(key)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [ERROR] {key}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    for key in TARGETS:
        d = PREPARED_DIR / key
        meta = d / "metadata.jsonl"
        if meta.exists():
            n = sum(1 for _ in open(meta))
            total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {key:25s} {n:>5} samples  {total_bytes/1e6:>8.1f} MB")
        else:
            print(f"  {key:25s} MISSING")
    check_mem("Final")


if __name__ == "__main__":
    main()
