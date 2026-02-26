#!/usr/bin/env python3
"""Prepare nanonets_kie and handwritten_forms with streaming (low memory)."""

from __future__ import annotations

import gc
import io
import json
import shutil
import sys
import time

import psutil
from datasets import load_dataset
from PIL import Image

from config import PREPARED_DIR, DATA_CACHE_DIR

MEM_LIMIT_PCT = 90.0


def mem_info() -> str:
    m = psutil.virtual_memory()
    return f"RAM {m.used / 1e9:.1f}/{m.total / 1e9:.1f} GB ({m.percent:.0f}%)"


def check_mem(label: str = "") -> None:
    pct = psutil.virtual_memory().percent
    print(f"  [{label}] {mem_info()}")
    if pct >= MEM_LIMIT_PCT:
        print(f"\n  *** ABORT: Memory {pct:.0f}% >= {MEM_LIMIT_PCT}% ***")
        sys.exit(1)


def ensure_rgb(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    return None


def prepare_nanonets_kie() -> None:
    """Stream nanonets/key_information_extraction one sample at a time."""
    key = "nanonets_kie"
    out_dir = PREPARED_DIR / key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"\n[nanonets_kie] Streaming from nanonets/key_information_extraction ...")
    check_mem("Before")

    ds = load_dataset("nanonets/key_information_extraction", split="test", streaming=True)
    count = 0

    with open(out_dir / "metadata.jsonl", "w") as mf:
        for i, row in enumerate(ds):
            if i % 100 == 0:
                pct = psutil.virtual_memory().percent
                if pct >= MEM_LIMIT_PCT:
                    print(f"\n  *** ABORT at sample {i}: Memory {pct:.0f}% ***")
                    sys.exit(1)
                if i > 0:
                    print(f"  ... {i} samples, {mem_info()}")

            img = ensure_rgb(row.get("image"))
            if img is None:
                continue

            img_path = img_dir / f"{i:05d}.jpg"
            img.save(str(img_path), format="JPEG", quality=95)

            gt_fields = row.get("annotations", {})
            if not isinstance(gt_fields, dict):
                gt_fields = {}

            meta = {
                "idx": count,
                "sample_id": f"nanonets_kie_{i}",
                "ground_truth": gt_fields,
                "metadata": {"index": i},
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1

            # Free image memory immediately
            del img
            if i % 50 == 0:
                gc.collect()

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  [Done] {count} samples, {total_bytes / 1e6:.1f} MB")
    gc.collect()
    check_mem("After GC")


def prepare_handwritten() -> None:
    """Stream Teklia/IAM-line one sample at a time."""
    key = "handwritten_forms"
    out_dir = PREPARED_DIR / key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    max_samples = 200
    print(f"\n[handwritten_forms] Streaming from Teklia/IAM-line (max={max_samples}) ...")
    check_mem("Before")

    ds = load_dataset("Teklia/IAM-line", split="test", streaming=True)
    # For sampling: collect indices first, then stream
    # Since streaming doesn't support random access, take first max_samples
    count = 0

    with open(out_dir / "metadata.jsonl", "w") as mf:
        for i, row in enumerate(ds):
            if count >= max_samples:
                break
            if i % 100 == 0 and i > 0:
                print(f"  ... {count} samples saved, {mem_info()}")

            img = ensure_rgb(row.get("image"))
            if img is None:
                continue

            img_path = img_dir / f"{count:05d}.jpg"
            img.save(str(img_path), format="JPEG", quality=95)

            meta = {
                "idx": count,
                "sample_id": f"handwritten_{i}",
                "ground_truth": row.get("text", ""),
                "metadata": {"index": i},
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1

            del img
            if count % 50 == 0:
                gc.collect()

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  [Done] {count} samples, {total_bytes / 1e6:.1f} MB")
    gc.collect()
    check_mem("After GC")


def main() -> None:
    print(f"Memory limit: {MEM_LIMIT_PCT}%")
    check_mem("Start")

    t0 = time.time()
    prepare_nanonets_kie()
    print(f"  Time: {time.time() - t0:.1f}s")

    t0 = time.time()
    prepare_handwritten()
    print(f"  Time: {time.time() - t0:.1f}s")

    # Summary
    print("\n" + "=" * 50)
    for key in ["nanonets_kie", "handwritten_forms"]:
        d = PREPARED_DIR / key
        meta = d / "metadata.jsonl"
        if meta.exists():
            n = sum(1 for _ in open(meta))
            total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {key:25s} {n:>5} samples  {total_bytes/1e6:>8.1f} MB")
    check_mem("Final")


if __name__ == "__main__":
    main()
