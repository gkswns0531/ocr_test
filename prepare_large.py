#!/usr/bin/env python3
"""Prepare omnidocbench and unimernet with ultra-careful streaming."""

from __future__ import annotations

import gc
import io
import json
import sys
import time

import psutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from config import PREPARED_DIR

MEM_LIMIT_PCT = 85.0  # conservative — stop early


def mem() -> tuple[float, str]:
    m = psutil.virtual_memory()
    return m.percent, f"RAM {m.used / 1e9:.1f}/{m.total / 1e9:.1f} GB ({m.percent:.0f}%)"


def guard(label: str) -> None:
    pct, info = mem()
    print(f"  [{label}] {info}", flush=True)
    if pct >= MEM_LIMIT_PCT:
        print(f"\n  *** ABORT: {pct:.0f}% >= {MEM_LIMIT_PCT}% ***", flush=True)
        sys.exit(1)


def to_rgb(img) -> Image.Image | None:
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    return None


# ─── OmniDocBench ─────────────────────────────────────────────────────

def prepare_omnidocbench() -> None:
    from pathlib import Path as _Path

    key = "omnidocbench"
    out_dir = PREPARED_DIR / key
    meta_path = out_dir / "metadata.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Load annotation JSON (small, ~15MB)
    print("  Loading annotations ...", flush=True)
    anno_path = hf_hub_download(
        "opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset"
    )
    with open(anno_path) as f:
        annotations = json.load(f)
    total = len(annotations)
    print(f"  {total} annotations loaded", flush=True)
    guard("After anno")

    # Build stem → annotation index map (annotations use .jpg, HF uses .png)
    stem_to_anno: dict[str, int] = {}
    for ai, anno in enumerate(annotations):
        ip = anno.get("page_info", {}).get("image_path", "")
        if ip:
            stem_to_anno[_Path(ip).stem] = ai
    print(f"  {len(stem_to_anno)} annotation stems mapped", flush=True)

    # Load dataset non-streaming (lazy image loading — metadata only in RAM)
    print("  Loading HF dataset (lazy) ...", flush=True)
    from config import DATA_CACHE_DIR
    ds = load_dataset("opendatalab/OmniDocBench", split="train",
                       cache_dir=str(DATA_CACHE_DIR))
    hf_total = len(ds)
    print(f"  {hf_total} HF images available", flush=True)
    guard("After dataset load")

    # Build HF stem → hf_idx map
    print("  Building filename map ...", flush=True)
    hf_stem_to_idx: dict[str, int] = {}
    for idx in range(hf_total):
        img = ds[idx]["image"]
        if hasattr(img, "filename") and img.filename:
            stem = _Path(img.filename).stem
            hf_stem_to_idx[stem] = idx
        del img
    gc.collect()
    print(f"  {len(hf_stem_to_idx)} HF stems mapped", flush=True)
    guard("After filename map")

    # Iterate annotations, fetch matching HF image, save aligned pairs
    count = 0
    with open(meta_path, "w") as mf:
        for ai in range(total):
            anno = annotations[ai]
            ip = anno.get("page_info", {}).get("image_path", "")
            stem = _Path(ip).stem if ip else ""
            hf_idx = hf_stem_to_idx.get(stem)
            if hf_idx is None:
                continue  # No matching HF image

            # Memory guard every 20 samples
            if count % 20 == 0:
                pct, info = mem()
                print(f"  [{count}/{total}] {info}", flush=True)
                if pct >= MEM_LIMIT_PCT:
                    print(f"\n  *** ABORT at {count}: {pct:.0f}% ***", flush=True)
                    sys.exit(1)

            img = to_rgb(ds[hf_idx]["image"])
            if img is None:
                continue

            # Save image
            save_path = img_dir / f"{count:05d}.jpg"
            img.save(str(save_path), format="JPEG", quality=95)
            del img
            if count % 10 == 0:
                gc.collect()

            # Store full annotation as ground_truth for official element-wise evaluation
            ground_truth = {
                "annotation": anno,
                "annotation_index": ai,
            }

            meta = {
                "idx": count,
                "sample_id": f"omnidocbench_{ai}",
                "ground_truth": ground_truth,
                "metadata": {
                    "index": ai,
                    "hf_index": hf_idx,
                    "image_filename": stem,
                    "page_info": anno.get("page_info", {}),
                },
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            mf.flush()
            count += 1

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  [Done] {count} samples, {total_bytes / 1e6:.0f} MB ({total_bytes / 1e9:.1f} GB)")
    del annotations, ds
    gc.collect()
    guard("After GC")


# ─── UniMERNet ────────────────────────────────────────────────────────

def prepare_unimernet() -> None:
    key = "unimernet"
    out_dir = PREPARED_DIR / key
    meta_path = out_dir / "metadata.jsonl"
    max_samples = 200

    if meta_path.exists():
        n = sum(1 for _ in open(meta_path))
        if n >= max_samples:
            print(f"  [Skip] Already have {n} samples")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"  Streaming deepcopy/UniMER (max={max_samples}) ...", flush=True)
    guard("Before")

    ds = load_dataset("deepcopy/UniMER", split="test", streaming=True)
    count = 0

    with open(meta_path, "w") as mf:
        for i, row in enumerate(ds):
            if count >= max_samples:
                break
            if i % 50 == 0 and i > 0:
                pct, info = mem()
                print(f"  [{count}/{max_samples}] {info}", flush=True)
                if pct >= MEM_LIMIT_PCT:
                    print(f"\n  *** ABORT at {i}: {pct:.0f}% ***", flush=True)
                    sys.exit(1)

            img = to_rgb(row.get("image"))
            if img is None:
                continue

            img_path = img_dir / f"{count:05d}.jpg"
            img.save(str(img_path), format="JPEG", quality=95)

            meta = {
                "idx": count,
                "sample_id": f"unimernet_{i}",
                "ground_truth": row.get("text", ""),
                "metadata": {"index": i},
            }
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1

            del img
            if count % 20 == 0:
                gc.collect()

    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"  [Done] {count} samples, {total_bytes / 1e6:.1f} MB")
    gc.collect()
    guard("After GC")


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    # Parse optional CLI argument to prepare a specific dataset
    target = None
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        valid = {"omnidocbench", "unimernet"}
        if target not in valid:
            print(f"Usage: {sys.argv[0]} [omnidocbench|unimernet]")
            sys.exit(1)

    print(f"Memory limit: {MEM_LIMIT_PCT}%\n")
    guard("Start")

    if target is None or target == "omnidocbench":
        print("\n[1/2] omnidocbench (est. ~4.6 GB, 1358 samples)")
        t0 = time.time()
        prepare_omnidocbench()
        print(f"  Time: {time.time() - t0:.0f}s")

    if target is None or target == "unimernet":
        print("\n[2/2] unimernet (200 samples)")
        t0 = time.time()
        prepare_unimernet()
        print(f"  Time: {time.time() - t0:.0f}s")

    print("\n" + "=" * 50)
    for key in ["omnidocbench", "unimernet"]:
        d = PREPARED_DIR / key
        meta = d / "metadata.jsonl"
        if meta.exists():
            n = sum(1 for _ in open(meta))
            total_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {key:20s} {n:>5} samples  {total_bytes/1e6:>8.1f} MB")
    guard("Final")


if __name__ == "__main__":
    main()
