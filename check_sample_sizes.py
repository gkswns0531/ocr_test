#!/usr/bin/env python3
"""Load exactly 1 sample from each dataset and report memory/image sizes."""

from __future__ import annotations

import gc
import io
import sys
import traceback

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

DATA_CACHE_DIR = "/home/ubuntu/ocr_test/data_cache"

DATASETS = [
    ("omnidocbench",      "opendatalab/OmniDocBench",              "train",      "image"),
    ("upstage_dp_bench",  "upstage/dp-bench",                      None,         "pdf"),
    ("ocrbench",          "echo840/OCRBench",                      "test",       "image"),
    ("unimernet",         "deepcopy/UniMER",                       "test",       "image"),
    ("pubtabnet",         "apoidea/pubtabnet-html",                "validation", "image"),
    ("nanonets_kie",      "nanonets/key_information_extraction",   "test",       "image"),
    ("handwritten_forms", "Teklia/IAM-line",                       "test",       "image"),
]


def get_image_size_bytes(img: Image.Image) -> int:
    """Get JPEG-compressed size of a PIL image."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.tell()


def get_raw_pixel_bytes(img: Image.Image) -> int:
    """Get raw uncompressed pixel size."""
    w, h = img.size
    channels = len(img.getbands())
    return w * h * channels


def check_one(name: str, dataset_id: str, split: str | None, img_type: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name} ({dataset_id})")
    print(f"{'='*60}")

    try:
        if img_type == "pdf":
            # DP-Bench: download 1 PDF, convert to image
            import json
            import fitz
            ref_path = hf_hub_download(dataset_id, "dataset/reference.json", repo_type="dataset")
            with open(ref_path) as f:
                ref = json.load(f)
            pdf_name = sorted(ref.keys())[0]
            pdf_path = hf_hub_download(dataset_id, f"dataset/pdfs/{pdf_name}", repo_type="dataset")
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            total_samples = len(ref)
            gt_sample = str(ref[pdf_name])[:200]
        else:
            # Use streaming to avoid loading full dataset
            ds_stream = load_dataset(dataset_id, split=split, streaming=True)
            row = next(iter(ds_stream))
            img = row.get("image")
            if img is None:
                print(f"  [WARN] No 'image' field. Keys: {list(row.keys())}")
                return
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img)) if isinstance(img, bytes) else None
                if img is None:
                    print(f"  [WARN] Cannot convert image")
                    return

            # Get total count (non-streaming, just metadata)
            try:
                from datasets import load_dataset_builder
                builder = load_dataset_builder(dataset_id)
                info = builder.info
                total_samples = info.splits[split].num_examples if info.splits and split in info.splits else "?"
            except Exception:
                total_samples = "?"

            # GT sample
            gt_keys = [k for k in row.keys() if k != "image"]
            gt_sample = {k: str(row[k])[:100] for k in gt_keys[:5]}

        img = img.convert("RGB")
        w, h = img.size
        raw_bytes = get_raw_pixel_bytes(img)
        jpeg_bytes = get_image_size_bytes(img)

        print(f"  Total samples : {total_samples}")
        print(f"  Image size    : {w} x {h}")
        print(f"  Raw pixels    : {raw_bytes / 1e6:.1f} MB")
        print(f"  JPEG (q=95)   : {jpeg_bytes / 1e6:.2f} MB")
        print(f"  GT sample     : {gt_sample}")

        # Estimate full dataset size if we know sample count
        if isinstance(total_samples, int):
            est_mb = total_samples * jpeg_bytes / 1e6
            print(f"  Est. full JPEG: {est_mb:.0f} MB ({est_mb/1e3:.1f} GB)")

    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
    finally:
        gc.collect()


def main():
    for name, dataset_id, split, img_type in DATASETS:
        check_one(name, dataset_id, split, img_type)
    print("\nDone.")


if __name__ == "__main__":
    main()
