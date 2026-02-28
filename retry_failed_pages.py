#!/usr/bin/env python3
"""Retry failed OCR pages to achieve 100% completion.

Handles two failure types:
1. Layout detection failures: pages with 0 regions but known GT content
   → Send entire page image directly to VLM (bypass layout model)
2. VLM timeout errors: pages with error field set
   → Re-run through normal pipeline

Usage:
    # Ensure vLLM server is already running on port 8000
    python3 retry_failed_pages.py [--port 8000] [--workers 64] [--output-dir output]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_ID = "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark"
OUTPUT_DIR = Path("output")
OCR_RESULTS_FILE = OUTPUT_DIR / "ocr_results.jsonl"
CROPS_DIR = OUTPUT_DIR / "crops"


def find_failed_pages(ocr_path: Path) -> tuple[list[str], list[str]]:
    """Identify pages that need retry.

    Returns:
        (layout_failures, error_pages) - page_ids for each failure type
    """
    layout_failures: list[str] = []
    error_pages: list[str] = []

    with open(ocr_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d["page_id"]
            if d.get("error"):
                error_pages.append(pid)
            elif not d.get("regions") and not d.get("markdown"):
                layout_failures.append(pid)

    return layout_failures, error_pages


def vlm_ocr_whole_page(
    img: Image.Image,
    port: int = 8000,
    max_tokens: int = 8192,
    timeout: int = 300,
) -> str:
    """Send entire page image to VLM for OCR without layout detection."""
    # Convert image to base64
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "model": "glm-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "OCR this document page. Extract all text content preserving structure. Use markdown formatting."},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


def retry_layout_failures(
    page_ids: list[str],
    ds_map: dict[str, int],
    ds,
    port: int,
    workers: int,
    ocr_path: Path,
) -> int:
    """Retry pages where layout detection failed by sending whole page to VLM."""
    if not page_ids:
        print("  No layout failures to retry.")
        return 0

    print(f"  Retrying {len(page_ids)} layout failures (whole-page VLM OCR)...")

    success = 0
    results: list[dict] = []

    def process_one(pid: str) -> dict | None:
        idx = ds_map.get(pid)
        if idx is None:
            return None
        row = ds[idx]
        img = row["image"]
        try:
            markdown = vlm_ocr_whole_page(img, port=port)
            return {
                "page_id": pid,
                "page_idx": idx,
                "regions": [{"index": 0, "label": "text", "native_label": "full_page_ocr", "bbox_2d": [0, 0, 1000, 1000], "content": markdown}],
                "markdown": markdown,
                "image_crops": [],
                "retry_method": "whole_page_vlm",
            }
        except Exception as e:
            print(f"    Failed {pid}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, pid): pid for pid in page_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Layout retry"):
            result = fut.result()
            if result:
                results.append(result)
                success += 1

    # Append new results to file
    if results:
        with open(ocr_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

    print(f"  Layout retry: {success}/{len(page_ids)} recovered")
    return success


def retry_error_pages(
    page_ids: list[str],
    ds_map: dict[str, int],
    ds,
    port: int,
    workers: int,
    ocr_path: Path,
) -> int:
    """Retry pages that had VLM timeout errors using whole-page approach."""
    if not page_ids:
        print("  No error pages to retry.")
        return 0

    print(f"  Retrying {len(page_ids)} error pages...")
    # Same approach as layout failures - whole page VLM OCR
    return retry_layout_failures(page_ids, ds_map, ds, port, workers, ocr_path)


def deduplicate_results(ocr_path: Path) -> int:
    """Remove duplicate page_ids, keeping the last (retry) entry. Sort by page_idx."""
    seen: dict[str, str] = {}  # page_id -> last line
    with open(ocr_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seen[d["page_id"]] = line

    # Sort by page_idx to ensure line N == dataset[N]
    sorted_lines = sorted(seen.values(), key=lambda l: json.loads(l)["page_idx"])

    with open(ocr_path, "w", encoding="utf-8") as f:
        for line in sorted_lines:
            f.write(line + "\n")

    return len(seen)


def find_missing_pages(ocr_path: Path, total_ids: list[str]) -> list[str]:
    """Find page_ids that are not in the results file at all."""
    processed = set()
    if ocr_path.exists():
        with open(ocr_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    processed.add(d["page_id"])
    return [pid for pid in total_ids if pid not in processed]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed OCR pages")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OCR_RESULTS_FILE = OUTPUT_DIR / "ocr_results.jsonl"

    print("=" * 60)
    print("OCR Retry: Achieving 100% Completion")
    print("=" * 60)

    # Load dataset for id→index mapping
    print("\n1. Loading dataset...")
    ds = load_dataset(DATASET_ID, "SDS-KoPub-corpus", split="test", cache_dir="data")
    all_ids = [ds[i]["id"] for i in range(len(ds))]
    ds_map = {pid: i for i, pid in enumerate(all_ids)}
    print(f"   Total pages in dataset: {len(all_ids)}")

    # Find failures
    print("\n2. Analyzing failures...")
    layout_fails, error_pages = find_failed_pages(OCR_RESULTS_FILE)
    missing = find_missing_pages(OCR_RESULTS_FILE, all_ids)
    print(f"   Layout failures (0 regions): {len(layout_fails)}")
    print(f"   Error pages:                 {len(error_pages)}")
    print(f"   Missing pages (not in file): {len(missing)}")

    total_to_fix = len(layout_fails) + len(error_pages) + len(missing)
    if total_to_fix == 0:
        print("\n   All pages accounted for! No retry needed.")
        return

    # Retry layout failures
    print("\n3. Retrying layout failures...")
    n1 = retry_layout_failures(layout_fails, ds_map, ds, args.port, args.workers, OCR_RESULTS_FILE)

    # Retry error pages
    print("\n4. Retrying error pages...")
    n2 = retry_error_pages(error_pages, ds_map, ds, args.port, args.workers, OCR_RESULTS_FILE)

    # Retry missing pages
    if missing:
        print(f"\n5. Retrying {len(missing)} missing pages...")
        n3 = retry_layout_failures(missing, ds_map, ds, args.port, args.workers, OCR_RESULTS_FILE)
    else:
        n3 = 0

    # Deduplicate
    print("\n6. Deduplicating results...")
    unique = deduplicate_results(OCR_RESULTS_FILE)
    print(f"   Unique pages in file: {unique}/{len(all_ids)}")
    completion = unique / len(all_ids) * 100
    print(f"   Completion rate: {completion:.1f}%")

    # Regenerate parsed_texts
    if n1 + n2 + n3 > 0:
        parsed_path = OUTPUT_DIR / "parsed_texts.jsonl"
        if parsed_path.exists():
            parsed_path.unlink()
            print(f"   Removed {parsed_path} (will be regenerated by embedding step)")

    print(f"\n{'=' * 60}")
    print(f"Retry complete! Recovered {n1 + n2 + n3} pages. Completion: {completion:.1f}%")


if __name__ == "__main__":
    main()
