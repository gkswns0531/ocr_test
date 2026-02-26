#!/usr/bin/env python3
"""
GLM-OCR Pipeline Latency Profiler (Detailed)

Measures per-stage latency with fine-grained breakdown:
  1. Image load from Arrow (PIL decode)
  2. Image save to disk (PIL → JPEG)
  3. Layout detection (PP-DocLayoutV3)
     - Preprocessing (resize, normalize)
     - Model inference
     - Postprocessing (NMS, bbox)
  4. Region cropping (bbox crop from original image)
  5. Image encoding (resize + base64 for VLM)
  6. VLM HTTP request (vLLM GLM-OCR per region)
  7. Result formatting (JSON parse, markdown)
  8. Record creation (page_result_to_record)
  9. Cleanup (file delete, gc)

Uses monkey-patching to instrument SDK internals.
Runs on diverse pages (text-heavy, table, image, mixed).

Usage:
  # vLLM server must already be running
  python3 profile_pipeline.py [--pages N] [--batch-size B] [--port PORT]
"""

import argparse
import os
import sys
import time
import tempfile
import shutil
import threading
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from datasets import Dataset


# ── Timing accumulator ──

@dataclass
class TimingBucket:
    """Thread-safe timing accumulator."""
    times: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, elapsed: float):
        with self._lock:
            self.times.append(elapsed)

    def reset(self):
        with self._lock:
            self.times.clear()

    @property
    def count(self):
        return len(self.times)

    @property
    def total(self):
        return sum(self.times)

    def stats(self):
        if not self.times:
            return {}
        arr = np.array(self.times)
        return {
            "count": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "total": float(arr.sum()),
        }


# Global timing buckets
T = {
    "arrow_load":       TimingBucket(),  # Arrow → PIL Image
    "img_save":         TimingBucket(),  # PIL → JPEG on disk
    "layout_detect":    TimingBucket(),  # PP-DocLayoutV3 batch
    "layout_preproc":   TimingBucket(),  # layout internal preprocess
    "layout_infer":     TimingBucket(),  # layout model forward
    "layout_postproc":  TimingBucket(),  # layout NMS/postprocess
    "region_crop":      TimingBucket(),  # crop_image_region per region
    "img_encode":       TimingBucket(),  # resize + base64 encode per region
    "img_resize":       TimingBucket(),  # just the resize portion
    "vlm_http":         TimingBucket(),  # vLLM HTTP request per region
    "vlm_build_req":    TimingBucket(),  # build_request_from_image
    "result_format":    TimingBucket(),  # result formatter
    "record_create":    TimingBucket(),  # page_result_to_record
    "cleanup":          TimingBucket(),  # file cleanup + gc
    "parse_total":      TimingBucket(),  # ocr.parse() total
}

vlm_status_codes = defaultdict(int)
vlm_status_lock = threading.Lock()


def install_hooks(pipeline):
    """Monkey-patch SDK pipeline components for detailed timing."""

    # ── 1. Layout detector ──
    layout = pipeline.layout_detector

    # Try to instrument internal steps of layout detector
    orig_layout_process = layout.process

    def timed_layout_process(images, **kwargs):
        t0 = time.perf_counter()
        result = orig_layout_process(images, **kwargs)
        T["layout_detect"].add(time.perf_counter() - t0)
        return result

    layout.process = timed_layout_process

    # ── 2. Region crop (in pipeline) ──
    import glmocr.utils.image_utils as img_mod
    orig_crop = img_mod.crop_image_region

    def timed_crop(*a, **kw):
        t0 = time.perf_counter()
        result = orig_crop(*a, **kw)
        T["region_crop"].add(time.perf_counter() - t0)
        return result

    img_mod.crop_image_region = timed_crop
    # Also patch where it's imported in pipeline module
    import glmocr.pipeline.pipeline as pp_mod
    pp_mod.crop_image_region = timed_crop

    # ── 3. Image encode (load_image_to_base64) ──
    orig_encode = img_mod.load_image_to_base64

    def timed_encode(*a, **kw):
        t0 = time.perf_counter()
        result = orig_encode(*a, **kw)
        T["img_encode"].add(time.perf_counter() - t0)
        return result

    img_mod.load_image_to_base64 = timed_encode
    pp_mod.load_image_to_base64 = timed_encode
    # Also patch in page_loader where it's imported
    import glmocr.dataloader.page_loader as pl_mod
    pl_mod.load_image_to_base64 = timed_encode

    # ── 4. VLM HTTP call (ocr_client.process) ──
    orig_ocr_process = pipeline.ocr_client.process

    def timed_ocr_process(*a, **kw):
        t0 = time.perf_counter()
        result = orig_ocr_process(*a, **kw)
        T["vlm_http"].add(time.perf_counter() - t0)
        status = result[1] if isinstance(result, tuple) else 0
        with vlm_status_lock:
            vlm_status_codes[status] += 1
        return result

    pipeline.ocr_client.process = timed_ocr_process

    # ── 5. build_request_from_image ──
    orig_build = pipeline.page_loader.build_request_from_image

    def timed_build(*a, **kw):
        t0 = time.perf_counter()
        result = orig_build(*a, **kw)
        T["vlm_build_req"].add(time.perf_counter() - t0)
        return result

    pipeline.page_loader.build_request_from_image = timed_build

    # ── 6. Result formatter ──
    orig_format = pipeline.result_formatter.format_ocr_result
    orig_format_multi = pipeline.result_formatter.format_multi_page_results
    orig_process_fmt = pipeline.result_formatter.process

    def timed_format(*a, **kw):
        t0 = time.perf_counter()
        result = orig_format(*a, **kw)
        T["result_format"].add(time.perf_counter() - t0)
        return result

    def timed_format_multi(*a, **kw):
        t0 = time.perf_counter()
        result = orig_format_multi(*a, **kw)
        T["result_format"].add(time.perf_counter() - t0)
        return result

    def timed_process_fmt(*a, **kw):
        t0 = time.perf_counter()
        result = orig_process_fmt(*a, **kw)
        T["result_format"].add(time.perf_counter() - t0)
        return result

    pipeline.result_formatter.format_ocr_result = timed_format
    pipeline.result_formatter.format_multi_page_results = timed_format_multi
    pipeline.result_formatter.process = timed_process_fmt

    # ── 7. Try to instrument layout detector internals ──
    try:
        from glmocr.layout.layout_detector import PPDocLayoutDetector
        # Check if we can patch _preprocess, _infer, _postprocess
        if hasattr(layout, '_detector') and hasattr(layout._detector, 'predict'):
            orig_predict = layout._detector.predict

            def timed_predict(*a, **kw):
                t0 = time.perf_counter()
                result = orig_predict(*a, **kw)
                T["layout_infer"].add(time.perf_counter() - t0)
                return result

            layout._detector.predict = timed_predict
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Profile GLM-OCR pipeline latency")
    parser.add_argument("--pages", type=int, default=32, help="Number of pages to profile")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--diverse", action="store_true", default=True,
                        help="Sample diverse pages (text/table/chart)")
    args = parser.parse_args()

    # ── Find Arrow data ──
    data_dir = Path("data")
    arrow_dir = None
    for root, _, files in os.walk(data_dir):
        if any(f.endswith(".arrow") for f in files):
            arrow_dir = Path(root)
            break
    if arrow_dir is None:
        sys.exit("No arrow files found under data/")

    arrow_files = sorted(f for f in os.listdir(arrow_dir) if f.endswith(".arrow"))

    # Load from multiple shards for diversity
    print("Loading pages for profiling...")
    all_rows = []
    for af in arrow_files[:3]:  # first 3 shards
        ds = Dataset.from_file(str(arrow_dir / af))
        shard_len = len(ds)
        # Sample evenly: start, 25%, 50%, 75%
        indices = [0, shard_len // 4, shard_len // 2, 3 * shard_len // 4]
        # Plus some random
        np.random.seed(42)
        indices += list(np.random.choice(shard_len, min(20, shard_len), replace=False))
        indices = sorted(set(int(i) for i in indices if i < shard_len))
        for i in indices:
            all_rows.append(ds[i])
            if len(all_rows) >= args.pages:
                break
        del ds
        if len(all_rows) >= args.pages:
            break

    n_pages = min(args.pages, len(all_rows))
    all_rows = all_rows[:n_pages]

    # Show page diversity
    sizes = [(r["image"].size[0] * r["image"].size[1]) for r in all_rows]
    print(f"  {n_pages} pages loaded from {min(3, len(arrow_files))} shards")
    print(f"  Image sizes: min={min(sizes):,}px, max={max(sizes):,}px, "
          f"mean={np.mean(sizes):,.0f}px")

    # ── Patch SDK config ──
    sdk_config = Path("/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml")
    cfg_dir = Path(tempfile.mkdtemp(prefix="glmocr_prof_"))
    cfg_path = cfg_dir / "config.yaml"
    shutil.copy(sdk_config, cfg_path)
    text = cfg_path.read_text()
    text = text.replace("api_port: 8080", f"api_port: {args.port}")
    text = text.replace("api_port: 5002", f"api_port: {args.port}")
    text = text.replace("port: 5002", f"port: {args.port}")
    text = text.replace("max_tokens: 4096", "max_tokens: 16384")
    text = text.replace("max_workers: 32", f"max_workers: {args.workers}")
    text = text.replace("batch_size: 1", f"batch_size: {args.batch_size}")
    cfg_path.write_text(text)

    # ── Initialize SDK + hooks ──
    from glmocr import GlmOcr
    ocr = GlmOcr(config_path=str(cfg_path))
    install_hooks(ocr._pipeline)
    print("  SDK initialized, hooks installed")

    # ── Import pipeline helpers ──
    from run_b200_pipeline import (
        page_result_to_record, _release_memory, ensure_dir,
    )

    # ── Warmup (1 page) ──
    print("\nWarmup (1 page)...")
    warmup_img = all_rows[0]["image"].convert("RGB")
    warmup_path = "/tmp/prof_warmup.jpg"
    warmup_img.save(warmup_path, format="JPEG", quality=95)
    try:
        ocr.parse(warmup_path)
    except Exception:
        pass
    os.remove(warmup_path)
    # Reset all timers after warmup
    for bucket in T.values():
        bucket.reset()
    vlm_status_codes.clear()

    # ── Run profiling ──
    batch_size = args.batch_size
    n_batches = (n_pages + batch_size - 1) // batch_size
    batch_wall_times = []
    regions_per_page = []

    print(f"\nProfiling {n_pages} pages in {n_batches} batches (batch_size={batch_size})")
    print("=" * 100)
    print(f"{'Batch':>5} | {'ArrowLoad':>9} | {'ImgSave':>8} | {'Parse':>8} | "
          f"{'Layout':>8} | {'Crop':>6} | {'Encode':>8} | {'VLM':>8} | "
          f"{'Format':>7} | {'Record':>7} | {'Rgns':>5}")
    print("-" * 100)

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_pages)
        batch_pages = end - start

        # Snapshot counters before batch
        snap = {k: v.count for k, v in T.items()}
        snap_total = {k: v.total for k, v in T.items()}

        # ── Stage 1: Arrow load (PIL decode) ──
        pil_images = []
        t0 = time.perf_counter()
        for i in range(start, end):
            t_load = time.perf_counter()
            pil_img = all_rows[i]["image"].convert("RGB")
            T["arrow_load"].add(time.perf_counter() - t_load)
            pil_images.append(pil_img)

        # ── Stage 2: Save to disk ──
        tmp_paths = []
        for i, pil_img in enumerate(pil_images):
            t_save = time.perf_counter()
            tmp_path = f"/tmp/prof_batch_{batch_idx}_{i}.jpg"
            pil_img.save(tmp_path, format="JPEG", quality=95)
            T["img_save"].add(time.perf_counter() - t_save)
            tmp_paths.append(tmp_path)
        del pil_images

        # ── Stage 3: ocr.parse() (layout + VLM + format) ──
        t_parse_start = time.perf_counter()
        try:
            if len(tmp_paths) == 1:
                results = [ocr.parse(tmp_paths[0])]
            else:
                results = ocr.parse(tmp_paths)
        except Exception as e:
            print(f"  ERROR batch {batch_idx}: {e}")
            results = [None] * batch_pages
        T["parse_total"].add(time.perf_counter() - t_parse_start)

        # ── Stage 4: Record creation ──
        batch_regions = 0
        for idx, (result, tmp_path) in enumerate(zip(results, tmp_paths)):
            if result is None:
                continue
            t_rec = time.perf_counter()
            record = page_result_to_record(
                page_id=f"prof_{start + idx}",
                page_idx=start + idx,
                result=result,
                img_path=tmp_path,
                do_crops=False,
            )
            T["record_create"].add(time.perf_counter() - t_rec)
            n_rgns = len(record.get("regions", []))
            regions_per_page.append(n_rgns)
            batch_regions += n_rgns

        # ── Stage 5: Cleanup ──
        t_clean = time.perf_counter()
        for tmp in tmp_paths:
            try:
                os.remove(tmp)
            except OSError:
                pass
        _release_memory()
        T["cleanup"].add(time.perf_counter() - t_clean)

        batch_wall = time.perf_counter() - t0
        batch_wall_times.append(batch_wall)

        # Delta for this batch
        def delta(key):
            return T[key].total - snap_total.get(key, 0)

        print(f"{batch_idx+1:>3}/{n_batches:>2} | "
              f"{delta('arrow_load'):>8.3f}s | "
              f"{delta('img_save'):>7.3f}s | "
              f"{delta('parse_total'):>7.2f}s | "
              f"{delta('layout_detect'):>7.3f}s | "
              f"{delta('region_crop'):>5.3f}s | "
              f"{delta('img_encode'):>7.3f}s | "
              f"{delta('vlm_http'):>7.2f}s | "
              f"{delta('result_format'):>6.3f}s | "
              f"{delta('record_create'):>6.3f}s | "
              f"{batch_regions:>5}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    total_wall = sum(batch_wall_times)

    print("\n" + "=" * 100)
    print("DETAILED LATENCY SUMMARY")
    print("=" * 100)

    print(f"\n  Pages: {n_pages}, Batches: {n_batches}, Batch size: {batch_size}")
    print(f"  Total wall time: {total_wall:.1f}s")
    print(f"  Throughput: {n_pages / total_wall:.2f} pages/s")
    print(f"  Avg per page: {total_wall / n_pages:.2f}s")

    def print_bucket(label, key, indent=2):
        s = T[key].stats()
        if not s:
            print(f"{'':>{indent}}{label:.<40s} (no data)")
            return
        print(f"{'':>{indent}}{label:.<40s} "
              f"total={s['total']:>7.2f}s  "
              f"mean={s['mean']*1000:>7.1f}ms  "
              f"med={s['median']*1000:>7.1f}ms  "
              f"p95={s['p95']*1000:>7.1f}ms  "
              f"max={s['max']*1000:>7.1f}ms  "
              f"n={s['count']}")

    print(f"\n{'─'*100}")
    print("PER-STAGE STATISTICS")
    print(f"{'─'*100}")

    print("\n  [Pre-processing]")
    print_bucket("Arrow → PIL decode", "arrow_load")
    print_bucket("PIL → JPEG save", "img_save")

    print("\n  [Layout Detection]")
    print_bucket("Layout batch (total)", "layout_detect")
    print_bucket("Layout model inference", "layout_infer")

    print("\n  [VLM Recognition]")
    print_bucket("Region crop", "region_crop")
    print_bucket("Image encode (resize+b64)", "img_encode")
    print_bucket("Build VLM request", "vlm_build_req")
    print_bucket("VLM HTTP call", "vlm_http")

    print("\n  [Post-processing]")
    print_bucket("Result formatting", "result_format")
    print_bucket("Record creation", "record_create")
    print_bucket("Cleanup (rm + gc)", "cleanup")

    print("\n  [End-to-end]")
    print_bucket("ocr.parse() total", "parse_total")

    # ── Percentage breakdown ──
    print(f"\n{'─'*100}")
    print("TIME BREAKDOWN (% of wall time)")
    print(f"{'─'*100}")

    categories = [
        ("Arrow load",       "arrow_load"),
        ("Image save",       "img_save"),
        ("Layout detection", "layout_detect"),
        ("Region crop",      "region_crop"),
        ("Image encode",     "img_encode"),
        ("VLM HTTP",         "vlm_http"),
        ("Build request",    "vlm_build_req"),
        ("Result format",    "result_format"),
        ("Record create",    "record_create"),
        ("Cleanup",          "cleanup"),
    ]

    accounted = 0
    for label, key in categories:
        t = T[key].total
        pct = 100 * t / total_wall if total_wall > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<20s} {t:>7.2f}s  ({pct:>5.1f}%)  {bar}")
        accounted += t

    # VLM is concurrent (thread pool), so sum of individual > wall time
    # Show "concurrency factor"
    vlm_total = T["vlm_http"].total
    parse_total = T["parse_total"].total
    if vlm_total > 0 and parse_total > 0:
        concurrency = vlm_total / parse_total
        print(f"\n  VLM concurrency factor: {concurrency:.1f}x "
              f"(sum of VLM calls / parse wall time)")

    # ── Region statistics ──
    print(f"\n{'─'*100}")
    print("REGION & VLM STATISTICS")
    print(f"{'─'*100}")
    if regions_per_page:
        arr = np.array(regions_per_page)
        print(f"  Regions/page: mean={arr.mean():.1f}, med={np.median(arr):.0f}, "
              f"min={arr.min()}, max={arr.max()}, total={arr.sum()}")
    print(f"  VLM calls: {T['vlm_http'].count}")
    print(f"  VLM status codes: {dict(vlm_status_codes)}")
    if T["vlm_http"].count > 0:
        print(f"  VLM avg latency: {T['vlm_http'].total / T['vlm_http'].count * 1000:.0f}ms")

    # ── Bottleneck analysis ──
    print(f"\n{'─'*100}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'─'*100}")

    # Find the dominant stage
    stage_times = [(label, T[key].total) for label, key in categories]
    stage_times.sort(key=lambda x: -x[1])
    print("\n  Top stages by total time:")
    for i, (label, t) in enumerate(stage_times[:5]):
        pct = 100 * t / total_wall if total_wall > 0 else 0
        print(f"    {i+1}. {label:<20s} {t:>7.2f}s ({pct:.1f}%)")

    # ── Estimated full run ──
    rate = n_pages / total_wall
    est_40k = 40781 / rate / 3600
    print(f"\n  Estimated 40K pages: {est_40k:.1f}h at {rate:.2f} pages/s")

    ocr.close()
    shutil.rmtree(cfg_dir, ignore_errors=True)
    print("\nDone!")


if __name__ == "__main__":
    main()
