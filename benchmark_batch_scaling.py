"""Benchmark: batch size scaling test.

Warmup with batch_size=16, then test 64/128/256/512/1024.
Uses the real pipeline (run_b200_pipeline.py) with a fixed page count.
vLLM server must be running externally before this script.
"""

import json
import os
import re
import subprocess
import sys
import time

# Each config: (batch_size, workers, label)
CONFIGS = [
    (16,   16,   "warmup-16"),
    (64,   64,   "batch-64"),
    (128,  128,  "batch-128"),
    (256,  256,  "batch-256"),
    (512,  512,  "batch-512"),
    (1024, 1024, "batch-1024"),
]

LIMIT = 1024  # pages per test (need >= max batch_size for meaningful results)
GPU_MEM_UTIL = 0.90


def run_pipeline(batch_size: int, workers: int, limit: int) -> dict | None:
    """Run the OCR pipeline and parse performance report from output."""
    cmd = [
        sys.executable, "run_b200_pipeline.py",
        "--step", "ocr",
        "--batch-size", str(batch_size),
        "--workers", str(workers),
        "--limit", str(limit),
        "--no-crops",
        "--gpu-mem-util", str(GPU_MEM_UTIL),
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800,
        cwd="/root/ocr_test",
    )
    wall_time = time.time() - t0
    output = result.stdout + result.stderr

    # Parse performance report
    metrics: dict = {"wall_time_s": wall_time, "batch_size": batch_size, "workers": workers}

    for line in output.splitlines():
        line = line.strip()
        if "OCR total:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                metrics["ocr_total_s"] = float(m.group(1))
        elif "Time to first:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                metrics["time_to_first_s"] = float(m.group(1))
        elif "[SDK] Layout:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                metrics["layout_total_s"] = float(m.group(1))
        elif "[SDK] VLM recog:" in line:
            m = re.search(r"([\d.]+)s", line)
            if m:
                metrics["vlm_total_s"] = float(m.group(1))
        elif "batch" in line and "img/s" in line:
            m = re.search(r"batch\s+(\d+)\s+imgs?:\s+([\d.]+)s\s+\(([\d.]+)\s+img/s\)", line)
            if m:
                batches = metrics.setdefault("layout_batches", [])
                batches.append({
                    "n_imgs": int(m.group(1)),
                    "time_s": float(m.group(2)),
                    "img_per_s": float(m.group(3)),
                })
        elif "[GPU] SM util:" in line:
            m = re.search(r"avg=(\d+)%\s+max=(\d+)%", line)
            if m:
                metrics["gpu_sm_avg"] = int(m.group(1))
                metrics["gpu_sm_max"] = int(m.group(2))
        elif "[GPU] VRAM used:" in line:
            m = re.search(r"max=([\d.]+)GB", line)
            if m:
                metrics["vram_max_gb"] = float(m.group(1))
        elif "Processed" in line and "pages/s" in line:
            m = re.search(r"(\d+)\s+new pages in\s+(\d+)s\s+\(([\d.]+)\s+pages/s\)", line)
            if m:
                metrics["pages_processed"] = int(m.group(1))
                metrics["total_s"] = int(m.group(2))
                metrics["pages_per_s"] = float(m.group(3))

    if "ocr_total_s" not in metrics:
        print(f"  WARNING: Failed to parse metrics. Exit code: {result.returncode}")
        print(f"  Last 20 lines:\n{''.join(output.splitlines()[-20:])}")
        return None

    return metrics


def main() -> None:
    print("=" * 70)
    print("Batch Size Scaling Benchmark")
    print(f"Pages per test: {LIMIT}, GPU mem util: {GPU_MEM_UTIL}")
    print("=" * 70)

    # Verify vLLM server is running
    import requests
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        model = resp.json()["data"][0]["id"]
        print(f"vLLM server: OK (model={model})")
    except Exception as e:
        print(f"ERROR: vLLM server not available: {e}")
        print("Start it first with: python3 run_b200_pipeline.py --step ocr ...")
        sys.exit(1)

    # Delete previous OCR results so each run starts fresh
    results_file = "/root/ocr_test/output/ocr_results.jsonl"

    all_results = []
    for batch_size, workers, label in CONFIGS:
        # Clean previous results
        if os.path.exists(results_file):
            os.remove(results_file)

        print(f"\n{'─' * 60}")
        print(f"Config: {label} (batch_size={batch_size}, workers={workers})")
        print(f"{'─' * 60}")

        metrics = run_pipeline(batch_size, workers, LIMIT)
        if metrics is None:
            print("  SKIPPED (parse failed)")
            continue

        all_results.append({"label": label, **metrics})

        # Print key metrics
        print(f"  OCR total:     {metrics.get('ocr_total_s', '?')}s")
        print(f"  Layout total:  {metrics.get('layout_total_s', '?')}s")
        print(f"  VLM total:     {metrics.get('vlm_total_s', '?')}s")
        print(f"  Pages/s:       {metrics.get('pages_per_s', '?')}")
        print(f"  GPU SM:        avg={metrics.get('gpu_sm_avg', '?')}% max={metrics.get('gpu_sm_max', '?')}%")
        for i, b in enumerate(metrics.get("layout_batches", [])):
            print(f"  Layout batch {i}: {b['n_imgs']} imgs in {b['time_s']}s ({b['img_per_s']} img/s)")

    # Summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    hdr = f"{'Config':<14} {'Batch':>5} {'Workers':>7} {'Total':>7} {'Layout':>7} {'VLM':>7} {'Pages/s':>7} {'SM%':>5} {'VRAM':>6}"
    print(hdr)
    print("─" * 90)
    for r in all_results:
        if r["label"] == "warmup-16":
            continue  # skip warmup from summary
        print(
            f"{r['label']:<14} "
            f"{r['batch_size']:>5} "
            f"{r['workers']:>7} "
            f"{r.get('ocr_total_s', 0):>7.1f} "
            f"{r.get('layout_total_s', 0):>7.1f} "
            f"{r.get('vlm_total_s', 0):>7.1f} "
            f"{r.get('pages_per_s', 0):>7.2f} "
            f"{r.get('gpu_sm_avg', 0):>5} "
            f"{r.get('vram_max_gb', 0):>6.1f}"
        )

    # Layout batch throughput comparison (exclude warmup/first batch)
    print(f"\n{'=' * 70}")
    print("LAYOUT BATCH THROUGHPUT (excluding torch.compile warmup batch)")
    print(f"{'=' * 70}")
    print(f"{'Config':<14} {'Batches':>7} {'Avg img/s':>10} {'Best img/s':>10}")
    print("─" * 50)
    for r in all_results:
        if r["label"] == "warmup-16":
            continue
        batches = r.get("layout_batches", [])
        # Skip first batch (torch.compile warmup)
        steady = batches[1:] if len(batches) > 1 else batches
        if steady:
            avg_ips = sum(b["img_per_s"] for b in steady) / len(steady)
            best_ips = max(b["img_per_s"] for b in steady)
        else:
            avg_ips = batches[0]["img_per_s"] if batches else 0
            best_ips = avg_ips
        print(f"{r['label']:<14} {len(batches):>7} {avg_ips:>10.1f} {best_ips:>10.1f}")

    # Save raw results
    out_path = "/root/ocr_test/benchmark_batch_scaling_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
