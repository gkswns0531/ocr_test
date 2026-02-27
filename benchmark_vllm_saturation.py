"""Benchmark: vLLM tensor core saturation test.

Send N concurrent OCR requests and monitor GPU SM utilization to verify
that the B200 tensor cores are fully utilized at high batch sizes.
"""

import time
import base64
import io
import json
import subprocess
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
import requests

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "glm-ocr"

OCR_PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings/paragraphs/tables/formulas). "
    "Do not fabricate content that does not exist in the image."
)


def make_test_image(w: int = 600, h: int = 400, seed: int = 0) -> str:
    """Generate a test image with text-like patterns and return base64."""
    rng = np.random.RandomState(seed)
    arr = np.full((h, w, 3), 245, dtype=np.uint8)
    for _ in range(15):
        x1, y1 = rng.randint(20, w - 150), rng.randint(20, h - 50)
        x2, y2 = x1 + rng.randint(80, 300), y1 + rng.randint(15, 40)
        x2, y2 = min(x2, w), min(y2, h)
        arr[y1:y2, x1:x2] = rng.randint(0, 80, 3, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_request(img_b64: str) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }


def send_request(session: requests.Session, req: dict) -> tuple[float, int]:
    """Send one request, return (latency_ms, output_tokens)."""
    t0 = time.perf_counter()
    resp = session.post(VLLM_URL, json=req, timeout=120)
    latency = (time.perf_counter() - t0) * 1000
    if resp.status_code == 200:
        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return latency, tokens
    return latency, 0


def monitor_gpu(interval: float, stop_event: threading.Event) -> list[dict]:
    """Poll nvidia-smi at interval, return list of samples."""
    samples = []
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip()
            parts = out.split(", ")
            samples.append({
                "ts": time.time(),
                "sm_util": int(parts[0]),
                "mem_util": int(parts[1]),
                "mem_used_mb": int(parts[2]),
                "power_w": float(parts[3]),
            })
        except Exception:
            pass
        stop_event.wait(interval)
    return samples


def run_batch(n_requests: int, concurrency: int, images: list[str]) -> dict:
    """Run a batch test with given concurrency."""
    reqs = [build_request(images[i % len(images)]) for i in range(n_requests)]

    # Start GPU monitor
    stop_evt = threading.Event()
    gpu_samples = []
    monitor_thread = threading.Thread(
        target=lambda: gpu_samples.extend(monitor_gpu(0.5, stop_evt)),
        daemon=True,
    )
    monitor_thread.start()

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=concurrency + 10)
    session.mount("http://", adapter)

    latencies = []
    total_tokens = 0

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(send_request, session, req): i for i, req in enumerate(reqs)}
        for f in as_completed(futures):
            lat, tok = f.result()
            latencies.append(lat)
            total_tokens += tok
    t_total = time.perf_counter() - t_start

    stop_evt.set()
    monitor_thread.join(timeout=3)
    session.close()

    # GPU stats (skip first 2 samples — warmup)
    active = gpu_samples[2:] if len(gpu_samples) > 4 else gpu_samples
    sm_utils = [s["sm_util"] for s in active] if active else [0]
    powers = [s["power_w"] for s in active] if active else [0]

    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "total_s": t_total,
        "throughput_req_s": n_requests / t_total,
        "throughput_tok_s": total_tokens / t_total if t_total > 0 else 0,
        "total_tokens": total_tokens,
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "gpu_sm_avg": float(np.mean(sm_utils)),
        "gpu_sm_max": int(np.max(sm_utils)),
        "gpu_sm_min": int(np.min(sm_utils)),
        "gpu_power_avg_w": float(np.mean(powers)),
        "gpu_power_max_w": float(np.max(powers)),
        "gpu_samples": len(active),
    }


def main() -> None:
    print("=" * 70)
    print("vLLM Tensor Core Saturation Test (B200 + GLM-OCR FP8)")
    print("=" * 70)

    # Verify server
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        models = resp.json()
        print(f"Server: OK, model={models['data'][0]['id']}")
    except Exception as e:
        print(f"ERROR: vLLM server not available: {e}")
        sys.exit(1)

    # Generate test images (varied sizes to simulate real crops)
    print("\nGenerating test images...")
    images = []
    rng = np.random.RandomState(42)
    for i in range(32):
        w = rng.randint(200, 1200)
        h = rng.randint(100, 600)
        images.append(make_test_image(w, h, seed=i))
    print(f"  {len(images)} images ready")

    # Warmup (2 sequential requests)
    print("\nWarmup (2 requests)...")
    session = requests.Session()
    for i in range(2):
        req = build_request(images[0])
        resp = session.post(VLLM_URL, json=req, timeout=120)
        print(f"  Warmup {i+1}: status={resp.status_code}")
    session.close()

    # Test configs: (n_requests, concurrency)
    configs = [
        (8, 8),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ]

    results = []
    for n_req, conc in configs:
        print(f"\n{'─' * 60}")
        print(f"Batch: {n_req} requests, {conc} concurrent workers")
        print(f"{'─' * 60}")
        r = run_batch(n_req, conc, images)
        results.append(r)
        print(f"  Total time:      {r['total_s']:>7.1f}s")
        print(f"  Throughput:      {r['throughput_req_s']:>7.1f} req/s, {r['throughput_tok_s']:.0f} tok/s")
        print(f"  Latency p50/95:  {r['latency_p50_ms']:.0f}ms / {r['latency_p95_ms']:.0f}ms")
        print(f"  GPU SM util:     avg={r['gpu_sm_avg']:.0f}%, min={r['gpu_sm_min']}%, max={r['gpu_sm_max']}%")
        print(f"  GPU power:       avg={r['gpu_power_avg_w']:.0f}W, max={r['gpu_power_max_w']:.0f}W")
        print(f"  ({r['gpu_samples']} GPU samples)")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Conc':>6} {'Reqs':>6} {'Time':>7} {'Req/s':>7} {'Tok/s':>8} {'p50ms':>7} {'SM%':>5} {'Power':>6}")
    print(f"{'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*5} {'─'*6}")
    for r in results:
        print(f"{r['concurrency']:>6} {r['n_requests']:>6} {r['total_s']:>7.1f} "
              f"{r['throughput_req_s']:>7.1f} {r['throughput_tok_s']:>8.0f} "
              f"{r['latency_p50_ms']:>7.0f} {r['gpu_sm_avg']:>5.0f} {r['gpu_power_avg_w']:>6.0f}")


if __name__ == "__main__":
    main()
