"""A/B test: max-num-batched-tokens 131072 vs 262144 vs 524288.

Starts vLLM server with each config, runs identical throughput test, compares.
"""

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

VLLM_URL = "http://localhost:8000/v1/chat/completions"
N_REQUESTS = 512
CONCURRENCY = 256
GPU_MEM_UTIL = 0.90

CONFIGS = [32768, 65536, 131072]



def make_images(n: int = 64) -> list[str]:
    """Generate test images."""
    imgs = []
    rng = np.random.RandomState(42)
    for i in range(n):
        w, h = rng.randint(300, 800), rng.randint(200, 500)
        arr = np.full((h, w, 3), 240, dtype=np.uint8)
        for _ in range(10):
            x1, y1 = rng.randint(10, w - 100), rng.randint(10, h - 30)
            arr[y1:y1 + 20, x1:x1 + rng.randint(50, 200)] = rng.randint(0, 80, 3, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=75)
        imgs.append(base64.b64encode(buf.getvalue()).decode())
    return imgs


def start_vllm(batched_tokens: int) -> subprocess.Popen:
    """Start vLLM server and wait until ready."""
    log_path = f"/tmp/vllm_bt{batched_tokens}.log"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "zai-org/GLM-OCR",
        "--served-model-name", "glm-ocr",
        "--port", "8000",
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--max-model-len", "32768",
        "--max-num-batched-tokens", str(batched_tokens),
        "--max-num-seqs", "1024",
        "--trust-remote-code",
        "--enable-chunked-prefill",
        "--quantization", "fp8", "--dtype", "auto",
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    print(f"  Starting vLLM (batched_tokens={batched_tokens}), PID={proc.pid}...")

    for i in range(120):
        time.sleep(3)
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=3)
            if r.status_code == 200:
                # Parse KV cache info from log
                with open(log_path) as f:
                    for line in f:
                        if "Available KV cache memory" in line:
                            print(f"  {line.strip().split('] ')[-1]}")
                        elif "KV cache size" in line:
                            print(f"  {line.strip().split('] ')[-1]}")
                        elif "Maximum concurrency" in line:
                            print(f"  {line.strip().split('] ')[-1]}")
                print(f"  Server ready in {(i+1)*3}s")
                return proc
        except Exception:
            pass

    print(f"  TIMEOUT: Server did not start")
    proc.kill()
    sys.exit(1)


def stop_vllm(proc: subprocess.Popen) -> None:
    """Stop vLLM server."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Also kill any engine core children
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    time.sleep(5)
    # Verify GPU is free
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if out:
        for pid in out.splitlines():
            try:
                os.kill(int(pid.strip()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
        time.sleep(3)


def run_throughput_test(images: list[str]) -> dict:
    """Send N_REQUESTS concurrent and measure throughput."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=CONCURRENCY + 10)
    session.mount("http://", adapter)

    def send(i: int) -> tuple[float, int, int]:
        t0 = time.perf_counter()
        r = session.post(VLLM_URL, json={
            "model": "glm-ocr",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[i % len(images)]}"}},
                {"type": "text", "text": "OCR this image. Output markdown."},
            ]}],
            "max_tokens": 512, "temperature": 0.1,
        }, timeout=120)
        lat = time.perf_counter() - t0
        tok = 0
        if r.status_code == 200:
            tok = r.json().get("usage", {}).get("completion_tokens", 0)
        return lat, r.status_code, tok

    # Warmup (16 sequential to fill prefix cache)
    for i in range(16):
        send(i)

    # Actual test
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(send, i) for i in range(N_REQUESTS)]
        results = [f.result() for f in as_completed(futs)]
    total = time.time() - t0

    lats = [r[0] for r in results]
    ok = sum(1 for r in results if r[1] == 200)
    total_tok = sum(r[2] for r in results)

    session.close()
    return {
        "n_requests": N_REQUESTS,
        "ok": ok,
        "total_s": total,
        "req_per_s": N_REQUESTS / total,
        "tok_per_s": total_tok / total if total > 0 else 0,
        "total_tokens": total_tok,
        "p50_ms": float(np.median(lats)) * 1000,
        "p95_ms": float(np.percentile(lats, 95)) * 1000,
        "p99_ms": float(np.percentile(lats, 99)) * 1000,
    }


def main() -> None:
    print("=" * 70)
    print("A/B Test: max-num-batched-tokens Scaling")
    print(f"Requests: {N_REQUESTS}, Concurrency: {CONCURRENCY}, GPU mem: {GPU_MEM_UTIL}")
    print("=" * 70)

    images = make_images(64)
    print(f"Test images: {len(images)} ready\n")

    all_results = []
    for bt in CONFIGS:
        print(f"\n{'─' * 60}")
        print(f"Config: max-num-batched-tokens = {bt:,}")
        print(f"{'─' * 60}")

        proc = start_vllm(bt)
        result = run_throughput_test(images)
        result["batched_tokens"] = bt
        all_results.append(result)

        print(f"  Results: {result['ok']}/{N_REQUESTS} OK")
        print(f"  Throughput: {result['req_per_s']:.1f} req/s, {result['tok_per_s']:.0f} tok/s")
        print(f"  Latency: p50={result['p50_ms']:.0f}ms, p95={result['p95_ms']:.0f}ms")

        stop_vllm(proc)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Batched Tokens':>16} {'OK':>5} {'Req/s':>8} {'Tok/s':>8} {'p50ms':>8} {'p95ms':>8}")
    print("─" * 60)
    for r in all_results:
        print(f"{r['batched_tokens']:>16,} {r['ok']:>5} {r['req_per_s']:>8.1f} "
              f"{r['tok_per_s']:>8.0f} {r['p50_ms']:>8.0f} {r['p95_ms']:>8.0f}")

    with open("/root/ocr_test/benchmark_batched_tokens_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to benchmark_batched_tokens_results.json")


if __name__ == "__main__":
    main()
