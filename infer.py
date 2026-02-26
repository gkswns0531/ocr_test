#!/usr/bin/env python3
"""Inference-only script: runs model predictions and saves to disk.

Completely decoupled from evaluation. Predictions are saved to:
    predictions/{model_key}/{benchmark_key}/{file}

OmniDocBench: {original_image_stem}.md  (official naming for pdf_validation.py)
Others:       {sample_id}.txt

Usage:
    python3 infer.py --model deepseek-ocr2 --benchmarks omnidocbench dp_bench
    python3 infer.py --model glm-ocr-pipeline --benchmarks all
    python3 infer.py --model deepseek-ocr2 --benchmarks all --max-samples 5
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from PIL import Image

from config import BENCHMARKS, MODELS, PREPARED_DIR, ModelConfig

# Allow loading very large images (OmniDocBench has 145M-pixel scans).
# VLLMOCRClient resizes for base64 transmission; pipeline SDKs handle internally.
Image.MAX_IMAGE_PIXELS = None

PREDICTIONS_DIR = Path("/home/ubuntu/ocr_test/predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Inference (predictions only)")
    parser.add_argument(
        "--model", required=True,
        choices=list(MODELS.keys()),
        help="Model key to run inference with",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", required=True,
        help="Benchmark(s) to run, or 'all'",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="vLLM server port",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible dataset sampling",
    )
    return parser.parse_args()


def resolve_benchmarks(benchmark_args: list[str]) -> list[str]:
    if "all" in benchmark_args:
        return list(BENCHMARKS.keys())
    for b in benchmark_args:
        if b not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {b}. Available: {list(BENCHMARKS.keys())}")
    return benchmark_args


def _load_prepared_samples(benchmark_key: str) -> list[dict]:
    """Load samples from prepared JSONL + images directory."""
    prepared_dir = PREPARED_DIR / benchmark_key
    meta_path = prepared_dir / "metadata.jsonl"
    img_dir = prepared_dir / "images"

    if not meta_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {meta_path}")

    samples = []
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            img_path = str(img_dir / f"{rec['idx']:05d}.jpg")
            samples.append({
                "idx": rec["idx"],
                "sample_id": rec["sample_id"],
                "ground_truth": rec["ground_truth"],
                "metadata": rec.get("metadata", {}),
                "image_path": img_path,
            })
    return samples


def _get_prediction_path(out_dir: Path, benchmark_key: str, sample: dict) -> Path:
    """Determine the prediction file path based on benchmark type.

    OmniDocBench: {original_image_stem}.md (official naming for pdf_validation.py)
    Others: {sample_id}.txt
    """
    if benchmark_key == "omnidocbench":
        # Use original image filename stem from metadata (matches GT annotation)
        metadata = sample.get("metadata", {})
        image_filename = metadata.get("image_filename", "")
        if not image_filename:
            page_info = metadata.get("page_info", {})
            image_path = page_info.get("image_path", "")
            if image_path:
                image_filename = Path(image_path).stem
        if image_filename:
            return out_dir / f"{image_filename}.md"
        return out_dir / f"{sample['sample_id']}.md"
    else:
        return out_dir / f"{sample['sample_id']}.txt"


def _load_image_original(image_path: str) -> Image.Image:
    """Load image at original resolution (no resizing — official protocol).

    VLLMOCRClient resizes in _image_to_base64() for base64 transmission.
    Pipeline SDKs handle image preprocessing internally.
    """
    return Image.open(image_path).convert("RGB")


def _check_server_alive(port: int) -> bool:
    """Quick health check on vLLM server."""
    import requests
    try:
        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def run_infer_benchmark(
    model_key: str,
    model_config: ModelConfig,
    client,
    benchmark_key: str,
    max_samples: int | None,
    port: int,
    server,
) -> None:
    """Run inference for a single benchmark and save predictions to disk.

    Features:
    - Resume support: skips samples with existing prediction files
    - Consecutive error tracking: restarts server after 3 consecutive failures
    - Max timeout limit: aborts benchmark after 20 total timeouts
    """
    from tqdm import tqdm
    from prompts import get_prompt

    bench_cfg = BENCHMARKS[benchmark_key]
    needs_server = model_config.backend in ("vllm", "glmocr_pipeline")

    # Create output directory
    model_dir_name = model_key.replace("-", "_")
    out_dir = PREDICTIONS_DIR / model_dir_name / benchmark_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load samples
    print(f"[Infer] Loading {bench_cfg.name}...")
    samples = _load_prepared_samples(benchmark_key)
    if max_samples is not None:
        samples = samples[:max_samples]
    print(f"[Infer] {len(samples)} samples loaded")

    # Resume: filter out samples with existing prediction files
    remaining = []
    skipped = 0
    for s in samples:
        pred_path = _get_prediction_path(out_dir, benchmark_key, s)
        if pred_path.exists():
            skipped += 1
        else:
            remaining.append(s)
    if skipped:
        print(f"[Infer] Resuming: {skipped} already done, {len(remaining)} remaining")

    if not remaining:
        print(f"[Infer] All samples already have predictions — skipping {bench_cfg.name}")
        return

    # Inference loop with error tracking
    CONSECUTIVE_ERROR_LIMIT = 3
    MAX_TOTAL_TIMEOUTS = 50

    desc = f"{model_config.name} | {bench_cfg.name}"
    pbar = tqdm(remaining, desc=desc, unit="sample")
    consecutive_errors = 0
    total_timeouts = 0
    total_latency = 0.0
    n_completed = 0

    for sample in pbar:
        pred_path = _get_prediction_path(out_dir, benchmark_key, sample)

        # Get prompt
        question = sample.get("metadata", {}).get("question")
        prompt = get_prompt(bench_cfg.prompt_key, question, model_name=model_config.name)

        # Load image (original resolution)
        try:
            image = _load_image_original(sample["image_path"])
        except Exception as e:
            print(f"\n[Infer] Failed to load image for {sample['sample_id']}: {e}")
            pred_path.write_text("", encoding="utf-8")
            continue

        # Run inference
        try:
            prediction, latency_ms = client.infer(image, prompt)
            total_latency += latency_ms
        except Exception as e:
            print(f"\n[Infer] Inference error for {sample['sample_id']}: {type(e).__name__}: {e}")
            prediction = ""
            latency_ms = 0.0

        del image  # Free memory immediately

        # Track consecutive empty predictions (server hang indicator)
        if not prediction.strip():
            consecutive_errors += 1
            total_timeouts += 1

            if consecutive_errors >= CONSECUTIVE_ERROR_LIMIT:
                if needs_server and not _check_server_alive(port):
                    print(f"\n[Infer] Server dead after {consecutive_errors} consecutive errors — restarting...")
                    server.stop()
                    gc.collect()
                    time.sleep(3)
                    server.start()
                    from client import create_client as _create
                    client = _create(model_config, port=port)
                    consecutive_errors = 0
                else:
                    print(f"\n[Infer] {consecutive_errors} consecutive timeouts but server alive — continuing")
                    consecutive_errors = 0

            if total_timeouts >= MAX_TOTAL_TIMEOUTS:
                print(f"\n[Infer] {total_timeouts} total timeouts — aborting {bench_cfg.name}")
                # Save empty prediction for current sample
                pred_path.write_text(prediction, encoding="utf-8")
                break
        else:
            consecutive_errors = 0

        # Post-process: strip LaTeX display math delimiters for formula benchmarks
        if benchmark_key == "unimernet" and prediction.strip():
            p = prediction.strip()
            if p.startswith("\\[") and p.endswith("\\]"):
                prediction = p[2:-2].strip()
            elif p.startswith("$$") and p.endswith("$$"):
                prediction = p[2:-2].strip()

        # Save prediction as plain text
        pred_path.write_text(prediction, encoding="utf-8")
        n_completed += 1
        pbar.set_postfix(errors=total_timeouts)

    pbar.close()
    avg_latency = total_latency / n_completed if n_completed else 0
    print(f"[Infer] {bench_cfg.name} complete: {n_completed} samples, "
          f"{total_timeouts} timeouts, avg latency {avg_latency:.0f}ms")


def main() -> None:
    args = parse_args()
    benchmark_keys = resolve_benchmarks(args.benchmarks)
    model_key = args.model
    model_config = MODELS[model_key]

    # Set seed for reproducible sampling
    import datasets_loader
    datasets_loader.DEFAULT_SEED = args.seed

    print(f"Model: {model_config.name} ({model_key})")
    print(f"Benchmarks: {benchmark_keys}")
    print(f"Max samples: {args.max_samples}")
    print(f"Port: {args.port}")
    print(f"Seed: {args.seed}")
    print()

    from server import VLLMServer
    from client import create_client

    # Start server if needed
    server = None
    needs_server = model_config.backend in ("vllm", "glmocr_pipeline")
    if needs_server:
        model_config.port = args.port
        server = VLLMServer(model_config, port=args.port)
        server.start()

    client = None
    try:
        client = create_client(model_config, port=args.port)

        for bench_key in benchmark_keys:
            try:
                # Check server health before each benchmark
                if needs_server and not _check_server_alive(args.port):
                    print(f"\n[Infer] Server dead — restarting for {bench_key}...")
                    if server:
                        server.stop()
                    gc.collect()
                    time.sleep(3)
                    server = VLLMServer(model_config, port=args.port)
                    server.start()
                    client = create_client(model_config, port=args.port)

                run_infer_benchmark(
                    model_key=model_key,
                    model_config=model_config,
                    client=client,
                    benchmark_key=bench_key,
                    max_samples=args.max_samples,
                    port=args.port,
                    server=server,
                )
            except Exception as e:
                import traceback
                print(f"[Infer] Benchmark {bench_key} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
            gc.collect()
    finally:
        if client and hasattr(client, "close"):
            client.close()
        if server is not None:
            server.stop()

    print("\nInference complete!")


if __name__ == "__main__":
    main()
