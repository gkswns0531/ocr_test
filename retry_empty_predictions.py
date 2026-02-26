#!/usr/bin/env python3
"""Retry empty predictions (vLLM timeouts) by re-running inference on failed samples only.

Memory-efficient: loads only the specific failed samples from prepared_datasets,
one at a time, to avoid OOM when vLLM server is running.
"""

import json
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from PIL import Image

from config import MODELS, BENCHMARKS, RESULTS_DIR, PREPARED_DIR
from server import VLLMServer
from client import VLLMOCRClient
from prompts import get_prompt
from benchmarks import _get_evaluator, _aggregate_metrics

TIMEOUT_THRESHOLD_MS = 100_000


def get_timeout_empty_ids(result_path: str) -> list[str]:
    """Get sample IDs with empty predictions from timeout (latency >= 100s)."""
    with open(result_path) as f:
        data = json.load(f)
    ids = []
    for s in data.get("per_sample_results", []):
        pred = s.get("prediction", "")
        latency = s.get("latency_ms", 0)
        if (not pred or not pred.strip()) and latency >= TIMEOUT_THRESHOLD_MS and not s.get("error"):
            ids.append(s["sample_id"])
    return ids


def load_single_sample(bench_key: str, sample_id: str):
    """Load a single sample from prepared_datasets (memory-efficient)."""
    prep_dir = PREPARED_DIR / bench_key
    meta_path = prep_dir / "metadata.jsonl"
    img_dir = prep_dir / "images"

    with open(meta_path) as f:
        for line in f:
            row = json.loads(line)
            if row["sample_id"] == sample_id:
                # Try common naming patterns
                idx = row["idx"]
                for fmt in [f"{idx:05d}.jpg", f"{idx:05d}.png", f"{idx:06d}.jpg", f"{idx:06d}.png"]:
                    img_path = img_dir / fmt
                    if img_path.exists():
                        image = Image.open(img_path).convert("RGB")
                        return {
                            "image": image,
                            "ground_truth": row["ground_truth"],
                            "metadata": row.get("metadata", {}),
                            "sample_id": row["sample_id"],
                        }
                print(f"    WARNING: no image found for idx={idx} in {img_dir}")
                return None
    return None


def retry_model(model_key: str, port: int = 8000):
    """Retry all timeout-empty predictions for one model."""
    model_config = MODELS[model_key]
    model_safe = model_config.name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

    # Collect benchmarks with timeout empties
    bench_retries = {}
    for bench_key in BENCHMARKS:
        rpath = RESULTS_DIR / f"{model_safe}_{bench_key}.json"
        if not rpath.exists():
            continue
        empty_ids = get_timeout_empty_ids(str(rpath))
        if empty_ids:
            bench_retries[bench_key] = empty_ids
            print(f"  {bench_key}: {len(empty_ids)} timeout-empties")

    if not bench_retries:
        print(f"  No timeout-empties for {model_config.name}")
        return

    total = sum(len(v) for v in bench_retries.values())
    print(f"\n  Starting vLLM for {model_config.name} ({total} samples)...")

    # Check memory before starting
    import subprocess
    mem = subprocess.run(["free", "-m"], capture_output=True, text=True)
    print(f"  Memory before start:\n{mem.stdout}")

    model_config.port = port
    server = VLLMServer(model_config, port=port)
    server.start()

    try:
        client = VLLMOCRClient(
            base_url=f"http://localhost:{port}/v1",
            model_name=model_config.model_id,
        )

        for bench_key, empty_ids in bench_retries.items():
            _retry_bench(client, model_config.name, model_safe, bench_key, empty_ids)

    finally:
        server.stop()
        gc.collect()
        time.sleep(5)


def _retry_bench(client, model_name, model_safe, bench_key, empty_ids):
    """Retry specific samples for one benchmark, loading one at a time."""
    bench_cfg = BENCHMARKS[bench_key]
    evaluator = _get_evaluator(bench_cfg)

    rpath = RESULTS_DIR / f"{model_safe}_{bench_key}.json"
    with open(rpath) as f:
        data = json.load(f)

    psr = data.get("per_sample_results", [])
    psr_map = {s["sample_id"]: i for i, s in enumerate(psr)}

    retried = 0
    still_empty = 0

    for sid in empty_ids:
        # Load only this one sample
        sample = load_single_sample(bench_key, sid)
        if sample is None:
            print(f"    {sid}: not found in prepared_datasets, skipping")
            continue

        question = sample["metadata"].get("question") if sample["metadata"] else None
        prompt_text = get_prompt(bench_cfg.prompt_key, question, model_name=model_name)
        prediction, latency_ms = client.infer(sample["image"], prompt_text)

        # Free the image immediately
        del sample["image"]
        gc.collect()

        idx = psr_map.get(sid)
        if idx is None:
            continue

        if prediction and prediction.strip():
            try:
                scores = evaluator(prediction, sample["ground_truth"], sample["metadata"])
            except Exception as e:
                scores = {}
                print(f"    {sid}: eval error: {e}")

            psr[idx]["prediction"] = prediction
            psr[idx]["latency_ms"] = latency_ms
            psr[idx]["scores"] = scores
            retried += 1
            print(f"    {sid}: OK ({latency_ms:.0f}ms, len={len(prediction)})")
        else:
            still_empty += 1
            if latency_ms >= TIMEOUT_THRESHOLD_MS:
                print(f"    {sid}: timeout again ({latency_ms:.0f}ms)")
            else:
                print(f"    {sid}: still empty ({latency_ms:.0f}ms)")

    print(f"  [{bench_key}] {retried} fixed, {still_empty} still empty")

    if retried > 0:
        data["per_sample_results"] = psr
        data["metrics"] = _aggregate_metrics(psr, bench_cfg.metric_type)
        with open(rpath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [{bench_key}] Saved updated results")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["glm-ocr", "paddleocr-vl", "deepseek-ocr2"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"Retrying timeouts for: {model_key}")
        print(f"{'='*60}")
        retry_model(model_key, port=args.port)
        # Memory check between models
        import subprocess
        mem = subprocess.run(["free", "-m"], capture_output=True, text=True)
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True)
        print(f"\n  Memory after model: RAM={mem.stdout.split(chr(10))[1].split()[6]}Mi free, GPU={gpu.stdout.strip()}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
