#!/usr/bin/env python3
"""OCR Model Benchmarking System — Main Orchestrator."""

from __future__ import annotations

import argparse
import json
import time

from config import BENCHMARKS, MODELS, RESULTS_DIR
from server import VLLMServer
from client import VLLMOCRClient, MinerUClient
from benchmarks import BenchmarkResult, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Model Benchmarking System")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["glm-ocr", "paddleocr-vl", "deepseek-ocr2", "mineru"],
        choices=list(MODELS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to run (or 'all')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max samples per benchmark (None = use config defaults)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset sampling (default: 42)",
    )
    return parser.parse_args()


def resolve_benchmarks(benchmark_args: list[str]) -> list[str]:
    """Resolve benchmark names, expanding 'all'."""
    if "all" in benchmark_args:
        return list(BENCHMARKS.keys())
    for b in benchmark_args:
        if b not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {b}. Available: {list(BENCHMARKS.keys())}")
    return benchmark_args


def set_seed(seed: int = 42) -> None:
    """Set global seed for reproducible dataset sampling across models."""
    import datasets_loader
    datasets_loader.DEFAULT_SEED = seed


def _check_server_alive(port: int) -> bool:
    """Quick health check on vLLM server."""
    import requests as _requests
    try:
        resp = _requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def run_model_benchmarks(
    model_key: str,
    benchmark_keys: list[str],
    port: int,
    max_samples: int | None,
    resume: bool,
) -> list[BenchmarkResult]:
    """Run all benchmarks for a single model.

    Datasets are loaded on-demand per benchmark to avoid RAM exhaustion.
    Same seed-based sampling ensures identical samples across models.
    Server is restarted automatically if it dies between benchmarks.
    """
    import gc
    model_config = MODELS[model_key]
    results: list[BenchmarkResult] = []

    if model_config.backend in ("vllm", "glmocr_pipeline"):
        model_config.port = port
        server = VLLMServer(model_config, port=port)
        server.start()

        client = None
        try:
            if model_config.backend == "glmocr_pipeline":
                from client import GLMOCRPipelineClient
                client = GLMOCRPipelineClient(vllm_port=port)
            else:
                client = VLLMOCRClient(
                    base_url=f"http://localhost:{port}/v1",
                    model_name=model_config.model_id,
                )
            for bench_key in benchmark_keys:
                # Restart server if it died during previous benchmark
                if not _check_server_alive(port):
                    print(f"\n[Main] Server dead — restarting for {bench_key}...")
                    server.stop()
                    gc.collect()
                    time.sleep(3)
                    server.start()
                    if model_config.backend == "glmocr_pipeline":
                        client = GLMOCRPipelineClient(vllm_port=port)
                    else:
                        client = VLLMOCRClient(
                            base_url=f"http://localhost:{port}/v1",
                            model_name=model_config.model_id,
                        )

                try:
                    result = run_benchmark(
                        client=client,
                        model_name=model_config.name,
                        benchmark_key=bench_key,
                        max_samples_override=max_samples,
                        resume=resume,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"[Main] Benchmark {bench_key} failed: {type(e).__name__}: {e}")
                gc.collect()
        finally:
            server.stop()
            if client and hasattr(client, 'close'):
                client.close()
    elif model_config.backend == "paddleocr_pipeline":
        from client import PaddleOCRVLPipelineClient
        client = PaddleOCRVLPipelineClient()
        for bench_key in benchmark_keys:
            result = run_benchmark(
                client=client,
                model_name=model_config.name,
                benchmark_key=bench_key,
                max_samples_override=max_samples,
                resume=resume,
            )
            results.append(result)
            gc.collect()

    return results


def generate_comparison_table(all_results: list[BenchmarkResult]) -> None:
    """Print a comparison table and save as JSON."""
    if not all_results:
        print("No results to compare.")
        return

    # Group by model
    by_model: dict[str, dict[str, dict]] = {}
    for r in all_results:
        if r.model_name not in by_model:
            by_model[r.model_name] = {}
        by_model[r.model_name][r.benchmark_name] = r.metrics

    # Collect all benchmark names
    bench_names = []
    for model_results in by_model.values():
        for bn in model_results:
            if bn not in bench_names:
                bench_names.append(bn)

    # Print markdown table
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 80)

    # Collect all metric keys per benchmark
    for bench_name in bench_names:
        print(f"\n### {bench_name}")
        # Get all metric keys for this benchmark
        metric_keys: list[str] = []
        for model_results in by_model.values():
            if bench_name in model_results:
                for k in model_results[bench_name]:
                    if k not in metric_keys:
                        metric_keys.append(k)

        if not metric_keys:
            continue

        # Header
        header = "| Model | " + " | ".join(metric_keys) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(metric_keys)) + "|"
        print(header)
        print(sep)

        # Rows
        for model_name, model_results in by_model.items():
            metrics = model_results.get(bench_name, {})
            vals = []
            for k in metric_keys:
                v = metrics.get(k, "-")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            row = f"| {model_name} | " + " | ".join(vals) + " |"
            print(row)

    # Save comparison JSON
    comparison = {
        "models": list(by_model.keys()),
        "benchmarks": bench_names,
        "results": {
            model: {bench: metrics for bench, metrics in benches.items()}
            for model, benches in by_model.items()
        },
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"comparison_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {out_path}")


def main() -> None:
    args = parse_args()
    benchmark_keys = resolve_benchmarks(args.benchmarks)

    print(f"Models: {args.models}")
    print(f"Benchmarks: {benchmark_keys}")
    print(f"Max samples override: {args.max_samples}")
    print(f"Port: {args.port}")
    print(f"Resume: {args.resume}")
    print(f"Seed: {args.seed}")
    print()

    # Set seed for reproducible dataset sampling (same samples across all models)
    set_seed(args.seed)
    print(f"[Main] Seed={args.seed} set for reproducible sampling.\n")

    all_results: list[BenchmarkResult] = []

    for i, model_key in enumerate(args.models):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}/{len(args.models)}: {MODELS[model_key].name}")
        print(f"{'='*60}")

        try:
            results = run_model_benchmarks(
                model_key=model_key,
                benchmark_keys=benchmark_keys,
                port=args.port,
                max_samples=args.max_samples,
                resume=args.resume,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n[Main] Model {model_key} failed entirely: {type(e).__name__}: {e}")
            print("[Main] Continuing to next model...")

        # Wait between models to free GPU memory
        if i < len(args.models) - 1:
            print("\n[Main] Waiting 5s before next model...")
            time.sleep(5)

    generate_comparison_table(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
