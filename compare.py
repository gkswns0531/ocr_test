#!/usr/bin/env python3
"""Load all result JSONs and generate a comparison table."""

from __future__ import annotations

import json
import time

from config import RESULTS_DIR


def load_results() -> list[dict]:
    """Load all benchmark result JSONs from the results directory."""
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        # Skip checkpoints and comparison files
        if "checkpoint" in p.name or "comparison" in p.name:
            continue
        with open(p) as f:
            data = json.load(f)
        results.append(data)
    return results


def build_comparison(results: list[dict]) -> dict:
    """Build model × benchmark comparison matrix."""
    by_model: dict[str, dict[str, dict]] = {}
    for r in results:
        model = r.get("model_name", "unknown")
        bench = r.get("benchmark_name", "unknown")
        metrics = r.get("metrics", {})
        if model not in by_model:
            by_model[model] = {}
        by_model[model][bench] = metrics
    return by_model


def print_tables(by_model: dict[str, dict[str, dict]]) -> None:
    """Print markdown comparison tables."""
    # Collect all benchmark names
    bench_names = []
    for model_benches in by_model.values():
        for bn in model_benches:
            if bn not in bench_names:
                bench_names.append(bn)

    models = list(by_model.keys())

    print("=" * 80)
    print("OCR BENCHMARK COMPARISON")
    print("=" * 80)

    for bench_name in bench_names:
        print(f"\n### {bench_name}")

        # Gather all metric keys
        metric_keys: list[str] = []
        for model in models:
            if bench_name in by_model[model]:
                for k in by_model[model][bench_name]:
                    if k not in metric_keys:
                        metric_keys.append(k)

        if not metric_keys:
            print("(no metrics)")
            continue

        # Header
        header = "| Model | " + " | ".join(metric_keys) + " |"
        sep = "|" + "-------|" * (1 + len(metric_keys))
        print(header)
        print(sep)

        # Rows
        for model in models:
            metrics = by_model[model].get(bench_name, {})
            vals = []
            for k in metric_keys:
                v = metrics.get(k)
                if v is None:
                    vals.append("-")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            print(f"| {model} | " + " | ".join(vals) + " |")

    # Summary: one row per model, one column per benchmark with key metric
    print(f"\n### Summary (key metric per benchmark)")
    key_metrics = {
        "OmniDocBench": "mean_bleu",
        "Upstage DP-Bench": "mean_nid",
        "OCRBench": "mean_accuracy",
        "UniMERNet": "mean_exact_match",
        "PubTabNet": "mean_teds",
        "TEDS_TEST": "mean_teds",
        "Nanonets-KIE": "mean_f1",
        "Handwritten-Forms": "mean_char_accuracy",
    }

    available_benches = [b for b in bench_names if b in key_metrics]
    if available_benches:
        header = "| Model | " + " | ".join(available_benches) + " |"
        sep = "|" + "-------|" * (1 + len(available_benches))
        print(header)
        print(sep)

        for model in models:
            vals = []
            for bench in available_benches:
                m = by_model[model].get(bench, {})
                key = key_metrics[bench]
                v = m.get(key)
                if v is None:
                    vals.append("-")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            print(f"| {model} | " + " | ".join(vals) + " |")


def main() -> None:
    results = load_results()
    if not results:
        print(f"No result files found in {RESULTS_DIR}")
        return

    print(f"Loaded {len(results)} result files")
    by_model = build_comparison(results)
    print_tables(by_model)

    # Save comparison JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"comparison_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(by_model, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
