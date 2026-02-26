#!/usr/bin/env python3
"""Evaluation-only script: reads saved predictions and computes metrics.

Reads predictions from predictions/{model_key}/{benchmark_key}/ and evaluates
using the official protocol for each benchmark.

OmniDocBench: calls official pdf_validation.py (End2EndDataset + metrics)
Others:       uses existing per-benchmark evaluators from benchmarks.py

Note: named eval_bench.py (not evaluate.py) to avoid shadowing HuggingFace 'evaluate' package.

Usage:
    python3 eval_bench.py --model deepseek-ocr2 --benchmarks omnidocbench
    python3 eval_bench.py --model glm-ocr-pipeline --benchmarks all
    python3 eval_bench.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from config import BENCHMARKS, MODELS, PREPARED_DIR, RESULTS_DIR

PREDICTIONS_DIR = Path("/home/ubuntu/ocr_test/predictions")
OMNIDOCBENCH_ROOT = Path("/home/ubuntu/OmniDocBench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR Evaluation (from saved predictions)")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["all"],
        help="Benchmarks to evaluate (or 'all')",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all models that have prediction directories",
    )
    return parser.parse_args()


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def resolve_benchmarks(benchmark_args: list[str]) -> list[str]:
    if "all" in benchmark_args:
        return list(BENCHMARKS.keys())
    for b in benchmark_args:
        if b not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {b}. Available: {list(BENCHMARKS.keys())}")
    return benchmark_args


# ─── OmniDocBench evaluation (official protocol) ─────────────────────

def eval_omnidocbench(model_key: str) -> dict | None:
    """Evaluate OmniDocBench using official pdf_validation.py protocol.

    1. Downloads OmniDocBench.json (GT annotations) via huggingface_hub
    2. Writes a temporary YAML config pointing to our prediction dir
    3. Runs pdf_validation.py as subprocess from OmniDocBench directory
    4. Reads results from OmniDocBench/result/ directory
    5. Saves formatted results to our results/ directory
    """
    import yaml
    from huggingface_hub import hf_hub_download

    model_dir = model_key.replace("-", "_")
    pred_dir = PREDICTIONS_DIR / model_dir / "omnidocbench"

    if not pred_dir.exists():
        print(f"[Evaluate] No predictions directory: {pred_dir}")
        return None

    md_files = list(pred_dir.glob("*.md"))
    if not md_files:
        print(f"[Evaluate] No .md prediction files in {pred_dir}")
        return None

    print(f"[Evaluate] Found {len(md_files)} prediction .md files for OmniDocBench")

    # Get GT JSON path (cached by huggingface_hub)
    gt_json_path = hf_hub_download(
        "opendatalab/OmniDocBench", "OmniDocBench.json", repo_type="dataset"
    )

    # Create config YAML for pdf_validation.py
    config = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist", "CDM"]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": gt_json_path},
                "prediction": {"data_path": str(pred_dir)},
                "match_method": "quick_match",
            },
        }
    }

    config_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            config_path = f.name

        print(f"[Evaluate] Running OmniDocBench official evaluation (pdf_validation.py)...")
        print(f"[Evaluate] GT: {gt_json_path}")
        print(f"[Evaluate] Predictions: {pred_dir}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "pdf_validation.py", "-c", config_path],
            cwd=str(OMNIDOCBENCH_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for full dataset
        )
        elapsed = time.time() - t0

        # Print stdout (contains metric tables)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"[Evaluate] OmniDocBench evaluation failed (exit code {result.returncode})")
            if result.stderr:
                print(f"[Evaluate] stderr:\n{result.stderr[-3000:]}")
            return None

        print(f"[Evaluate] OmniDocBench evaluation completed in {elapsed:.1f}s")

    finally:
        if config_path and os.path.exists(config_path):
            os.unlink(config_path)

    # Read results from OmniDocBench's result/ directory
    # save_name = basename(prediction_data_path) + "_" + match_method
    save_name = f"omnidocbench_quick_match"
    metric_path = OMNIDOCBENCH_ROOT / "result" / f"{save_name}_metric_result.json"

    if not metric_path.exists():
        print(f"[Evaluate] Results file not found: {metric_path}")
        return None

    with open(metric_path) as f:
        omnidoc_results = json.load(f)

    # Extract key metrics into flat dict
    metrics = _extract_omnidoc_metrics(omnidoc_results)

    # Save our formatted results
    result_data = {
        "benchmark_name": "OmniDocBench",
        "model_name": MODELS[model_key].name,
        "metrics": metrics,
        "omnidocbench_raw": omnidoc_results,
        "total_predictions": len(md_files),
        "elapsed_seconds": elapsed,
    }

    result_path = RESULTS_DIR / f"{_safe_name(MODELS[model_key].name)}_omnidocbench_eval.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"[Evaluate] OmniDocBench results saved to {result_path}")
    _print_omnidoc_summary(metrics)
    return result_data


def _extract_omnidoc_metrics(omnidoc_results: dict) -> dict:
    """Extract key metrics from OmniDocBench result JSON into flat dict."""
    metrics: dict[str, float] = {}

    for element_type, element_data in omnidoc_results.items():
        all_data = element_data.get("all", {})
        for metric_name, metric_vals in all_data.items():
            if isinstance(metric_vals, dict):
                for sub_key, value in metric_vals.items():
                    if isinstance(value, (int, float)):
                        key = f"{element_type}_{metric_name}_{sub_key}"
                        metrics[key] = value

    return metrics


def _print_omnidoc_summary(metrics: dict) -> None:
    """Print a readable summary of OmniDocBench metrics."""
    print("\n--- OmniDocBench Summary ---")

    # Text
    text_ed = metrics.get("text_block_Edit_dist_ALL_page_avg")
    if text_ed is not None:
        print(f"  Text Block Edit Dist (page avg):  {text_ed:.4f}")
        print(f"  Text Block Score:                 {(1 - text_ed) * 100:.1f}")

    # Formula
    formula_ed = metrics.get("display_formula_Edit_dist_ALL_page_avg")
    formula_cdm = metrics.get("display_formula_CDM_all")
    if formula_ed is not None:
        print(f"  Formula Edit Dist (page avg):     {formula_ed:.4f}")
        print(f"  Formula ED Score:                 {(1 - formula_ed) * 100:.1f}")
    if formula_cdm is not None:
        print(f"  Formula CDM:                      {formula_cdm:.4f}")
        print(f"  Formula CDM Score:                {formula_cdm * 100:.1f}")

    # Table
    table_teds = metrics.get("table_TEDS_all")
    if table_teds is not None:
        print(f"  Table TEDS:                       {table_teds:.4f}")
    table_ed = metrics.get("table_Edit_dist_ALL_page_avg")
    if table_ed is not None:
        print(f"  Table Edit Dist (page avg):       {table_ed:.4f}")

    # Reading Order
    order_ed = metrics.get("reading_order_Edit_dist_ALL_page_avg")
    if order_ed is not None:
        print(f"  Reading Order Edit Dist:          {order_ed:.4f}")

    # Compute overall score: official = ((1-text_ED)*100 + table_TEDS*100 + formula_CDM*100) / 3
    components = []
    if text_ed is not None:
        components.append((1 - text_ed) * 100)
    if table_teds is not None:
        components.append(table_teds * 100)
    # Prefer CDM for formula (official metric), fall back to ED
    if formula_cdm is not None:
        components.append(formula_cdm * 100)
    elif formula_ed is not None:
        components.append((1 - formula_ed) * 100)
    if components:
        overall = sum(components) / len(components)
        print(f"  Overall Score:                    {overall:.1f}")
    print()


# ─── Other benchmark evaluation ──────────────────────────────────────

def eval_benchmark(model_key: str, benchmark_key: str) -> dict | None:
    """Evaluate a non-OmniDocBench benchmark by reading predictions from disk.

    1. Loads ground truth from prepared_datasets/{benchmark_key}/metadata.jsonl
    2. Reads prediction files from predictions/{model_key}/{benchmark_key}/
    3. Runs the appropriate evaluator for each sample
    4. Aggregates metrics and saves results
    """
    from benchmarks import (
        _eval_dp_bench,
        _eval_ocrbench,
        _eval_formula,
        _eval_table,
        _eval_kie,
        _eval_handwritten,
        _aggregate_metrics,
    )

    evaluators = {
        "document_parse_dp": _eval_dp_bench,
        "text_recognition": _eval_ocrbench,
        "formula_recognition": _eval_formula,
        "table_recognition": _eval_table,
        "kie_extraction": _eval_kie,
        "handwritten": _eval_handwritten,
    }

    bench_cfg = BENCHMARKS[benchmark_key]
    model_dir = model_key.replace("-", "_")
    pred_dir = PREDICTIONS_DIR / model_dir / benchmark_key

    if not pred_dir.exists():
        print(f"[Evaluate] No predictions directory: {pred_dir}")
        return None

    # Load ground truth from prepared datasets
    prepared_dir = PREPARED_DIR / benchmark_key
    meta_path = prepared_dir / "metadata.jsonl"

    if not meta_path.exists():
        print(f"[Evaluate] No prepared dataset for {benchmark_key}")
        return None

    # Build GT map: sample_id → {ground_truth, metadata}
    gt_map: dict[str, dict] = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            gt_map[rec["sample_id"]] = {
                "ground_truth": rec["ground_truth"],
                "metadata": rec.get("metadata", {}),
            }

    evaluator = evaluators.get(bench_cfg.metric_type)
    if evaluator is None:
        print(f"[Evaluate] No evaluator for metric type: {bench_cfg.metric_type}")
        return None

    # Find prediction files
    pred_files = sorted(pred_dir.glob("*.txt"))
    if not pred_files:
        print(f"[Evaluate] No prediction files in {pred_dir}")
        return None

    print(f"[Evaluate] Evaluating {bench_cfg.name}: {len(pred_files)} predictions")

    # Evaluate each prediction
    per_sample_results = []
    from tqdm import tqdm
    for pred_path in tqdm(pred_files, desc=f"Eval {bench_cfg.name}", unit="sample"):
        sample_id = pred_path.stem
        if sample_id not in gt_map:
            continue

        raw_text = pred_path.read_text(encoding="utf-8")
        # OCRBench predictions are stored as JSON {question, answer}
        if benchmark_key == "ocrbench" and pred_path.suffix == ".json":
            try:
                pred_data = json.loads(raw_text)
                prediction = pred_data.get("answer", "")
            except json.JSONDecodeError:
                prediction = raw_text
        else:
            prediction = raw_text
        sample_data = gt_map[sample_id]

        try:
            scores = evaluator(prediction, sample_data["ground_truth"], sample_data["metadata"])
            per_sample_results.append({
                "sample_id": sample_id,
                "prediction": prediction[:2000],  # truncate for storage
                "scores": scores,
                "error": None,
            })
        except Exception as e:
            per_sample_results.append({
                "sample_id": sample_id,
                "prediction": prediction[:2000],
                "scores": {},
                "error": f"{type(e).__name__}: {e}",
            })

    if not per_sample_results:
        print(f"[Evaluate] No valid predictions matched GT for {bench_cfg.name}")
        return None

    # Aggregate metrics
    aggregate = _aggregate_metrics(per_sample_results, bench_cfg.metric_type)

    # UniMERNet: compute corpus-level BLEU (official protocol)
    if benchmark_key == "unimernet":
        all_preds = []
        all_refs = []
        for r in per_sample_results:
            if r.get("error") or not r.get("scores"):
                continue
            np = r["scores"].get("norm_pred", "")
            ng = r["scores"].get("norm_gt", "")
            if np.strip() and ng.strip():
                all_preds.append(np)
                all_refs.append(ng)
        if all_preds:
            import evaluate as hf_evaluate
            import random
            bleu = hf_evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, int(1e8)))
            try:
                bleu_result = bleu.compute(predictions=all_preds, references=all_refs)
                aggregate["corpus_bleu"] = bleu_result["bleu"]
            except (ZeroDivisionError, ValueError):
                aggregate["corpus_bleu"] = 0.0
            print(f"[Evaluate] UniMERNet corpus-level BLEU: {aggregate.get('corpus_bleu', 0):.4f}")

    result_data = {
        "benchmark_name": bench_cfg.name,
        "model_name": MODELS[model_key].name,
        "metrics": aggregate,
        "per_sample_results": per_sample_results,
        "total_samples": len(per_sample_results),
    }

    result_path = RESULTS_DIR / f"{_safe_name(MODELS[model_key].name)}_{benchmark_key}_eval.json"
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"[Evaluate] {bench_cfg.name}: {aggregate}")
    print(f"[Evaluate] Results saved to {result_path}")
    return result_data


# ─── Comparison table ─────────────────────────────────────────────────

def generate_comparison(all_results: list[dict]) -> None:
    """Print a comparison table across models and benchmarks."""
    if not all_results:
        print("No results to compare.")
        return

    by_model: dict[str, dict[str, dict]] = {}
    for r in all_results:
        model = r["model_name"]
        bench = r["benchmark_name"]
        if model not in by_model:
            by_model[model] = {}
        by_model[model][bench] = r.get("metrics", {})

    bench_names = []
    for model_results in by_model.values():
        for bn in model_results:
            if bn not in bench_names:
                bench_names.append(bn)

    print("\n" + "=" * 80)
    print("EVALUATION COMPARISON")
    print("=" * 80)

    for bench_name in bench_names:
        print(f"\n### {bench_name}")
        metric_keys: list[str] = []
        for model_results in by_model.values():
            if bench_name in model_results:
                for k in model_results[bench_name]:
                    if k not in metric_keys and not k.startswith("latency_") and k != "throughput_samples_per_sec":
                        metric_keys.append(k)

        if not metric_keys:
            continue

        header = "| Model | " + " | ".join(metric_keys) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(metric_keys)) + "|"
        print(header)
        print(sep)

        for model_name, model_results in by_model.items():
            m = model_results.get(bench_name, {})
            vals = []
            for k in metric_keys:
                v = m.get(k, "-")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            print(f"| {model_name} | " + " | ".join(vals) + " |")

    # Save comparison JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    comparison = {
        "models": list(by_model.keys()),
        "benchmarks": bench_names,
        "results": {
            model: {bench: metrics for bench, metrics in benches.items()}
            for model, benches in by_model.items()
        },
    }
    out_path = RESULTS_DIR / f"eval_comparison_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Determine which models to evaluate
    if args.all:
        model_keys = []
        for model_key in MODELS:
            model_dir = PREDICTIONS_DIR / model_key.replace("-", "_")
            if model_dir.exists() and any(model_dir.iterdir()):
                model_keys.append(model_key)
        if not model_keys:
            print("No models with predictions found.")
            sys.exit(1)
        print(f"Found predictions for: {model_keys}")
    else:
        if not args.model:
            print("Error: --model is required (or use --all)")
            sys.exit(1)
        model_keys = [args.model]

    benchmark_keys = resolve_benchmarks(args.benchmarks)

    all_results = []

    for model_key in model_keys:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {MODELS[model_key].name} ({model_key})")
        print(f"{'=' * 60}")

        for bench_key in benchmark_keys:
            try:
                if bench_key == "omnidocbench":
                    result = eval_omnidocbench(model_key)
                else:
                    result = eval_benchmark(model_key, bench_key)
                if result:
                    all_results.append(result)
            except Exception as e:
                import traceback
                print(f"[Evaluate] {bench_key} failed: {type(e).__name__}: {e}")
                traceback.print_exc()

    if len(all_results) > 1:
        generate_comparison(all_results)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
