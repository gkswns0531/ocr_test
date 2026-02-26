#!/usr/bin/env python3
"""Recompute scores for samples that had evaluation errors.

Fixes:
1. Upstage DP-Bench: 42 samples with metrics.table_metric import error
   - Predictions exist but eval failed due to import collision (now fixed)
   - Ground truth loaded from reference.json via huggingface_hub
2. DeepSeek Nanonets-KIE: 1 sample with json_repair RecursionError
   - Added try/except in _parse_json_fields, recompute here
   - Ground truth loaded from HF dataset
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import compute_nid, compute_teds, compute_kie_anls


def _parse_json_fields(text: str) -> dict:
    """Extract JSON dict from model output (with RecursionError protection)."""
    import json_repair
    text = text.strip()
    try:
        parsed = json_repair.repair_json(text, ensure_ascii=False, return_objects=True)
    except RecursionError:
        return {}
    if isinstance(parsed, list):
        merged = {}
        for item in parsed:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key not in merged:
                        merged[key] = value
                    else:
                        if isinstance(merged[key], list):
                            merged[key].append(value)
                        else:
                            merged[key] = [merged[key], value]
        return merged
    if isinstance(parsed, dict):
        return parsed
    return {}


def eval_dp_bench(pred: str, gt) -> dict:
    """Upstage DP-Bench: NID for text, TEDS for tables."""
    if not isinstance(gt, dict):
        return {"nid": compute_nid(pred, str(gt))}

    elements = gt.get("elements", [])
    if not elements:
        raw = gt.get("raw", {})
        gt_text = raw.get("text", raw.get("markdown", str(gt)))
        return {"nid": compute_nid(pred, str(gt_text))}

    gt_texts = []
    gt_tables_html = []
    for elem in elements:
        if isinstance(elem, dict):
            content = elem.get("content", {})
            if isinstance(content, dict):
                cat = elem.get("category", "")
                if cat == "Table":
                    html = content.get("html", "")
                    if html:
                        gt_tables_html.append(html)
                text = content.get("text", content.get("markdown", ""))
                if text:
                    gt_texts.append(str(text))
            elif isinstance(content, str):
                gt_texts.append(content)

    gt_full_text = "\n".join(gt_texts)
    scores = {"nid": compute_nid(pred, gt_full_text)}

    if gt_tables_html:
        teds_scores = []
        teds_s_scores = []
        for gt_html in gt_tables_html:
            teds_scores.append(compute_teds(pred, gt_html))
            teds_s_scores.append(compute_teds(pred, gt_html, structure_only=True))
        scores["teds"] = sum(teds_scores) / len(teds_scores)
        scores["teds_structure"] = sum(teds_s_scores) / len(teds_s_scores)

    return scores


def eval_kie(pred: str, gt) -> dict:
    """Nanonets-KIE: ANLS."""
    pred_fields = _parse_json_fields(pred)
    gt_fields = gt if isinstance(gt, dict) else {}
    return {"anls": compute_kie_anls(pred_fields, gt_fields)}


def recompute_aggregate(per_sample: list[dict]) -> dict:
    """Recompute aggregate metrics from per-sample results."""
    all_scores = [r["scores"] for r in per_sample if r["scores"]]
    if not all_scores:
        return {}

    keys = set()
    for s in all_scores:
        keys.update(s.keys())

    aggregate = {}
    for key in sorted(keys):
        vals = [s[key] for s in all_scores if key in s and isinstance(s[key], (int, float))]
        if vals:
            aggregate[f"mean_{key}"] = sum(vals) / len(vals)

    error_count = sum(1 for r in per_sample if r.get("error"))
    aggregate["error_rate"] = error_count / len(per_sample)

    latencies = sorted(
        r["latency_ms"] for r in per_sample if isinstance(r.get("latency_ms"), (int, float))
    )
    if latencies:
        n = len(latencies)
        aggregate["latency_mean_ms"] = sum(latencies) / n
        aggregate["latency_median_ms"] = latencies[n // 2]
        aggregate["latency_p90_ms"] = latencies[int(n * 0.9)]
        aggregate["latency_p99_ms"] = latencies[int(n * 0.99)]
        aggregate["latency_min_ms"] = latencies[0]
        aggregate["latency_max_ms"] = latencies[-1]
        aggregate["throughput_samples_per_sec"] = n / (sum(latencies) / 1000) if sum(latencies) > 0 else 0.0

    return aggregate


def load_dpbench_reference() -> dict:
    """Load DP-Bench reference.json with ground truth annotations."""
    from huggingface_hub import hf_hub_download
    ref_path = hf_hub_download(
        "upstage/dp-bench", "dataset/reference.json", repo_type="dataset"
    )
    with open(ref_path) as f:
        reference = json.load(f)
    # Build mapping: sample_id -> ground_truth
    gt_map = {}
    for pdf_name, gt in reference.items():
        sample_id = f"dpbench_{pdf_name}"
        gt_map[sample_id] = gt
    return gt_map


def load_nanonets_gt() -> dict:
    """Load Nanonets-KIE ground truth from HF dataset."""
    from datasets import load_dataset
    ds = load_dataset("nanonets/key_information_extraction", split="test",
                      cache_dir=str(os.path.join(os.path.dirname(__file__), "data_cache")))
    gt_map = {}
    for i, row in enumerate(ds):
        sample_id = f"nanonets_kie_{i}"
        gt_fields = row.get("annotations", {})
        if not isinstance(gt_fields, dict):
            gt_fields = {}
        gt_map[sample_id] = gt_fields
    return gt_map


def fix_dpbench(filepath: str, gt_map: dict) -> int:
    """Fix DP-Bench result file by recomputing error samples."""
    with open(filepath) as f:
        data = json.load(f)

    psr = data.get("per_sample_results", [])
    fixed = 0
    for sample in psr:
        if not sample.get("error"):
            continue

        sid = sample["sample_id"]
        pred = sample.get("prediction", "")
        gt = gt_map.get(sid)
        if gt is None:
            print(f"  WARNING: No GT found for {sid}")
            continue

        try:
            scores = eval_dp_bench(pred, gt)
            sample["scores"] = scores
            sample["error"] = None
            fixed += 1
            score_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"  Fixed {sid}: {score_str}")
        except Exception as e:
            print(f"  Still failing {sid}: {e}")

    if fixed > 0:
        data["metrics"] = recompute_aggregate(psr)
        data["per_sample_results"] = psr
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        nid = data["metrics"].get("mean_nid", 0)
        err = data["metrics"].get("error_rate", 0)
        print(f"  -> Saved ({fixed} fixed): error_rate={err:.4f}, mean_nid={nid:.4f}")

    return fixed


def fix_nanonets(filepath: str, gt_map: dict) -> int:
    """Fix Nanonets-KIE result file by recomputing error samples."""
    with open(filepath) as f:
        data = json.load(f)

    psr = data.get("per_sample_results", [])
    fixed = 0
    for sample in psr:
        if not sample.get("error"):
            continue

        sid = sample["sample_id"]
        pred = sample.get("prediction", "")
        gt = gt_map.get(sid, {})

        try:
            scores = eval_kie(pred, gt)
            sample["scores"] = scores
            sample["error"] = None
            fixed += 1
            print(f"  Fixed {sid}: {scores}")
        except Exception as e:
            print(f"  Still failing {sid}: {e}")

    if fixed > 0:
        data["metrics"] = recompute_aggregate(psr)
        data["per_sample_results"] = psr
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  -> Saved ({fixed} fixed)")

    return fixed


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    # Fix 1: Upstage DP-Bench (all 3 models)
    print("=" * 60)
    print("Fix 1: Upstage DP-Bench - recompute table TEDS for error samples")
    print("=" * 60)
    print("Loading DP-Bench reference.json...")
    gt_map = load_dpbench_reference()
    print(f"Loaded {len(gt_map)} GT entries")

    total_dpbench = 0
    for model in ["glm_ocr", "paddleocr_vl", "deepseek_ocr2"]:
        fpath = os.path.join(results_dir, f"{model}_upstage_dp_bench.json")
        if os.path.exists(fpath):
            print(f"\n[{model}]")
            total_dpbench += fix_dpbench(fpath, gt_map)
    print(f"\nDP-Bench total fixed: {total_dpbench}")

    # Fix 2: DeepSeek Nanonets-KIE (1 error sample)
    print("\n" + "=" * 60)
    print("Fix 2: DeepSeek-OCR2 Nanonets-KIE - recompute RecursionError sample")
    print("=" * 60)
    fpath = os.path.join(results_dir, "deepseek_ocr2_nanonets_kie.json")
    if os.path.exists(fpath):
        print("Loading Nanonets-KIE ground truth...")
        nanonets_gt = load_nanonets_gt()
        print(f"Loaded {len(nanonets_gt)} GT entries")
        fix_nanonets(fpath, nanonets_gt)

    print("\nDone!")


if __name__ == "__main__":
    main()
