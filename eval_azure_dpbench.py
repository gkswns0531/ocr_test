"""Call Azure Layout API for 200 DP-Bench samples, then evaluate."""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────────────
ENDPOINT = "https://allganize-document-inteligence-for-dev.cognitiveservices.azure.com"
API_KEY = "040069b99ceb415a87343be1ef58cdbc"
API_VERSION = "2024-11-30"
MODEL_ID = "prebuilt-layout"

PREPARED_DIR = Path("/Users/hanjuncho/Downloads/prepared_datasets")
IMG_DIR = PREPARED_DIR / "upstage_dp_bench" / "images"
RESULTS_DIR = Path("/Users/hanjuncho/code_base/CCW/ocr_test/results_local")
CHECKPOINT_PATH = RESULTS_DIR / "azure_dpbench_checkpoint.json"
FINAL_OUTPUT = RESULTS_DIR / "azure_layout_dpbench.json"


def _call_azure_layout(image_path: Path) -> tuple[str, float]:
    """Call Azure Document Intelligence Layout API and return (markdown, latency_ms)."""
    url = f"{ENDPOINT}/documentintelligence/documentModels/{MODEL_ID}:analyze"
    params = {"api-version": API_VERSION, "outputContentFormat": "markdown"}
    headers = {
        "Ocp-Apim-Subscription-Key": API_KEY,
        "Content-Type": "application/octet-stream",
    }

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    t0 = time.time()

    resp = requests.post(url, params=params, headers=headers, data=img_bytes, timeout=60)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "30"))
        print(f"  429 rate limited, retry after {retry_after}s")
        time.sleep(min(retry_after, 60))
        resp = requests.post(url, params=params, headers=headers, data=img_bytes, timeout=60)

    resp.raise_for_status()
    operation_url = resp.headers["Operation-Location"]

    poll_headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    while True:
        time.sleep(1)
        poll_resp = requests.get(operation_url, headers=poll_headers, timeout=30)
        poll_resp.raise_for_status()
        result = poll_resp.json()
        status = result.get("status", "")
        if status == "succeeded":
            latency_ms = (time.time() - t0) * 1000
            content = result.get("analyzeResult", {}).get("content", "")
            return content, latency_ms
        elif status == "failed":
            raise RuntimeError(f"Azure analysis failed: {result}")


def step1_call_azure_api() -> list[dict]:
    """Call Azure API for DP-Bench samples with checkpointing."""
    # Load checkpoint
    checkpoint: list[dict] = []
    done_idxs: set[int] = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        done_idxs = {s["idx"] for s in checkpoint}
        print(f"Loaded checkpoint: {len(checkpoint)} already done")

    # All indices
    meta_path = PREPARED_DIR / "upstage_dp_bench" / "metadata.jsonl"
    all_idxs: list[int] = []
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            all_idxs.append(rec["idx"])

    remaining = [i for i in all_idxs if i not in done_idxs]
    print(f"Remaining: {len(remaining)} samples")

    for i, idx in enumerate(remaining):
        img_path = IMG_DIR / f"{idx:05d}.jpg"
        file_name = f"{idx:05d}.jpg"

        try:
            markdown, latency_ms = _call_azure_layout(img_path)
            checkpoint.append({
                "idx": idx,
                "file_name": file_name,
                "markdown": markdown,
                "latency_ms": latency_ms,
                "error": None,
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
                print(f"  [{i+1}/{len(remaining)}] {file_name} done ({latency_ms:.0f}ms)")
        except Exception as e:
            checkpoint.append({
                "idx": idx,
                "file_name": file_name,
                "markdown": "",
                "latency_ms": 0.0,
                "error": str(e),
            })
            print(f"  [{i+1}/{len(remaining)}] {file_name} ERROR: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(checkpoint, f, ensure_ascii=False)

    return checkpoint


def _patch():
    """Patch config and benchmarks for local execution."""
    local_base = Path("/Users/hanjuncho/code_base/CCW/ocr_test")
    results_dir = local_base / "results_local"
    data_cache_dir = local_base / "data_cache_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_cache_dir.mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("config", local_base / "config.py")
    config_mod = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_mod

    _orig_mkdir = Path.mkdir

    def _safe_mkdir(self, *args, **kwargs):
        if str(self).startswith("/root"):
            return
        return _orig_mkdir(self, *args, **kwargs)

    Path.mkdir = _safe_mkdir
    try:
        spec.loader.exec_module(config_mod)
    finally:
        Path.mkdir = _orig_mkdir

    config_mod.RESULTS_DIR = results_dir
    config_mod.DATA_CACHE_DIR = data_cache_dir
    config_mod.PREPARED_DIR = PREPARED_DIR

    import benchmarks as _b
    _b._OMNIDOCBENCH_ROOT = str(local_base / "OmniDocBench")
    import metrics as _m
    _m._OMNIDOCBENCH_METRICS = str(local_base / "OmniDocBench" / "metrics")


def step2_evaluate(api_results: list[dict]) -> dict:
    """Evaluate all DP-Bench samples."""
    from benchmarks import _eval_dp_bench

    meta_path = PREPARED_DIR / "upstage_dp_bench" / "metadata.jsonl"
    gt_by_idx: dict[int, dict] = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            gt_by_idx[rec["idx"]] = rec

    per_sample = []
    errors = 0
    for i, r in enumerate(sorted(api_results, key=lambda x: x["idx"])):
        if r.get("error"):
            errors += 1
            continue
        gt = gt_by_idx.get(r["idx"])
        if not gt:
            continue
        try:
            scores = _eval_dp_bench(r["markdown"], gt["ground_truth"], gt.get("metadata", {}))
            per_sample.append({
                "sample_id": gt["sample_id"],
                "file_name": r["file_name"],
                "idx": r["idx"],
                "scores": scores,
                "latency_ms": r.get("latency_ms", 0.0),
                "error": None,
            })
        except Exception as e:
            errors += 1
            per_sample.append({
                "sample_id": gt["sample_id"],
                "file_name": r.get("file_name", f"{r['idx']:05d}.jpg"),
                "idx": r["idx"],
                "scores": {},
                "latency_ms": r.get("latency_ms", 0.0),
                "error": str(e),
            })
        if (i + 1) % 50 == 0:
            print(f"  Evaluated [{i+1}/{len(api_results)}]")

    valid = [r for r in per_sample if r["scores"]]
    aggregate: dict[str, float] = {}
    if valid:
        all_keys = set()
        for r in valid:
            all_keys.update(r["scores"].keys())
        for key in sorted(all_keys):
            vals = [r["scores"][key] for r in valid if key in r["scores"]]
            if vals:
                aggregate[key] = sum(vals) / len(vals)

    return {"aggregate": aggregate, "per_sample": per_sample, "total": len(per_sample), "errors": errors}


if __name__ == "__main__":
    _patch()

    print("=" * 60)
    print("Step 1: Call Azure Layout API for DP-Bench (200 samples)")
    print("=" * 60)
    api_results = step1_call_azure_api()
    api_errors = sum(1 for r in api_results if r.get("error"))
    print(f"API calls done: {len(api_results)} total, {api_errors} errors")

    print("\n" + "=" * 60)
    print("Step 2: Evaluate")
    print("=" * 60)
    t0 = time.time()
    results = step2_evaluate(api_results)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({results['total']} samples, {results['errors']} errors, {elapsed:.0f}s)")
    print(f"{'=' * 60}")
    for k, v in results["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {FINAL_OUTPUT}")
