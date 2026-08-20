"""Call Azure Layout API directly for remaining 856 OmniDocBench samples, then evaluate all 1355."""

from __future__ import annotations

import json
import time
import base64
import sys
from pathlib import Path
from dataclasses import dataclass

import requests

# ── Config ──────────────────────────────────────────────────────────
ENDPOINT = "https://allganize-document-inteligence-for-dev.cognitiveservices.azure.com"
API_KEY = "040069b99ceb415a87343be1ef58cdbc"
API_VERSION = "2024-11-30"
MODEL_ID = "prebuilt-layout"

PREPARED_DIR = Path("/Users/hanjuncho/Downloads/prepared_datasets")
IMG_DIR = PREPARED_DIR / "omnidocbench" / "images"
RESULTS_DIR = Path("/Users/hanjuncho/code_base/CCW/ocr_test/results_local")
EXISTING_RESULTS = RESULTS_DIR / "azure_layout_omnidocbench_499.json"
CHECKPOINT_PATH = RESULTS_DIR / "azure_remaining_checkpoint.json"
FINAL_OUTPUT = RESULTS_DIR / "azure_layout_omnidocbench_full.json"


@dataclass
class AzureResult:
    idx: int
    file_name: str
    markdown: str
    latency_ms: float


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

    # Submit
    resp = requests.post(url, params=params, headers=headers, data=img_bytes, timeout=60)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "30"))
        print(f"  429 rate limited, retry after {retry_after}s")
        time.sleep(min(retry_after, 60))
        resp = requests.post(url, params=params, headers=headers, data=img_bytes, timeout=60)

    resp.raise_for_status()
    operation_url = resp.headers["Operation-Location"]

    # Poll
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
        # else "running" → keep polling


def _get_remaining_indices() -> list[int]:
    """Get indices not yet in existing results."""
    done_idxs: set[int] = set()

    # From existing 499 results
    if EXISTING_RESULTS.exists():
        with open(EXISTING_RESULTS) as f:
            data = json.load(f)
        done_idxs.update(s["idx"] for s in data["per_sample"])

    # From checkpoint (in-progress)
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            cp = json.load(f)
        done_idxs.update(s["idx"] for s in cp)

    # All indices
    meta_path = PREPARED_DIR / "omnidocbench" / "metadata.jsonl"
    all_idxs: set[int] = set()
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            all_idxs.add(rec["idx"])

    return sorted(all_idxs - done_idxs)


def step1_call_azure_api() -> list[dict]:
    """Call Azure API for remaining samples, checkpointing along the way."""
    remaining = _get_remaining_indices()
    print(f"Remaining: {len(remaining)} samples")

    # Load checkpoint
    checkpoint: list[dict] = []
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {len(checkpoint)} already done")

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

        # Save checkpoint every 10 samples
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


def step2_evaluate_all() -> dict:
    """Evaluate all samples (existing 499 + new) against OmniDocBench."""
    from benchmarks import _eval_document_parse

    # Load GT
    meta_path = PREPARED_DIR / "omnidocbench" / "metadata.jsonl"
    gt_by_idx: dict[int, dict] = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            gt_by_idx[rec["idx"]] = rec

    # Merge all results: existing + new checkpoint
    all_results: dict[int, dict] = {}  # idx → {markdown, latency_ms, ...}

    # Existing 499
    if EXISTING_RESULTS.exists():
        with open(EXISTING_RESULTS) as f:
            existing = json.load(f)
        # Need to get markdown from cache or re-extract
        # Existing results don't have markdown stored — we'll re-evaluate from cache
        # Actually, existing results have scores already. Just carry them forward.
        for s in existing["per_sample"]:
            all_results[s["idx"]] = {"from_existing": True, **s}

    # New results from checkpoint
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        for s in checkpoint:
            if s["error"]:
                continue
            all_results[s["idx"]] = {
                "from_existing": False,
                "idx": s["idx"],
                "file_name": s["file_name"],
                "markdown": s["markdown"],
                "latency_ms": s["latency_ms"],
            }

    print(f"Total results to evaluate: {len(all_results)}")

    per_sample = []
    errors = 0
    need_eval = [(idx, r) for idx, r in sorted(all_results.items()) if not r.get("from_existing")]
    carried = [r for r in all_results.values() if r.get("from_existing")]

    # Carry forward existing evaluated results
    for r in carried:
        per_sample.append({
            "sample_id": r["sample_id"],
            "file_name": r["file_name"],
            "idx": r["idx"],
            "scores": r["scores"],
            "latency_ms": r.get("latency_ms", 0.0),
            "error": r.get("error"),
        })

    # Evaluate new results
    for i, (idx, r) in enumerate(need_eval):
        gt = gt_by_idx.get(idx)
        if not gt:
            continue
        try:
            scores = _eval_document_parse(r["markdown"], gt["ground_truth"], gt.get("metadata", {}))
            per_sample.append({
                "sample_id": gt["sample_id"],
                "file_name": r["file_name"],
                "idx": idx,
                "scores": scores,
                "latency_ms": r.get("latency_ms", 0.0),
                "error": None,
            })
        except Exception as e:
            errors += 1
            per_sample.append({
                "sample_id": gt["sample_id"],
                "file_name": r.get("file_name", f"{idx:05d}.jpg"),
                "idx": idx,
                "scores": {},
                "latency_ms": r.get("latency_ms", 0.0),
                "error": str(e),
            })
        if (i + 1) % 100 == 0:
            print(f"  Evaluated [{i+1}/{len(need_eval)}]")

    per_sample.sort(key=lambda s: s["idx"])

    # Aggregate
    valid = [r for r in per_sample if r["scores"]]
    aggregate = {}
    if valid:
        for key in valid[0]["scores"]:
            aggregate[key] = sum(r["scores"][key] for r in valid) / len(valid)

    return {"aggregate": aggregate, "per_sample": per_sample, "total": len(per_sample), "errors": errors}


if __name__ == "__main__":
    _patch()

    print("=" * 60)
    print("Step 1: Call Azure Layout API for remaining samples")
    print("=" * 60)
    new_results = step1_call_azure_api()
    api_errors = sum(1 for r in new_results if r["error"])
    print(f"API calls done: {len(new_results)} total, {api_errors} errors")

    print("\n" + "=" * 60)
    print("Step 2: Evaluate all samples")
    print("=" * 60)
    t0 = time.time()
    results = step2_evaluate_all()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({results['total']} samples, {results['errors']} errors, {elapsed:.0f}s)")
    print(f"{'=' * 60}")
    for k, v in results["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    # Save full results
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {FINAL_OUTPUT}")
