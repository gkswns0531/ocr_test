"""Evaluate 499 completed Azure Layout results against OmniDocBench."""

from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
import time
from pathlib import Path
from dataclasses import dataclass

COMPOSE_DIR = "/Users/hanjuncho/code_base/CCW/mally"
PROJECT_ID = "678f65c06e31e6ec0341ba81"
OMNI_FOLDER_ID = "69ab6d2e401ab959ed091c35"
PREPARED_DIR = Path("/Users/hanjuncho/Downloads/prepared_datasets")


@dataclass
class AzureResult:
    kb_id: str
    file_name: str
    idx: int
    markdown: str
    latency_ms: float = 0.0


def _run(cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=COMPOSE_DIR)
    return r.stdout.strip()


def _patch():
    """Patch config and benchmarks for local execution."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

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


def step1_dump_ocr_zips():
    """Bulk-download all ocr.zip files from MinIO to a local cache dir."""
    cache_dir = Path("/tmp/azure_ocr_cache")
    cache_dir.mkdir(exist_ok=True)

    # Get completed KB list
    out = _run(f"""docker compose exec -T mongodb mongosh mally_dev --quiet --eval '
    var kbs = db.knowledge_base.find({{
        target_folder_id: "{OMNI_FOLDER_ID}",
        process_state: "completed"
    }});
    var results = [];
    kbs.forEach(function(doc) {{
        results.push({{kb_id: doc._id.toString(), file_name: doc.file_name}});
    }});
    print(JSON.stringify(results));
    '""")
    kbs = json.loads(out)
    print(f"Found {len(kbs)} completed KBs")

    # For each KB, find ocr.zip and download
    results = []
    for i, kb in enumerate(kbs):
        kb_id = kb["kb_id"]
        fn = kb["file_name"]
        idx = int(Path(fn).stem)
        local_path = cache_dir / f"{idx:05d}_ocr.zip"

        if local_path.exists() and local_path.stat().st_size > 0:
            results.append({"kb_id": kb_id, "file_name": fn, "idx": idx, "zip_path": str(local_path)})
            continue

        # Find ocr.zip in MinIO
        try:
            ocr_path = _run(
                f"docker compose exec -T minio mc find local/alli-files/{PROJECT_ID}/{kb_id}/ --name 'ocr.zip'",
                timeout=10,
            ).strip().split("\n")[0]
        except Exception:
            print(f"  [{i+1}] {fn}: no ocr.zip found")
            continue

        if not ocr_path:
            continue

        # Copy to minio container /tmp, then to host
        try:
            _run(f"docker compose exec -T minio mc cp {ocr_path} /tmp/_e_{idx}.zip", timeout=10)
            _run(f"docker compose cp minio:/tmp/_e_{idx}.zip {local_path}", timeout=10)
            results.append({"kb_id": kb_id, "file_name": fn, "idx": idx, "zip_path": str(local_path)})
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(kbs)}] downloaded")
        except Exception as e:
            print(f"  [{i+1}] {fn}: download error: {e}")

    # Save manifest
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Downloaded {len(results)} ocr.zip files → {cache_dir}")
    return results


def step2_extract_markdowns(manifest: list[dict]) -> list[AzureResult]:
    """Extract markdown from ocr.zip files."""
    results = []
    for entry in manifest:
        zip_path = Path(entry["zip_path"])
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                ocr_files = sorted(f for f in zf.namelist() if f.startswith("ocr-") and f.endswith(".json"))
                markdowns = []
                for ocr_file in ocr_files:
                    data = json.loads(zf.read(ocr_file))
                    content = data.get("content", "")
                    if content:
                        markdowns.append(content)
                md = "\n\n".join(markdowns)

                # Try to get latency from meta.json
                latency = 0.0
                if "meta.json" in zf.namelist():
                    meta = json.loads(zf.read("meta.json"))
                    latency = meta.get("latency_ms", 0.0)

                results.append(AzureResult(
                    kb_id=entry["kb_id"],
                    file_name=entry["file_name"],
                    idx=entry["idx"],
                    markdown=md,
                    latency_ms=latency,
                ))
        except Exception as e:
            print(f"  {entry['file_name']}: extract error: {e}")

    return sorted(results, key=lambda r: r.idx)


def step3_evaluate(azure_results: list[AzureResult]) -> dict:
    """Evaluate against OmniDocBench."""
    from benchmarks import _eval_document_parse

    # Load GT
    meta_path = PREPARED_DIR / "omnidocbench" / "metadata.jsonl"
    gt_by_idx = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            gt_by_idx[rec["idx"]] = rec

    per_sample = []
    errors = 0
    for i, result in enumerate(azure_results):
        sample = gt_by_idx.get(result.idx)
        if not sample:
            print(f"  {result.file_name}: no GT found for idx={result.idx}")
            continue

        try:
            scores = _eval_document_parse(result.markdown, sample["ground_truth"], sample.get("metadata", {}))
            per_sample.append({
                "sample_id": sample["sample_id"],
                "file_name": result.file_name,
                "idx": result.idx,
                "scores": scores,
                "error": None,
            })
        except Exception as e:
            errors += 1
            per_sample.append({
                "sample_id": sample["sample_id"],
                "file_name": result.file_name,
                "idx": result.idx,
                "scores": {},
                "error": str(e),
            })

        if (i + 1) % 100 == 0:
            valid = [r for r in per_sample if r["scores"]]
            if valid:
                avg_text = sum(r["scores"]["text_score"] for r in valid) / len(valid)
                avg_table = sum(r["scores"]["table_teds"] for r in valid) / len(valid)
                avg_overall = sum(r["scores"]["overall"] for r in valid) / len(valid)
                print(f"  [{i+1}/{len(azure_results)}] text={avg_text:.1f} table={avg_table:.1f} overall={avg_overall:.1f} (errors={errors})")

    # Final aggregate
    valid = [r for r in per_sample if r["scores"]]
    aggregate = {}
    if valid:
        for key in valid[0]["scores"]:
            aggregate[key] = sum(r["scores"][key] for r in valid) / len(valid)

    return {"aggregate": aggregate, "per_sample": per_sample, "total": len(per_sample), "errors": errors}


if __name__ == "__main__":
    _patch()

    print("=" * 60)
    print("Step 1: Download ocr.zip files from MinIO")
    print("=" * 60)
    manifest = step1_dump_ocr_zips()

    print("\n" + "=" * 60)
    print("Step 2: Extract markdown from ocr.zip")
    print("=" * 60)
    azure_results = step2_extract_markdowns(manifest)
    print(f"Extracted {len(azure_results)} results")

    print("\n" + "=" * 60)
    print(f"Step 3: Evaluate {len(azure_results)} samples against OmniDocBench")
    print("=" * 60)
    t0 = time.time()
    results = step3_evaluate(azure_results)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({results['total']} samples, {results['errors']} errors, {elapsed:.0f}s)")
    print(f"{'=' * 60}")
    for k, v in results["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    # Save results
    out_path = Path("/Users/hanjuncho/code_base/CCW/ocr_test/results_local/azure_layout_omnidocbench_499.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
