"""Evaluate Azure Layout results from mally DI pipeline against benchmarks.

Extracts raw Azure Layout markdown from MinIO ocr.zip and evaluates
using the same metrics as the ocr_test benchmark framework.
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
MINIO_ALIAS = "local"
MINIO_BUCKET = "alli-files"
PROJECT_ID = "678f65c06e31e6ec0341ba81"
COMPOSE_DIR = "/Users/hanjuncho/code_base/CCW/mally"

PREPARED_DIR = Path("/Users/hanjuncho/Downloads/prepared_datasets")

# Map benchmark_key → folder of KB IDs to evaluate
# Will be populated dynamically from MongoDB


@dataclass
class AzureResult:
    kb_id: str
    file_name: str
    idx: int  # maps to prepared dataset index (00000.jpg → 0)
    markdown: str
    latency_ms: float = 0.0  # not available from stored results


def _run_cmd(cmd: str, timeout: int = 30) -> str:
    """Run a shell command and return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        cwd=COMPOSE_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nstderr: {result.stderr}")
    return result.stdout.strip()


def _get_kb_ids_for_folder(folder_name: str) -> list[dict]:
    """Get KB IDs and filenames from MongoDB for a given folder name."""
    script = f"""
    var folder = db.knowledge_base_folder.findOne({{name: "{folder_name}"}});
    if (!folder) {{ print("[]"); quit(); }}
    var folderId = folder._id.toString();
    var kbs = db.knowledge_base.find({{
        project: ObjectId("{PROJECT_ID}"),
        target_folder_id: folderId,
        process_state: "completed"
    }});
    var results = [];
    kbs.forEach(function(doc) {{
        results.push({{kb_id: doc._id.toString(), file_name: doc.file_name}});
    }});
    print(JSON.stringify(results));
    """
    out = _run_cmd(f"docker compose exec -T mongodb mongosh mally_dev --quiet --eval '{script}'")
    return json.loads(out)


def _get_kb_ids_by_ids(kb_ids: list[str]) -> list[dict]:
    """Get KB info directly by IDs."""
    oid_list = ", ".join(f'ObjectId("{kid}")' for kid in kb_ids)
    script = f"""
    var kbs = db.knowledge_base.find({{_id: {{$in: [{oid_list}]}}}});
    var results = [];
    kbs.forEach(function(doc) {{
        results.push({{kb_id: doc._id.toString(), file_name: doc.file_name}});
    }});
    print(JSON.stringify(results));
    """
    out = _run_cmd(f"docker compose exec -T mongodb mongosh mally_dev --quiet --eval '{script}'")
    return json.loads(out)


def _find_ocr_zip_path(kb_id: str) -> str | None:
    """Find the ocr.zip path in MinIO for a given KB."""
    try:
        out = _run_cmd(
            f"docker compose exec -T minio mc find {MINIO_ALIAS}/{MINIO_BUCKET}/{PROJECT_ID}/{kb_id}/ --name 'ocr.zip'",
            timeout=10,
        )
        if out:
            return out.strip().split("\n")[0]
    except Exception:
        pass
    return None


def _extract_markdown_from_ocr_zip(ocr_zip_path: str) -> str:
    """Download ocr.zip from MinIO and extract markdown content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        local_zip = Path(tmpdir) / "ocr.zip"
        # Copy from minio container to host
        _run_cmd(f"docker compose exec -T minio mc cp {ocr_zip_path} /tmp/_eval_ocr.zip", timeout=10)
        _run_cmd(f"docker compose cp minio:/tmp/_eval_ocr.zip {local_zip}", timeout=10)

        with zipfile.ZipFile(local_zip) as zf:
            # Find all ocr-*.json files
            ocr_files = sorted(f for f in zf.namelist() if f.startswith("ocr-") and f.endswith(".json"))
            markdowns = []
            for ocr_file in ocr_files:
                data = json.loads(zf.read(ocr_file))
                content = data.get("content", "")
                if content:
                    markdowns.append(content)
            return "\n\n".join(markdowns)


def load_azure_results(kb_ids: list[str]) -> list[AzureResult]:
    """Load Azure Layout markdown for given KB IDs."""
    kb_infos = _get_kb_ids_by_ids(kb_ids)
    results = []
    for info in kb_infos:
        kb_id = info["kb_id"]
        file_name = info["file_name"]
        # Extract index from filename: 00003.jpg → 3
        idx = int(Path(file_name).stem)

        ocr_zip_path = _find_ocr_zip_path(kb_id)
        if not ocr_zip_path:
            print(f"  [WARN] No ocr.zip for {file_name} (kb={kb_id})")
            continue

        markdown = _extract_markdown_from_ocr_zip(ocr_zip_path)
        results.append(AzureResult(kb_id=kb_id, file_name=file_name, idx=idx, markdown=markdown))
        print(f"  Loaded {file_name} → {len(markdown)} chars")

    return sorted(results, key=lambda r: r.idx)


def load_benchmark_samples(benchmark_key: str, indices: list[int]) -> list[dict]:
    """Load ground truth samples for given indices from prepared dataset."""
    meta_path = PREPARED_DIR / benchmark_key / "metadata.jsonl"
    samples = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["idx"] in indices:
                samples[rec["idx"]] = rec
    return [samples[i] for i in sorted(indices) if i in samples]


def _patch_config_paths():
    """Patch config.py paths before import.

    config.py calls mkdir() at import time with /root/... paths.
    We inject a pre-configured module into sys.modules so that
    `from config import ...` picks up our local paths instead.
    """
    import importlib.util
    import types

    local_base = Path("/Users/hanjuncho/code_base/CCW/ocr_test")
    results_dir = local_base / "results_local"
    data_cache_dir = local_base / "data_cache_local"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_cache_dir.mkdir(parents=True, exist_ok=True)

    # Load config source without executing mkdir
    spec = importlib.util.spec_from_file_location("config", local_base / "config.py")
    config_mod = importlib.util.module_from_spec(spec)

    # Override paths BEFORE exec
    import sys as _sys
    _sys.modules["config"] = config_mod

    # Patch Path.mkdir to be no-op for /root paths during config import
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

    # Now override to our local paths
    config_mod.RESULTS_DIR = results_dir
    config_mod.DATA_CACHE_DIR = data_cache_dir
    config_mod.PREPARED_DIR = PREPARED_DIR

    # Also patch OmniDocBench root path for benchmarks.py and metrics.py
    import benchmarks as _bench_mod
    _bench_mod._OMNIDOCBENCH_ROOT = str(local_base / "OmniDocBench")

    import metrics as _metrics_mod
    _metrics_mod._OMNIDOCBENCH_METRICS = str(local_base / "OmniDocBench" / "metrics")


def evaluate_omnidocbench(azure_results: list[AzureResult]) -> dict:
    """Evaluate Azure Layout results against OmniDocBench GT."""
    from benchmarks import _eval_document_parse

    indices = [r.idx for r in azure_results]
    samples = load_benchmark_samples("omnidocbench", indices)

    per_sample = []
    for result, sample in zip(azure_results, samples):
        assert result.idx == sample["idx"], f"Index mismatch: {result.idx} vs {sample['idx']}"
        gt = sample["ground_truth"]
        metadata = sample.get("metadata", {})
        try:
            scores = _eval_document_parse(result.markdown, gt, metadata)
            per_sample.append({
                "sample_id": sample["sample_id"],
                "file_name": result.file_name,
                "scores": scores,
                "error": None,
            })
            print(f"  {result.file_name}: text={scores.get('text_score', 0):.1f}, "
                  f"table={scores.get('table_teds', 0):.1f}, "
                  f"formula={scores.get('formula_score', 0):.1f}, "
                  f"overall={scores.get('overall', 0):.1f}")
        except Exception as e:
            import traceback as _tb
            per_sample.append({
                "sample_id": sample["sample_id"],
                "file_name": result.file_name,
                "scores": {},
                "error": str(e),
            })
            print(f"  {result.file_name}: ERROR - {e}")
            _tb.print_exc()

    # Aggregate
    valid = [r for r in per_sample if r["scores"]]
    if valid:
        avg = {}
        for key in valid[0]["scores"]:
            avg[key] = sum(r["scores"][key] for r in valid) / len(valid)
        return {"aggregate": avg, "per_sample": per_sample}
    return {"aggregate": {}, "per_sample": per_sample}


def evaluate_dp_bench(azure_results: list[AzureResult]) -> dict:
    """Evaluate Azure Layout results against DP-Bench GT."""
    from benchmarks import _eval_dp_bench

    indices = [r.idx for r in azure_results]
    samples = load_benchmark_samples("upstage_dp_bench", indices)

    per_sample = []
    for result, sample in zip(azure_results, samples):
        assert result.idx == sample["idx"]
        gt = sample["ground_truth"]
        metadata = sample.get("metadata", {})
        try:
            scores = _eval_dp_bench(result.markdown, gt, metadata)
            per_sample.append({
                "sample_id": sample["sample_id"],
                "file_name": result.file_name,
                "scores": scores,
                "error": None,
            })
            score_parts = [f"nid={scores.get('nid', 0):.4f}"]
            if "teds" in scores:
                score_parts.append(f"teds={scores['teds']:.4f}")
            print(f"  {result.file_name}: {', '.join(score_parts)}")
        except Exception as e:
            per_sample.append({
                "sample_id": sample["sample_id"],
                "file_name": result.file_name,
                "scores": {},
                "error": str(e),
            })
            print(f"  {result.file_name}: ERROR - {e}")

    valid = [r for r in per_sample if r["scores"]]
    if valid:
        avg = {}
        for key in ["nid", "teds", "teds_structure"]:
            vals = [r["scores"][key] for r in valid if key in r["scores"]]
            if vals:
                avg[key] = sum(vals) / len(vals)
        return {"aggregate": avg, "per_sample": per_sample}
    return {"aggregate": {}, "per_sample": per_sample}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    _patch_config_paths()

    # Test with 5 JPG samples uploaded to mally
    TEST_KB_IDS = [
        "69ab6734b66d3009ff073e76",  # 00001.jpg
        "69ab6734b66d3009ff073e77",  # 00003.jpg
        "69ab6735b66d3009ff073e78",  # 00002.jpg
        "69ab6735b66d3009ff073e79",  # 00000.jpg
        "69ab6735b66d3009ff073e7a",  # 00004.jpg
    ]

    print("=" * 60)
    print("Loading Azure Layout results from MinIO...")
    print("=" * 60)
    azure_results = load_azure_results(TEST_KB_IDS)
    print(f"\nLoaded {len(azure_results)} results")

    print("\n" + "=" * 60)
    print("Evaluating against OmniDocBench (5 samples)")
    print("=" * 60)
    omni_results = evaluate_omnidocbench(azure_results)
    print(f"\n--- OmniDocBench Aggregate ---")
    for k, v in omni_results["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    print("\nDone.")
