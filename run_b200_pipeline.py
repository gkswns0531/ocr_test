#!/usr/bin/env python3
"""B200 self-contained OCR + Embedding pipeline for SDS-KoPub 40K pages.

Single-file script: downloads dataset, runs GLM-OCR, computes embeddings,
uploads results to HuggingFace. No external src/ imports needed.

Usage:
    # Full pipeline (FP8 VLM + FP32 layout, B200 defaults)
    python3 run_b200_pipeline.py --hf-repo USER/REPO --hf-token hf_xxx

    # Individual steps
    python3 run_b200_pipeline.py --step ocr          # OCR only (FP8 default)
    python3 run_b200_pipeline.py --step embed         # Embeddings only
    python3 run_b200_pipeline.py --step upload --hf-repo USER/REPO --hf-token hf_xxx

    # Resume interrupted OCR
    python3 run_b200_pipeline.py --step ocr --resume

    # Custom dtype / L4 settings
    python3 run_b200_pipeline.py --step ocr --ocr-dtype bf16 --gpu-mem-util 0.60 --batch-size 4 --workers 64
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ===========================================================================
# Constants
# ===========================================================================
DATASET_ID = "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark"
DATASET_CACHE_DIR = Path("data")
OUTPUT_DIR = Path("output")
OCR_RESULTS_FILE = OUTPUT_DIR / "ocr_results.jsonl"
PARSED_TEXTS_FILE = OUTPUT_DIR / "parsed_texts.jsonl"
CROPS_DIR = OUTPUT_DIR / "crops"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

VLLM_LOG = Path("/tmp/vllm_glm_b200.log")
VLLM_PORT = 8000

# Embedding model defaults
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B-FP8"
DEFAULT_MAX_PIXELS = 1_843_200

# Text extraction limits
CHARS_PER_TOKEN = 1.2
MAX_TOKENS = 3800

# Query instruction for Qwen3-VL-Embedding
QUERY_INSTRUCTION = "Find the document page that answers this question."
IMAGE_INSTRUCTION = "You are given an image. Represent the image for retrieving relevant information."


# ===========================================================================
# Utility functions (from run_ocr_pipeline.py, src/utils.py)
# ===========================================================================

def _release_memory() -> None:
    """Force Python and C allocator to release memory back to OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================================
# OCR text extraction (from src/ocr_processor.py)
# ===========================================================================

def linearize_html_table(html: str) -> str:
    """Convert an HTML table to pipe-delimited plain text."""
    try:
        from lxml import etree
    except ImportError:
        return _linearize_html_table_regex(html)

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html, parser)
        if tree is None:
            return _linearize_html_table_regex(html)

        rows: list[str] = []
        for tr in tree.iter("tr"):
            cells: list[str] = []
            for td in tr.iter("td", "th"):
                text = td.text_content().strip()
                colspan = int(td.get("colspan", "1"))
                cells.extend([text] * colspan)
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    except Exception:
        return _linearize_html_table_regex(html)


def _linearize_html_table_regex(html: str) -> str:
    """Regex fallback for HTML table linearization."""
    rows: list[str] = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        tr_content = tr_match.group(1)
        cells: list[str] = []
        for td_match in re.finditer(
            r"<t[dh][^>]*>(.*?)</t[dh]>", tr_content, re.DOTALL | re.IGNORECASE
        ):
            cell_text = re.sub(r"<[^>]+>", "", td_match.group(1)).strip()
            cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_page_text(regions: list[dict]) -> str:
    """Merge OCR regions into a single text string for embedding."""
    parts: list[str] = []

    for r in regions:
        content = r.get("content")
        if not content:
            continue

        label = r.get("label", "")

        if label == "table":
            linearized = linearize_html_table(content)
            if linearized.strip():
                parts.append(linearized)
        elif label in ("display_formula", "inline_formula", "formula"):
            parts.append(content.strip())
        else:
            text = content.strip()
            if text:
                parts.append(text)

    merged = "\n\n".join(parts)

    # Token overflow protection
    max_chars = int(MAX_TOKENS * CHARS_PER_TOKEN)
    if len(merged) > max_chars:
        merged = merged[:max_chars]
        last_break = max(merged.rfind("\n\n"), merged.rfind("\n"), merged.rfind(". "))
        if last_break > max_chars * 0.8:
            merged = merged[:last_break]

    return merged


def process_ocr_to_texts(
    ocr_results_path: Path,
    output_path: Path,
) -> dict[str, str]:
    """Process ocr_results.jsonl → parsed_texts.jsonl and return page_id→text mapping."""
    parsed: dict[str, str] = {}

    with open(ocr_results_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Extracting page texts"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        page_id = record["page_id"]
        regions = record.get("regions", [])
        text = extract_page_text(regions)
        parsed[page_id] = text

    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for page_id, text in parsed.items():
            f.write(json.dumps({"page_id": page_id, "parsed_text": text}, ensure_ascii=False) + "\n")

    non_empty = sum(1 for t in parsed.values() if t.strip())
    avg_len = sum(len(t) for t in parsed.values()) / max(len(parsed), 1)
    print(f"Parsed {len(parsed)} pages: {non_empty} non-empty, avg {avg_len:.0f} chars")
    print(f"Saved to {output_path}")

    return parsed


# ===========================================================================
# Image resize (from src/data_loader.py)
# ===========================================================================

def resize_to_max_pixels(img: Image.Image, max_pixels: int = DEFAULT_MAX_PIXELS) -> Image.Image:
    """Resize image to fit within max_pixels while preserving aspect ratio."""
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ===========================================================================
# OCR pipeline functions (from run_ocr_pipeline.py)
# ===========================================================================

def load_processed_ids(path: Path) -> set[str]:
    """Load page_ids that have already been processed."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(obj["page_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _crop_region(img: Image.Image, bbox_2d: list, save_path: Path) -> bool:
    """Crop a region from a PIL Image using bbox_2d in 0-1000 normalized coords."""
    try:
        x1_n, y1_n, x2_n, y2_n = bbox_2d
        w, h = img.size
        x1 = int(x1_n * w / 1000)
        y1 = int(y1_n * h / 1000)
        x2 = int(x2_n * w / 1000)
        y2 = int(y2_n * h / 1000)
        box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        cropped = img.crop(box)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(str(save_path), format="JPEG", quality=95)
        return True
    except Exception:
        return False


def page_result_to_record(
    page_id: str,
    page_idx: int,
    result,
    img_path: str,
    do_crops: bool = True,
) -> dict:
    """Convert a GlmOcr result into a serializable dict."""
    regions_raw = result.json_result[0] if result.json_result else []

    regions = []
    for r in regions_raw:
        region = {
            "index": r.get("index"),
            "label": r.get("label"),
            "native_label": r.get("native_label"),
            "bbox_2d": r.get("bbox_2d"),
            "content": r.get("content"),
        }
        regions.append(region)

    image_crops: list[dict] = []
    if do_crops:
        crop_targets = [
            r for r in regions_raw
            if r.get("label") in ("image", "chart") and r.get("bbox_2d")
        ]
        if crop_targets:
            img = Image.open(img_path)
            safe_id = page_id.replace("/", "__")
            for r in crop_targets:
                crop_name = f"{safe_id}_crop_{r['index']}.jpg"
                crop_path = CROPS_DIR / crop_name
                ok = _crop_region(img, r["bbox_2d"], crop_path)
                image_crops.append({
                    "path": str(crop_path),
                    "bbox": r["bbox_2d"],
                    "label": r["label"],
                    "native_label": r.get("native_label", ""),
                    "index": r["index"],
                    "saved": ok,
                })

    record = {
        "page_id": page_id,
        "page_idx": page_idx,
        "regions": regions,
        "markdown": result.markdown_result or "",
        "image_crops": image_crops,
    }
    return record


def create_patched_config(port: int, workers: int, batch_size: int) -> str:
    """Create a patched GLM-OCR config with custom port/workers/batch_size."""
    sdk_config = Path("/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml")
    if not sdk_config.exists():
        import glmocr
        sdk_config = Path(glmocr.__file__).parent / "config.yaml"

    cfg_dir = Path(tempfile.mkdtemp(prefix="glmocr_cfg_"))
    cfg_path = cfg_dir / "config.yaml"
    shutil.copy(sdk_config, cfg_path)

    text = cfg_path.read_text()
    # Port: try both common defaults
    text = text.replace("api_port: 8080", f"api_port: {port}")
    text = text.replace("api_port: 5002", f"api_port: {port}")
    text = text.replace("port: 5002", f"port: {port}")
    text = text.replace("level: INFO", "level: WARNING")
    text = text.replace("max_tokens: 4096", "max_tokens: 16384")
    text = text.replace("max_workers: 32", f"max_workers: {workers}")
    text = text.replace("batch_size: 1", f"batch_size: {batch_size}")
    cfg_path.write_text(text)
    return str(cfg_path)


# ===========================================================================
# Embedding model (from src/embedding.py, self-contained)
# ===========================================================================

@dataclass
class EmbeddingModelConfig:
    name: str
    model_id: str
    batch_size_image: int = 32
    batch_size_text: int = 8
    quantization: str | None = None
    gpu_memory_utilization: float | None = None


class VLEmbeddingModel:
    """Wrapper around Qwen3-VL-Embedding using vLLM pooling runner."""

    def __init__(self, model_cfg: EmbeddingModelConfig):
        from vllm import LLM
        from transformers import AutoProcessor

        self.model_cfg = model_cfg

        print(f"Loading embedding model via vLLM: {model_cfg.model_id}")

        self.processor = AutoProcessor.from_pretrained(
            model_cfg.model_id,
            trust_remote_code=True,
        )

        kwargs = {
            "model": model_cfg.model_id,
            "runner": "pooling",
            "max_model_len": 4096,
            "trust_remote_code": True,
        }
        if model_cfg.quantization:
            kwargs["quantization"] = model_cfg.quantization
            kwargs["dtype"] = "auto"
        else:
            kwargs["dtype"] = "bfloat16"
        if model_cfg.gpu_memory_utilization:
            kwargs["gpu_memory_utilization"] = model_cfg.gpu_memory_utilization

        self.llm = LLM(**kwargs)
        print(f"Embedding model loaded: {model_cfg.name}")

    def _build_image_input(self, img: Image.Image, instruction: str) -> dict:
        msg = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt, "multi_modal_data": {"image": img}}

    def _build_text_input(self, text: str, instruction: str) -> dict:
        content = [{"type": "text", "text": text}]
        if instruction:
            msg = [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content},
            ]
        else:
            msg = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt}

    def _extract_embeddings(self, outputs) -> np.ndarray:
        embs = np.array([out.outputs.embedding for out in outputs])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        return embs / norms

    def encode_images_lazy(
        self,
        total: int,
        image_loader,
        batch_size: int | None = None,
        instruction: str = IMAGE_INSTRUCTION,
    ) -> np.ndarray:
        """Encode images lazily — loads batch_size images at a time via callback."""
        bs = batch_size or self.model_cfg.batch_size_image
        all_embeddings: list[np.ndarray] = []

        for start in tqdm(range(0, total, bs), desc="Encoding images (lazy)"):
            indices = list(range(start, min(start + bs, total)))
            batch_images = image_loader(indices)
            inputs = [self._build_image_input(img, instruction) for img in batch_images]
            del batch_images
            outputs = self.llm.embed(inputs)
            emb = self._extract_embeddings(outputs)
            all_embeddings.append(emb)
            del inputs, outputs
            _release_memory()

        return np.concatenate(all_embeddings, axis=0)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        instruction: str = "",
    ) -> np.ndarray:
        """Encode a list of texts into normalized embeddings."""
        bs = batch_size or self.model_cfg.batch_size_text
        all_embeddings: list[np.ndarray] = []

        for start in tqdm(range(0, len(texts), bs), desc="Encoding texts"):
            batch_texts = texts[start : start + bs]
            inputs = [self._build_text_input(t, instruction) for t in batch_texts]
            outputs = self.llm.embed(inputs)
            emb = self._extract_embeddings(outputs)
            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="B200 OCR + Embedding pipeline for SDS-KoPub 40K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python3 run_b200_pipeline.py --hf-repo user/repo --hf-token hf_xxx

  # Individual steps
  python3 run_b200_pipeline.py --step ocr
  python3 run_b200_pipeline.py --step embed
  python3 run_b200_pipeline.py --step upload --hf-repo user/repo --hf-token hf_xxx
""",
    )

    # Step control
    p.add_argument(
        "--step",
        choices=["ocr", "embed", "upload"],
        default=None,
        help="Run a specific step only. If omitted, runs all steps.",
    )
    p.add_argument("--resume", action="store_true", help="Resume interrupted OCR (skip processed pages)")

    # B200-optimized defaults
    p.add_argument("--batch-size", type=int, default=8, help="Layout model batch size (default: 8 for B200)")
    p.add_argument("--workers", type=int, default=128, help="VLM parallel workers (default: 128 for B200)")
    p.add_argument("--port", type=int, default=VLLM_PORT, help="vLLM server port")
    p.add_argument("--gpu-mem-util", type=float, default=0.80, help="vLLM GPU memory utilization (default: 0.80)")
    p.add_argument("--no-crops", action="store_true", help="Skip saving image/chart crops")
    p.add_argument(
        "--ocr-dtype",
        choices=["fp8", "bf16", "auto"],
        default="fp8",
        help="GLM-OCR vLLM inference dtype (default: fp8 for B200 Blackwell)",
    )

    # Embedding
    p.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model ID",
    )
    p.add_argument("--embed-batch-image", type=int, default=32, help="Image embedding batch size")
    p.add_argument("--embed-batch-text", type=int, default=8, help="Text embedding batch size")

    # HuggingFace upload
    p.add_argument("--hf-repo", type=str, default=None, help="HuggingFace dataset repo ID (e.g. user/repo)")
    p.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")

    # Output
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")

    return p.parse_args()


# ===========================================================================
# Step 1: Environment check
# ===========================================================================

def step_check_environment() -> None:
    """Verify CUDA, vLLM, glmocr, and other dependencies are available."""
    print("\n" + "=" * 60)
    print("Step 1: Environment Check")
    print("=" * 60)

    # CUDA
    import torch
    assert torch.cuda.is_available(), "CUDA not available!"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")

    # vLLM
    import vllm
    print(f"  vLLM: {vllm.__version__}")

    # GLM-OCR SDK
    import glmocr
    print(f"  GLM-OCR: {glmocr.__version__ if hasattr(glmocr, '__version__') else 'installed'}")

    # datasets
    import datasets
    print(f"  datasets: {datasets.__version__}")

    # huggingface_hub
    import huggingface_hub
    print(f"  huggingface_hub: {huggingface_hub.__version__}")

    print("  All dependencies OK!")


# ===========================================================================
# Step 2: Download dataset
# ===========================================================================

def step_download_dataset(cache_dir: Path) -> Path | None:
    """Download SDS-KoPub dataset from HuggingFace and return corpus path."""
    print("\n" + "=" * 60)
    print("Step 2: Download Dataset")
    print("=" * 60)

    from datasets import load_dataset

    print(f"  Downloading {DATASET_ID} corpus...")
    corpus_ds = load_dataset(
        DATASET_ID,
        name="SDS-KoPub-corpus",
        cache_dir=str(cache_dir),
        split="test",
    )
    print(f"  Corpus: {len(corpus_ds)} pages")

    print(f"  Downloading {DATASET_ID} QA...")
    qa_ds = load_dataset(
        DATASET_ID,
        name="SDS-KoPub-QA",
        cache_dir=str(cache_dir),
        split="test",
    )
    print(f"  QA: {len(qa_ds)} items")

    # Find the Arrow shard directory
    # The dataset is cached under data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/...
    # We need to find the actual Arrow files for the OCR pipeline
    arrow_dir = _find_arrow_dir(cache_dir)
    if arrow_dir:
        print(f"  Arrow dir: {arrow_dir}")
    else:
        print("  Warning: Could not locate Arrow shard directory. Will use HF dataset API instead.")

    del corpus_ds, qa_ds
    _release_memory()

    return arrow_dir  # type: ignore[return-value]


def _find_arrow_dir(cache_dir: Path) -> Path | None:
    """Find the directory containing Arrow shard files for the corpus."""
    # Walk the cache directory looking for .arrow files
    for root, _dirs, files in os.walk(cache_dir):
        arrow_files = [f for f in files if f.endswith(".arrow")]
        if len(arrow_files) >= 10:  # Corpus has 35 shards
            return Path(root)
    return None


# ===========================================================================
# Step 3: Start vLLM GLM-OCR server
# ===========================================================================

def step_start_vllm_server(
    port: int, gpu_mem_util: float, ocr_dtype: str = "fp8",
) -> subprocess.Popen | None:
    """Start vLLM GLM-OCR server as subprocess."""
    print("\n" + "=" * 60)
    print("Step 3: Start vLLM GLM-OCR Server")
    print("=" * 60)

    # Check if server is already running
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3)
        if resp.status == 200:
            print(f"  vLLM server already running on port {port}")
            return None
    except Exception:
        pass

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "zai-org/GLM-OCR",
        "--served-model-name", "glm-ocr",
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", "131072",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
    ]

    # Dtype / quantization
    if ocr_dtype == "fp8":
        # FP8 dynamic quantization — Blackwell B200 has native FP8 support
        # GLM-OCR is BF16 natively; vLLM quantizes weights+activations to FP8 on-the-fly
        cmd.extend(["--quantization", "fp8", "--dtype", "auto"])
        print(f"  Dtype: FP8 (dynamic quantization on BF16 model)")
    elif ocr_dtype == "bf16":
        cmd.extend(["--dtype", "bfloat16"])
        print(f"  Dtype: BF16 (native)")
    else:
        cmd.extend(["--dtype", "auto"])
        print(f"  Dtype: auto (model native)")

    print(f"  Starting: {' '.join(cmd[:6])}...")
    log_f = open(VLLM_LOG, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"  PID: {proc.pid}, log: {VLLM_LOG}")

    # Wait for server to be ready
    print("  Waiting for server to start (CUDA graph compilation)...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3)
            if resp.status == 200:
                elapsed = time.time() - start_time
                print(f"  Server ready in {elapsed:.0f}s")
                return proc
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            log_f.close()
            print(f"  ERROR: vLLM server exited with code {proc.returncode}")
            print(f"  Check log: {VLLM_LOG}")
            sys.exit(1)
        time.sleep(5)

    print(f"  ERROR: vLLM server did not start within {max_wait}s")
    print(f"  Check log: {VLLM_LOG}")
    step_stop_vllm_server(proc)
    sys.exit(1)


# ===========================================================================
# Step 4: OCR parsing
# ===========================================================================

def step_ocr_parsing(
    arrow_dir: Path | None,
    cache_dir: Path,
    port: int,
    workers: int,
    batch_size: int,
    do_crops: bool,
    resume: bool,
) -> None:
    """Run GLM-OCR on all 40K pages → ocr_results.jsonl + crops/."""
    print("\n" + "=" * 60)
    print("Step 4: OCR Parsing")
    print("=" * 60)

    ensure_dir(OUTPUT_DIR)
    if do_crops:
        ensure_dir(CROPS_DIR)

    # Resume support
    processed_ids = load_processed_ids(OCR_RESULTS_FILE) if resume else set()
    if processed_ids:
        print(f"  Resuming: {len(processed_ids)} pages already processed")

    # Patch SDK config
    cfg_path = create_patched_config(port, workers, batch_size)
    print(f"  Patched config: {cfg_path}")

    # Initialize GLM-OCR SDK
    from glmocr import GlmOcr
    ocr = GlmOcr(config_path=cfg_path)
    print("  GLM-OCR SDK initialized")

    # Load dataset: either from Arrow shards or HF API
    if arrow_dir and arrow_dir.exists():
        _ocr_from_arrow_shards(ocr, arrow_dir, processed_ids, batch_size, do_crops)
    else:
        _ocr_from_hf_dataset(ocr, cache_dir, processed_ids, batch_size, do_crops)

    ocr.close()
    print("  OCR parsing complete!")


def _ocr_from_arrow_shards(
    ocr,
    arrow_dir: Path,
    processed_ids: set[str],
    batch_size: int,
    do_crops: bool,
) -> None:
    """Process pages from Arrow shard files."""
    from datasets import Dataset

    arrow_files = sorted(f for f in os.listdir(arrow_dir) if f.endswith(".arrow"))
    print(f"  Found {len(arrow_files)} Arrow shards in {arrow_dir}")

    # Count total pages
    total_pages = 0
    for af in arrow_files:
        ds = Dataset.from_file(str(arrow_dir / af))
        total_pages += len(ds)
        del ds
    print(f"  Total pages: {total_pages}")

    global_idx = 0
    pages_processed = 0
    pages_skipped = len(processed_ids)
    t_start = time.time()

    out_f = open(OCR_RESULTS_FILE, "a", encoding="utf-8")

    try:
        for shard_idx, arrow_file in enumerate(arrow_files):
            ds = Dataset.from_file(str(arrow_dir / arrow_file))
            shard_size = len(ds)
            ids = ds["id"]

            # Collect pending indices
            pending_indices: list[int] = []
            pending_page_ids: list[str] = []
            for i in range(shard_size):
                page_id = str(ids[i])
                if page_id not in processed_ids:
                    pending_indices.append(i)
                    pending_page_ids.append(page_id)

            if not pending_indices:
                global_idx += shard_size
                print(f"  Shard {shard_idx}/{len(arrow_files)}: all done, skipping")
                del ds
                _release_memory()
                continue

            print(f"\n  Shard {shard_idx}/{len(arrow_files)}: {len(pending_indices)}/{shard_size} pending")

            pbar = tqdm(
                range(0, len(pending_indices), batch_size),
                desc=f"Shard {shard_idx}",
                total=(len(pending_indices) + batch_size - 1) // batch_size,
            )

            for batch_start in pbar:
                batch_end = min(batch_start + batch_size, len(pending_indices))
                batch_local_indices = pending_indices[batch_start:batch_end]
                batch_page_ids = pending_page_ids[batch_start:batch_end]

                # Save images to temp files
                tmp_paths: list[str] = []
                for local_idx in batch_local_indices:
                    row = ds[local_idx]
                    tmp_img = f"/tmp/ocr_batch_{shard_idx}_{local_idx}.jpg"
                    pil_img = row["image"].convert("RGB")
                    pil_img.save(tmp_img, format="JPEG", quality=95)
                    del pil_img, row
                    tmp_paths.append(tmp_img)
                _release_memory()

                # Run GLM-OCR
                results = None
                try:
                    if len(tmp_paths) == 1:
                        results = [ocr.parse(tmp_paths[0])]
                    else:
                        results = ocr.parse(tmp_paths)

                    for result, page_id, local_idx, tmp_path in zip(
                        results, batch_page_ids, batch_local_indices, tmp_paths
                    ):
                        record = page_result_to_record(
                            page_id=page_id,
                            page_idx=global_idx + local_idx,
                            result=result,
                            img_path=tmp_path,
                            do_crops=do_crops,
                        )
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pages_processed += 1
                    out_f.flush()

                except Exception as e:
                    print(f"\n  ERROR in batch: {e}")
                    for page_id, local_idx in zip(batch_page_ids, batch_local_indices):
                        record = {
                            "page_id": page_id,
                            "page_idx": global_idx + local_idx,
                            "regions": [],
                            "markdown": "",
                            "image_crops": [],
                            "error": str(e),
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pages_processed += 1
                    out_f.flush()

                # Clean up
                if results is not None:
                    del results
                for tmp in tmp_paths:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                _release_memory()

                # Progress
                elapsed = time.time() - t_start
                total_done = pages_processed + pages_skipped
                rate = pages_processed / elapsed if elapsed > 0 else 0
                eta_h = (total_pages - total_done) / rate / 3600 if rate > 0 else float("inf")
                pbar.set_postfix(done=total_done, rate=f"{rate:.1f}/s", eta=f"{eta_h:.1f}h")

            global_idx += shard_size
            del ds
            _release_memory()

    finally:
        out_f.close()

    elapsed = time.time() - t_start
    print(f"\n  Processed {pages_processed} new pages in {elapsed/3600:.1f}h")
    print(f"  Total in {OCR_RESULTS_FILE}: {pages_processed + pages_skipped} pages")


def _ocr_from_hf_dataset(
    ocr,
    cache_dir: Path,
    processed_ids: set[str],
    batch_size: int,
    do_crops: bool,
) -> None:
    """Process pages from HuggingFace dataset API (fallback when Arrow dir not found)."""
    from datasets import load_dataset

    print("  Loading dataset via HF API...")
    corpus_ds = load_dataset(
        DATASET_ID,
        name="SDS-KoPub-corpus",
        cache_dir=str(cache_dir),
        split="test",
    )
    total_pages = len(corpus_ds)
    print(f"  Total pages: {total_pages}")

    ids = corpus_ds["id"]
    pending_indices: list[int] = []
    pending_page_ids: list[str] = []
    for i in range(total_pages):
        page_id = str(ids[i])
        if page_id not in processed_ids:
            pending_indices.append(i)
            pending_page_ids.append(page_id)

    print(f"  Pending: {len(pending_indices)} pages")

    pages_processed = 0
    t_start = time.time()

    out_f = open(OCR_RESULTS_FILE, "a", encoding="utf-8")

    try:
        pbar = tqdm(
            range(0, len(pending_indices), batch_size),
            desc="OCR",
            total=(len(pending_indices) + batch_size - 1) // batch_size,
        )

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(pending_indices))
            batch_idx = pending_indices[batch_start:batch_end]
            batch_pids = pending_page_ids[batch_start:batch_end]

            # Save images to temp files
            tmp_paths: list[str] = []
            for idx in batch_idx:
                row = corpus_ds[idx]
                tmp_img = f"/tmp/ocr_hf_{idx}.jpg"
                pil_img = row["image"].convert("RGB")
                pil_img.save(tmp_img, format="JPEG", quality=95)
                del pil_img, row
                tmp_paths.append(tmp_img)
            _release_memory()

            results = None
            try:
                if len(tmp_paths) == 1:
                    results = [ocr.parse(tmp_paths[0])]
                else:
                    results = ocr.parse(tmp_paths)

                for result, page_id, idx, tmp_path in zip(results, batch_pids, batch_idx, tmp_paths):
                    record = page_result_to_record(
                        page_id=page_id,
                        page_idx=idx,
                        result=result,
                        img_path=tmp_path,
                        do_crops=do_crops,
                    )
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pages_processed += 1
                out_f.flush()

            except Exception as e:
                print(f"\n  ERROR in batch: {e}")
                for page_id, idx in zip(batch_pids, batch_idx):
                    record = {
                        "page_id": page_id,
                        "page_idx": idx,
                        "regions": [],
                        "markdown": "",
                        "image_crops": [],
                        "error": str(e),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pages_processed += 1
                out_f.flush()

            if results is not None:
                del results
            for tmp in tmp_paths:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            _release_memory()

            elapsed = time.time() - t_start
            rate = pages_processed / elapsed if elapsed > 0 else 0
            remaining = len(pending_indices) - (batch_start + batch_size)
            eta_h = remaining / rate / 3600 if rate > 0 else float("inf")
            pbar.set_postfix(done=pages_processed, rate=f"{rate:.1f}/s", eta=f"{eta_h:.1f}h")

    finally:
        out_f.close()

    del corpus_ds
    _release_memory()

    elapsed = time.time() - t_start
    print(f"\n  Processed {pages_processed} pages in {elapsed/3600:.1f}h")


# ===========================================================================
# Step 5: Stop vLLM server
# ===========================================================================

def step_stop_vllm_server(proc: subprocess.Popen | None) -> None:
    """Stop vLLM GLM-OCR server."""
    print("\n" + "=" * 60)
    print("Step 5: Stop vLLM Server")
    print("=" * 60)

    if proc is None:
        print("  No server process to stop (was already running or not started)")
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
        print(f"  Server stopped (PID {proc.pid})")
    except ProcessLookupError:
        print("  Server already stopped")
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)
        print(f"  Server force-killed (PID {proc.pid})")


# ===========================================================================
# Step 6: Embeddings
# ===========================================================================

def step_compute_embeddings(
    cache_dir: Path,
    embedding_model_id: str,
    batch_size_image: int,
    batch_size_text: int,
) -> None:
    """Compute image, text, and query embeddings with Qwen3-VL-Embedding."""
    print("\n" + "=" * 60)
    print("Step 6: Compute Embeddings")
    print("=" * 60)

    ensure_dir(EMBEDDINGS_DIR)

    # 6a. Extract parsed texts from OCR results
    if not PARSED_TEXTS_FILE.exists():
        print("  Extracting parsed texts from OCR results...")
        process_ocr_to_texts(OCR_RESULTS_FILE, PARSED_TEXTS_FILE)
    else:
        print(f"  Using existing {PARSED_TEXTS_FILE}")

    # Load parsed texts (preserving order)
    page_ids: list[str] = []
    page_texts: list[str] = []
    with open(PARSED_TEXTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                page_ids.append(obj["page_id"])
                page_texts.append(obj.get("parsed_text", ""))

    print(f"  Loaded {len(page_ids)} parsed texts")

    # Determine quantization from model name
    quantization = "fp8" if "FP8" in embedding_model_id.upper() else None

    model_cfg = EmbeddingModelConfig(
        name="qwen3-vl-embedding",
        model_id=embedding_model_id,
        batch_size_image=batch_size_image,
        batch_size_text=batch_size_text,
        quantization=quantization,
        gpu_memory_utilization=0.90,
    )

    model = VLEmbeddingModel(model_cfg)

    # 6b. Image embeddings (lazy loading)
    corpus_images_path = EMBEDDINGS_DIR / "corpus_images.npy"
    if corpus_images_path.exists():
        print(f"  Skipping image embeddings (already exists: {corpus_images_path})")
    else:
        print("  Computing image embeddings...")
        from datasets import load_dataset

        corpus_ds = load_dataset(
            DATASET_ID,
            name="SDS-KoPub-corpus",
            cache_dir=str(cache_dir),
            split="test",
        )

        def image_loader(indices: list[int]) -> list[Image.Image]:
            rows = corpus_ds.select(indices)
            return [resize_to_max_pixels(row["image"]) for row in rows]

        image_embs = model.encode_images_lazy(
            total=len(corpus_ds),
            image_loader=image_loader,
            batch_size=batch_size_image,
        )
        np.save(str(corpus_images_path), image_embs)
        print(f"  Saved: {corpus_images_path} (shape={image_embs.shape})")
        del image_embs, corpus_ds
        _release_memory()

    # 6c. OCR text embeddings
    corpus_text_path = EMBEDDINGS_DIR / "corpus_ocr_text.npy"
    if corpus_text_path.exists():
        print(f"  Skipping text embeddings (already exists: {corpus_text_path})")
    else:
        print("  Computing OCR text embeddings...")
        text_embs = model.encode_texts(
            page_texts,
            batch_size=batch_size_text,
            instruction="",
        )
        np.save(str(corpus_text_path), text_embs)
        print(f"  Saved: {corpus_text_path} (shape={text_embs.shape})")
        del text_embs
        _release_memory()

    # 6d. Query embeddings
    queries_path = EMBEDDINGS_DIR / "queries.npy"
    if queries_path.exists():
        print(f"  Skipping query embeddings (already exists: {queries_path})")
    else:
        print("  Computing query embeddings...")
        from datasets import load_dataset

        qa_ds = load_dataset(
            DATASET_ID,
            name="SDS-KoPub-QA",
            cache_dir=str(cache_dir),
            split="test",
        )
        query_texts = [item["query"] for item in qa_ds]
        print(f"  {len(query_texts)} queries")

        query_embs = model.encode_texts(
            query_texts,
            batch_size=batch_size_text,
            instruction=QUERY_INSTRUCTION,
        )
        np.save(str(queries_path), query_embs)
        print(f"  Saved: {queries_path} (shape={query_embs.shape})")
        del query_embs, qa_ds
        _release_memory()

    # Clean up model
    del model
    _release_memory()

    print("  Embedding computation complete!")


# ===========================================================================
# Step 7: HuggingFace upload
# ===========================================================================

def step_upload_to_hf(hf_repo: str, hf_token: str) -> None:
    """Upload results to HuggingFace."""
    print("\n" + "=" * 60)
    print("Step 7: Upload to HuggingFace")
    print("=" * 60)

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    # Create repo if not exists
    print(f"  Creating/verifying repo: {hf_repo}")
    api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)

    # Generate README
    readme_content = _generate_readme(hf_repo)
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    # Upload individual files
    files_to_upload = [
        ("README.md", "README.md"),
        ("ocr_results.jsonl", "ocr_results.jsonl"),
        ("parsed_texts.jsonl", "parsed_texts.jsonl"),
    ]

    for local_name, remote_name in files_to_upload:
        local_path = OUTPUT_DIR / local_name
        if local_path.exists():
            print(f"  Uploading {remote_name} ({local_path.stat().st_size / 1e6:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=hf_repo,
                repo_type="dataset",
            )
        else:
            print(f"  Skipping {remote_name} (not found)")

    # Upload embeddings
    for npy_name in ["corpus_images.npy", "corpus_ocr_text.npy", "queries.npy"]:
        npy_path = EMBEDDINGS_DIR / npy_name
        if npy_path.exists():
            size_mb = npy_path.stat().st_size / 1e6
            print(f"  Uploading embeddings/{npy_name} ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(npy_path),
                path_in_repo=f"embeddings/{npy_name}",
                repo_id=hf_repo,
                repo_type="dataset",
            )
        else:
            print(f"  Skipping embeddings/{npy_name} (not found)")

    # Upload crops directory (as tar.gz if it exists and has files)
    if CROPS_DIR.exists():
        crop_files = list(CROPS_DIR.glob("*.jpg"))
        if crop_files:
            print(f"  Compressing {len(crop_files)} crop images...")
            import tarfile
            tar_path = OUTPUT_DIR / "crops.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(str(CROPS_DIR), arcname="crops")
            size_mb = tar_path.stat().st_size / 1e6
            print(f"  Uploading crops.tar.gz ({size_mb:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(tar_path),
                path_in_repo="crops.tar.gz",
                repo_id=hf_repo,
                repo_type="dataset",
            )
        else:
            print("  No crop images to upload")

    print(f"\n  Upload complete! https://huggingface.co/datasets/{hf_repo}")


def _generate_readme(hf_repo: str) -> str:
    """Generate a HuggingFace dataset card README."""
    # Gather stats
    ocr_count = 0
    if OCR_RESULTS_FILE.exists():
        with open(OCR_RESULTS_FILE) as f:
            ocr_count = sum(1 for _ in f)

    emb_shapes = {}
    for name in ["corpus_images", "corpus_ocr_text", "queries"]:
        p = EMBEDDINGS_DIR / f"{name}.npy"
        if p.exists():
            arr = np.load(str(p), mmap_mode="r")
            emb_shapes[name] = arr.shape

    crop_count = len(list(CROPS_DIR.glob("*.jpg"))) if CROPS_DIR.exists() else 0

    return f"""---
license: cc-by-4.0
task_categories:
  - document-question-answering
  - visual-question-answering
language:
  - ko
tags:
  - ocr
  - document-understanding
  - embeddings
  - korean
size_categories:
  - 10K<n<100K
---

# SDS-KoPub OCR Results & Embeddings

OCR layout parsing results and VL embeddings for the
[SDS-KoPub-VDR-Benchmark](https://huggingface.co/datasets/SamsungSDS-Research/SDS-KoPub-VDR-Benchmark)
corpus ({ocr_count:,} Korean public document pages).

## Contents

| File | Description | Size |
|------|-------------|------|
| `ocr_results.jsonl` | GLM-OCR structured layout results (regions, markdown, bbox, labels) | {ocr_count:,} records |
| `parsed_texts.jsonl` | Extracted text per page (embedding input) | {ocr_count:,} records |
| `embeddings/corpus_images.npy` | Page image embeddings | {emb_shapes.get('corpus_images', 'N/A')} |
| `embeddings/corpus_ocr_text.npy` | OCR text embeddings | {emb_shapes.get('corpus_ocr_text', 'N/A')} |
| `embeddings/queries.npy` | Query embeddings | {emb_shapes.get('queries', 'N/A')} |
| `crops.tar.gz` | Image/chart region crops | {crop_count:,} images |

## Models Used

- **OCR**: [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B, layout via PP-DocLayoutV3)
- **Embeddings**: [Qwen3-VL-Embedding-2B-FP8](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B-FP8) (2048-dim)

## OCR Result Format

Each line in `ocr_results.jsonl`:
```json
{{
  "page_id": "doc_123_page_0",
  "page_idx": 0,
  "regions": [
    {{"index": 0, "label": "doc_title", "bbox_2d": [x1, y1, x2, y2], "content": "..."}},
    {{"index": 1, "label": "table", "bbox_2d": [...], "content": "<table>...</table>"}},
    {{"index": 2, "label": "image", "bbox_2d": [...], "content": null}}
  ],
  "markdown": "# Title\\n\\n| col1 | col2 |\\n...",
  "image_crops": [{{"path": "crops/doc_123_page_0_crop_2.jpg", "bbox": [...], "label": "image"}}]
}}
```

## Usage

```python
import json
import numpy as np
from huggingface_hub import hf_hub_download

# Load OCR results
path = hf_hub_download("{hf_repo}", "ocr_results.jsonl", repo_type="dataset")
with open(path) as f:
    records = [json.loads(line) for line in f]

# Load embeddings
img_emb = np.load(hf_hub_download("{hf_repo}", "embeddings/corpus_images.npy", repo_type="dataset"))
txt_emb = np.load(hf_hub_download("{hf_repo}", "embeddings/corpus_ocr_text.npy", repo_type="dataset"))
q_emb = np.load(hf_hub_download("{hf_repo}", "embeddings/queries.npy", repo_type="dataset"))

# Retrieval: cosine similarity (embeddings are L2-normalized)
scores = q_emb @ img_emb.T  # (num_queries, num_pages)
```

## Pipeline

Generated with `run_b200_pipeline.py` on NVIDIA B200 (192GB).
"""


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    # Override global paths if custom output dir
    global OUTPUT_DIR, OCR_RESULTS_FILE, PARSED_TEXTS_FILE, CROPS_DIR, EMBEDDINGS_DIR
    OUTPUT_DIR = Path(args.output_dir)
    OCR_RESULTS_FILE = OUTPUT_DIR / "ocr_results.jsonl"
    PARSED_TEXTS_FILE = OUTPUT_DIR / "parsed_texts.jsonl"
    CROPS_DIR = OUTPUT_DIR / "crops"
    EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

    print("=" * 60)
    print("B200 OCR + Embedding Pipeline")
    print("=" * 60)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Step: {args.step or 'all'}")
    print(f"  Batch size: {args.batch_size}, Workers: {args.workers}")
    print(f"  GPU mem util: {args.gpu_mem_util}")
    print(f"  OCR dtype: {args.ocr_dtype}")
    print(f"  Embedding model: {args.embedding_model}")

    steps = [args.step] if args.step else ["ocr", "embed", "upload"]

    t_total = time.time()
    vllm_proc = None

    try:
        # Step 1: Environment check (always run)
        step_check_environment()

        # Step 2: Download dataset (always ensure it's available)
        arrow_dir = step_download_dataset(DATASET_CACHE_DIR)

        if "ocr" in steps:
            # Step 3: Start vLLM server
            vllm_proc = step_start_vllm_server(args.port, args.gpu_mem_util, args.ocr_dtype)

            # Step 4: OCR parsing
            step_ocr_parsing(
                arrow_dir=arrow_dir,
                cache_dir=DATASET_CACHE_DIR,
                port=args.port,
                workers=args.workers,
                batch_size=args.batch_size,
                do_crops=not args.no_crops,
                resume=args.resume or (args.step is None),  # auto-resume in full mode
            )

            # Step 5: Stop vLLM server
            step_stop_vllm_server(vllm_proc)
            vllm_proc = None

        if "embed" in steps:
            # Step 6: Embeddings
            step_compute_embeddings(
                cache_dir=DATASET_CACHE_DIR,
                embedding_model_id=args.embedding_model,
                batch_size_image=args.embed_batch_image,
                batch_size_text=args.embed_batch_text,
            )

        if "upload" in steps:
            # Step 7: Upload
            if not args.hf_repo:
                print("\nERROR: --hf-repo is required for upload step")
                sys.exit(1)
            hf_token = args.hf_token or os.environ.get("HF_TOKEN")
            if not hf_token:
                print("\nERROR: --hf-token or HF_TOKEN env var required for upload")
                sys.exit(1)
            step_upload_to_hf(args.hf_repo, hf_token)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    finally:
        # Ensure vLLM server is stopped
        if vllm_proc is not None:
            step_stop_vllm_server(vllm_proc)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete! Total time: {elapsed/3600:.1f}h")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
