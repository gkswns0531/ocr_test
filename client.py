"""Inference clients for vLLM OCR models, DeepSeek-OCR2 offline, and MinerU."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import OpenAI
from PIL import Image

from config import ModelConfig

# Hard timeout (seconds) for a single inference call.
HARD_TIMEOUT = 120

# Default max_tokens per model (override the global 8192 default)
MODEL_MAX_TOKENS: dict[str, int] = {
    "DeepSeek-OCR2": 8192,
}


class VLLMOCRClient:
    """OpenAI-compatible client for vLLM-served OCR models."""

    def __init__(self, base_url: str, model_name: str, timeout: float = 120.0) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=httpx.Timeout(connect=10, read=timeout, write=30, pool=10),
        )

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Run inference with thread-based hard timeout. Returns (text, latency_ms)."""
        b64 = _image_to_base64(image)
        result_holder: list[str | Exception] = []

        def _call():
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                resp = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, max_tokens=max_tokens, temperature=0.0,
                )
                result_holder.append(resp.choices[0].message.content or "")
            except Exception as e:
                result_holder.append(e)

        t0 = time.time()
        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=HARD_TIMEOUT)
        latency_ms = (time.time() - t0) * 1000

        if thread.is_alive():
            print(f"[Client] HARD TIMEOUT ({HARD_TIMEOUT}s) — thread still running (daemon)")
            return "", latency_ms

        if result_holder:
            result = result_holder[0]
            if isinstance(result, Exception):
                print(f"[Client] Inference error ({latency_ms:.0f}ms): {type(result).__name__}: {result}")
                return "", latency_ms
            return result, latency_ms

        return "", latency_ms

    def batch_infer(
        self,
        images: list[Image.Image],
        prompts: list[str],
        max_tokens: int = 8192,
        max_workers: int = 1,
    ) -> list[tuple[str, float]]:
        """Send multiple inference requests concurrently via ThreadPoolExecutor.

        vLLM's continuous batching handles internal scheduling.
        We just send requests in parallel to keep the GPU saturated.
        Returns list of (text, latency_ms) tuples.
        """
        if max_workers <= 1:
            return [self.infer(img, p, max_tokens) for img, p in zip(images, prompts)]

        results: list[tuple[str, float] | None] = [None] * len(images)

        def _do(idx: int) -> tuple[int, str, float]:
            text, latency_ms = self.infer(images[idx], prompts[idx], max_tokens)
            return idx, text, latency_ms

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, i): i for i in range(len(images))}
            for future in as_completed(futures):
                idx, text, latency_ms = future.result()
                results[idx] = (text, latency_ms)

        return [(r[0], r[1]) if r else ("", 0.0) for r in results]


class MinerUClient:
    """Client that wraps MinerU's do_parse pipeline."""

    def __init__(self) -> None:
        # Patch transformers 5.x compat: find_pruneable_heads_and_indices was removed
        import transformers.pytorch_utils as _tpu
        if not hasattr(_tpu, 'find_pruneable_heads_and_indices'):
            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                import torch
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                return heads, mask.view(-1).contiguous().eq(1).nonzero().squeeze()
            _tpu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

        from mineru.cli.common import do_parse, read_fn
        from mineru.utils.enum_class import MakeMode
        self._do_parse = do_parse
        self._read_fn = read_fn
        self._make_mode = MakeMode.MM_MD

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Save image as temp file, run MinerU parse, return (markdown, latency_ms)."""
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.png"
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            image.save(str(img_path))
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            try:
                pdf_bytes = self._read_fn(img_path)
                self._do_parse(
                    output_dir=str(out_dir),
                    pdf_file_names=["input"],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["ch"],
                    backend="hybrid-auto-engine",
                    parse_method="auto",
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_orig_pdf=False,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_content_list=False,
                    f_dump_md=True,
                    f_make_md_mode=self._make_mode,
                )
                latency_ms = (time.time() - t0) * 1000
                md_files = list(Path(out_dir).rglob("*.md"))
                if md_files:
                    return md_files[0].read_text(encoding="utf-8"), latency_ms
                return "", latency_ms
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                print(f"[MinerU] Error: {e}")
                return "", latency_ms


class GLMOCRPipelineClient:
    """Full-pipeline client using glmocr SDK (PP-DocLayout-V3 + VLM)."""

    def __init__(self, vllm_port: int = 8000) -> None:
        import shutil
        from glmocr import GlmOcr

        # Copy default config and override only the port/host
        default_cfg = Path("/tmp/GLM-OCR/glmocr/config.yaml")
        self._config_dir = Path(tempfile.mkdtemp())
        config_path = self._config_dir / "config.yaml"
        shutil.copy(default_cfg, config_path)

        # Patch the config file: correct vLLM port + single worker.
        # request_timeout (60s) per region; PIPELINE_HARD_TIMEOUT (300s)
        # catches multi-region accumulation.
        text = config_path.read_text()
        text = text.replace("api_port: 8080", f"api_port: {vllm_port}")
        text = text.replace("level: INFO", "level: WARNING")
        config_path.write_text(text)

        self._config_path = str(config_path)
        self._parser = GlmOcr(config_path=self._config_path)
        self._GlmOcr = GlmOcr  # keep reference for re-init
        self._worker_thread: threading.Thread | None = None

    # Hard timeout for a single page (seconds).
    PIPELINE_HARD_TIMEOUT = 300

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Save image to temp file, run full pipeline, return (markdown, latency_ms).

        Uses original image resolution (official protocol — no resizing).
        Runs parse() in a daemon thread with PIPELINE_HARD_TIMEOUT.
        On timeout the parser is preserved — the thread is left to finish
        naturally (SDK request_timeout=60s caps each region). If still
        busy on the next call, we wait briefly then skip.
        """
        t0 = time.time()

        if self._parser is None or getattr(self._parser, '_pipeline', None) is None:
            print("[GLM-Pipeline] Parser unavailable — skipping sample")
            return "", (time.time() - t0) * 1000

        # If previous call timed out and thread is still running, wait briefly
        if self._worker_thread is not None and self._worker_thread.is_alive():
            print("[GLM-Pipeline] Previous call still running — waiting up to 60s...")
            self._worker_thread.join(timeout=60)
            if self._worker_thread.is_alive():
                print("[GLM-Pipeline] Still busy — skipping sample")
                return "", (time.time() - t0) * 1000

        img_fd = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img_path = img_fd.name
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(img_path, format="JPEG", quality=95)
        img_fd.close()

        result_holder: list[str | Exception] = []

        def _run():
            try:
                result = self._parser.parse(img_path, save_layout_visualization=False)
                result_holder.append(result.markdown_result or "")
            except Exception as e:
                result_holder.append(e)
            finally:
                try:
                    os.unlink(img_path)
                except OSError:
                    pass

        thread = threading.Thread(target=_run, daemon=True)
        self._worker_thread = thread
        thread.start()
        thread.join(timeout=self.PIPELINE_HARD_TIMEOUT)
        latency_ms = (time.time() - t0) * 1000

        if thread.is_alive():
            print(f"[GLM-Pipeline] HARD TIMEOUT ({self.PIPELINE_HARD_TIMEOUT}s) — "
                  f"leaving thread to finish (request_timeout=60s will cap it)")
            return "", latency_ms

        if result_holder:
            result = result_holder[0]
            if isinstance(result, Exception):
                print(f"[GLM-Pipeline] Error: {type(result).__name__}: {result}")
                return "", latency_ms
            return result, latency_ms

        return "", latency_ms

    def close(self):
        if self._parser:
            try:
                self._parser.close()
            except Exception:
                pass
            self._parser = None


class PaddleOCRVLPipelineClient:
    """Full-pipeline client using PaddleOCR-VL SDK (PP-DocLayout + VLM)."""

    def __init__(self) -> None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        # Register bfloat16 dtype before PaddlePaddle loads safetensors
        try:
            import ml_dtypes  # noqa: F401 — registers bfloat16 in numpy
        except ImportError:
            pass
        from paddleocr import PaddleOCRVL
        self._pipeline = PaddleOCRVL(pipeline_version="v1.5")

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Save image to temp file, run full pipeline, return (markdown, latency_ms).

        Uses original image resolution (official protocol — no resizing).
        """
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.jpg"
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            image.save(str(img_path), format="JPEG", quality=95)
            try:
                output = self._pipeline.predict(str(img_path))
                latency_ms = (time.time() - t0) * 1000
                # Extract markdown directly from result objects (no file I/O)
                md_parts = []
                for res in output:
                    if hasattr(res, 'markdown') and isinstance(res.markdown, dict):
                        texts = res.markdown.get("markdown_texts")
                        if texts:
                            md_parts.append(texts if isinstance(texts, str) else "\n".join(texts))
                    elif hasattr(res, 'save_to_markdown'):
                        # Fallback: use file I/O if direct access fails
                        md_dir = Path(tmpdir) / "md_out"
                        md_dir.mkdir(exist_ok=True)
                        res.save_to_markdown(save_path=str(md_dir))
                        md_files = list(md_dir.rglob("*.md"))
                        if md_files:
                            md_parts.append(md_files[0].read_text(encoding="utf-8"))
                    elif hasattr(res, 'rec_text'):
                        md_parts.append(str(res.rec_text))
                text = "\n".join(md_parts) if md_parts else ""
                return text, latency_ms
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                print(f"[Paddle-Pipeline] Error: {e}")
                return "", latency_ms


class DeepSeekOCR2Client:
    """Offline vLLM client for DeepSeek-OCR2 using native vLLM support.

    Optimizations applied (from SDS VQA pipeline):
    - FP8 quantization (W8A8, ~50% model size, ~37% latency reduction)
    - max_num_seqs=16, gpu_memory_utilization=0.90 (batch throughput)
    - RepetitionDetectionParams (prevent degenerate repetition)
    - Batch LLM.generate() for continuous batching
    """

    def __init__(self, model_id: str, max_tokens: int = 8192) -> None:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RepetitionDetectionParams

        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=8192,
            max_num_seqs=16,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            quantization="fp8",
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            skip_special_tokens=False,
            repetition_detection=RepetitionDetectionParams(
                max_pattern_size=40,
                min_pattern_size=5,
                min_count=3,
            ),
        )

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        """Correct EXIF orientation and convert to RGB."""
        from PIL import ExifTags

        try:
            exif = image._getexif()
            if exif is not None:
                orientation_key = next(
                    (tag for tag, name in ExifTags.TAGS.items() if name == 'Orientation'), None
                )
                if orientation_key:
                    orientation = exif.get(orientation_key, 1)
                    rotations = {3: 180, 6: 270, 8: 90}
                    if orientation in rotations:
                        image = image.rotate(rotations[orientation], expand=True)
        except Exception:
            pass
        return image.convert('RGB')

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Single-sample inference. Returns (text, latency_ms)."""
        t0 = time.time()
        image = self._prepare_image(image)
        inputs = {"prompt": f"<image>\n{prompt}", "multi_modal_data": {"image": [image]}}
        try:
            outputs = self.llm.generate([inputs], self.sampling_params)
            text = outputs[0].outputs[0].text if outputs else ""
            return text, (time.time() - t0) * 1000
        except Exception as e:
            print(f"[DeepSeek-OCR2] Error: {type(e).__name__}: {e}")
            return "", (time.time() - t0) * 1000

    def batch_infer(
        self, images: list[Image.Image], prompts: list[str],
    ) -> list[tuple[str, float]]:
        """Batch inference — pass all inputs to LLM.generate() at once.

        vLLM's continuous batching processes up to max_num_seqs concurrently,
        queuing the rest automatically.
        """
        t0 = time.time()
        inputs_list = []
        for image, prompt in zip(images, prompts):
            image = self._prepare_image(image)
            inputs_list.append({
                "prompt": f"<image>\n{prompt}",
                "multi_modal_data": {"image": [image]},
            })
        try:
            outputs = self.llm.generate(inputs_list, self.sampling_params)
            total_ms = (time.time() - t0) * 1000
            per_sample_ms = total_ms / len(inputs_list) if inputs_list else 0
            results = []
            for out in outputs:
                text = out.outputs[0].text if out.outputs else ""
                results.append((text, per_sample_ms))
            return results
        except Exception as e:
            print(f"[DeepSeek-OCR2] Batch error: {type(e).__name__}: {e}")
            return [("", 0.0)] * len(images)

    def close(self) -> None:
        """Release GPU resources."""
        if hasattr(self, 'llm'):
            del self.llm
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MinerUOptimizedClient:
    """MinerU pipeline using an external vLLM server (hybrid-http-client backend)."""

    def __init__(self, port: int = 8000) -> None:
        self._server_url = f"http://localhost:{port}"

        # Patch transformers 5.x compat
        import transformers.pytorch_utils as _tpu
        if not hasattr(_tpu, 'find_pruneable_heads_and_indices'):
            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                import torch
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                return heads, mask.view(-1).contiguous().eq(1).nonzero().squeeze()
            _tpu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

        from mineru.cli.common import do_parse, read_fn
        from mineru.utils.enum_class import MakeMode
        self._do_parse = do_parse
        self._read_fn = read_fn
        self._make_mode = MakeMode.MM_MD

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Save image as temp file, run MinerU parse with hybrid-http-client backend."""
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.png"
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            image.save(str(img_path))
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            try:
                pdf_bytes = self._read_fn(img_path)
                self._do_parse(
                    output_dir=str(out_dir),
                    pdf_file_names=["input"],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["ch"],
                    backend="hybrid-http-client",
                    parse_method="auto",
                    server_url=self._server_url,
                    f_draw_layout_bbox=False,
                    f_draw_span_bbox=False,
                    f_dump_orig_pdf=False,
                    f_dump_middle_json=False,
                    f_dump_model_output=False,
                    f_dump_content_list=False,
                    f_dump_md=True,
                    f_make_md_mode=self._make_mode,
                )
                latency_ms = (time.time() - t0) * 1000
                md_files = list(Path(out_dir).rglob("*.md"))
                if md_files:
                    return md_files[0].read_text(encoding="utf-8"), latency_ms
                return "", latency_ms
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                print(f"[MinerU-Optimized] Error: {e}")
                return "", latency_ms


class UpstageDocParserClient:
    """Client for Upstage Document Parse API (standard, enhanced, or auto mode).

    Env: UPSTAGE_API_KEY
    model: always "document-parse"
    mode: "standard", "enhanced", or "auto" (default: None = API default)

    Ref: https://console.upstage.ai/api/parse/document-parsing
    Endpoint: POST https://api.upstage.ai/v1/document-digitization
    """

    API_URL = "https://api.upstage.ai/v1/document-digitization"

    def __init__(self, model: str = "document-parse", mode: str | None = None) -> None:
        api_key = os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY environment variable required")
        self.api_key = api_key
        self.model = model
        self.mode = mode  # "standard", "enhanced", "auto", or None
        self._http = httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10))

    # Conservative max dimension — Upstage docs don't specify, but very large
    # images (e.g., 145M-pixel OmniDocBench scans) may cause timeouts or OOM.
    MAX_DIMENSION = 10000

    MAX_RETRIES = 5

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        t0 = time.time()
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        w, h = image.size
        if w > self.MAX_DIMENSION or h > self.MAX_DIMENSION:
            scale = min(self.MAX_DIMENSION / w, self.MAX_DIMENSION / h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)

        try:
            data: dict[str, str] = {
                "model": self.model,
                "output_formats": '["markdown"]',
            }
            if self.mode:
                data["mode"] = self.mode

            for attempt in range(self.MAX_RETRIES):
                buf.seek(0)
                resp = self._http.post(
                    self.API_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"document": ("image.jpg", buf, "image/jpeg")},
                    data=data,
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    print(f"[Upstage] 429 rate limited, waiting {retry_after}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                break
            else:
                print(f"[Upstage-{self.mode or 'standard'}] Max retries exceeded (429)")
                return "", (time.time() - t0) * 1000

            result = resp.json()
            content = result.get("content", {})
            if isinstance(content, dict):
                text = content.get("markdown", "") or content.get("text", "")
            else:
                text = str(content)
            return text, (time.time() - t0) * 1000
        except Exception as e:
            print(f"[Upstage-{self.mode or 'standard'}] Error: {type(e).__name__}: {e}")
            return "", (time.time() - t0) * 1000

    def batch_infer(
        self, images: list[Image.Image], prompts: list[str],
        max_tokens: int = 8192, max_workers: int = 2,
    ) -> list[tuple[str, float]]:
        """Concurrent inference via ThreadPoolExecutor."""
        results: list[tuple[str, float] | None] = [None] * len(images)

        def _do(idx: int) -> tuple[int, str, float]:
            text, latency_ms = self.infer(images[idx], prompts[idx])
            return idx, text, latency_ms

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, i): i for i in range(len(images))}
            for future in as_completed(futures):
                idx, text, latency_ms = future.result()
                results[idx] = (text, latency_ms)

        return [(r[0], r[1]) if r else ("", 0.0) for r in results]

    def close(self) -> None:
        self._http.close()


class AzureLayoutClient:
    """Client for Azure AI Document Intelligence (Layout model, REST API).

    Env: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY
    Returns markdown via outputContentFormat=markdown.

    Limits: image 50x50 ~ 10,000x10,000 px, file ≤ 500MB.
    Rate: S0 = 15 req/min → retry with backoff on 429.
    """

    API_VERSION = "2024-11-30"
    POLL_INTERVAL = 2.0
    POLL_TIMEOUT = 120
    MAX_PIXELS = 10000  # Azure DI max dimension
    MAX_RETRIES = 3

    def __init__(self) -> None:
        self.endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
        self.key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        if not self.endpoint or not self.key:
            raise ValueError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY "
                "environment variables required"
            )
        self._http = httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10))

    @staticmethod
    def _resize_if_needed(image: Image.Image) -> Image.Image:
        """Resize image if any dimension exceeds Azure's 10,000 px limit."""
        w, h = image.size
        if w > AzureLayoutClient.MAX_PIXELS or h > AzureLayoutClient.MAX_PIXELS:
            scale = min(AzureLayoutClient.MAX_PIXELS / w, AzureLayoutClient.MAX_PIXELS / h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return image

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        t0 = time.time()
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image = self._resize_if_needed(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        image_bytes = buf.getvalue()

        analyze_url = (
            f"{self.endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze"
            f"?api-version={self.API_VERSION}&outputContentFormat=markdown"
        )

        # Submit with retry on 429
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self._http.post(
                    analyze_url,
                    headers={
                        "Ocp-Apim-Subscription-Key": self.key,
                        "Content-Type": "application/octet-stream",
                    },
                    content=image_bytes,
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 15))
                    print(f"[Azure-Layout] Rate limited, retrying in {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(5)
                    continue
                print(f"[Azure-Layout] Submit error: {type(e).__name__}: {e}")
                return "", (time.time() - t0) * 1000
        else:
            print("[Azure-Layout] Max retries exceeded on submit")
            return "", (time.time() - t0) * 1000

        try:
            operation_url = resp.headers["Operation-Location"]

            poll_start = time.time()
            while time.time() - poll_start < self.POLL_TIMEOUT:
                time.sleep(self.POLL_INTERVAL)
                poll_resp = self._http.get(
                    operation_url,
                    headers={"Ocp-Apim-Subscription-Key": self.key},
                )
                if poll_resp.status_code == 429:
                    retry_after = int(poll_resp.headers.get("Retry-After", 15))
                    time.sleep(retry_after)
                    continue
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
                status = poll_data.get("status", "")
                if status == "succeeded":
                    content = poll_data.get("analyzeResult", {}).get("content", "")
                    return content, (time.time() - t0) * 1000
                elif status in ("failed", "cancelled"):
                    error = poll_data.get("error", {}).get("message", "unknown error")
                    print(f"[Azure-Layout] Analysis {status}: {error}")
                    return "", (time.time() - t0) * 1000

            print(f"[Azure-Layout] Poll timeout ({self.POLL_TIMEOUT}s)")
            return "", (time.time() - t0) * 1000
        except Exception as e:
            print(f"[Azure-Layout] Error: {type(e).__name__}: {e}")
            return "", (time.time() - t0) * 1000

    def batch_infer(
        self, images: list[Image.Image], prompts: list[str],
        max_tokens: int = 8192, max_workers: int = 4,
    ) -> list[tuple[str, float]]:
        """Concurrent inference via ThreadPoolExecutor.

        Conservative default (4 workers) due to Azure S0 rate limit (15 req/min).
        429 retry handles throttling automatically.
        """
        results: list[tuple[str, float] | None] = [None] * len(images)

        def _do(idx: int) -> tuple[int, str, float]:
            text, latency_ms = self.infer(images[idx], prompts[idx])
            return idx, text, latency_ms

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_do, i): i for i in range(len(images))}
            for future in as_completed(futures):
                idx, text, latency_ms = future.result()
                results[idx] = (text, latency_ms)

        return [(r[0], r[1]) if r else ("", 0.0) for r in results]

    def close(self) -> None:
        self._http.close()


class AllgaznieClient:
    """Adapter: wraps AllgaznieOCR pipeline for infer.py compatibility."""

    def __init__(self, model_config: ModelConfig, port: int = 8000) -> None:
        from allgaznie import AllgaznieConfig, AllgaznieOCR

        vlm_key = self._resolve_vlm_key(model_config.model_id)
        config = AllgaznieConfig(
            vlm=vlm_key,
            vlm_model_id=model_config.model_id,
            vlm_port=port,
        )
        self._pipeline = AllgaznieOCR(config=config)

    @staticmethod
    def _resolve_vlm_key(model_id: str) -> str:
        """Map model_id to VLM config key."""
        mapping = {
            "zai-org/GLM-OCR": "glm-ocr",
            "PaddlePaddle/PaddleOCR-VL": "paddleocr-vl",
            "deepseek-ai/DeepSeek-OCR-2": "deepseek-ocr2",
            "opendatalab/MinerU2.5-2509-1.2B": "mineru-vl",
        }
        return mapping.get(model_id, "glm-ocr")

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Run full pipeline on a single image. Returns (markdown, latency_ms)."""
        result = self._pipeline.parse_image(image)
        return result.markdown, result.latency_ms

    def close(self) -> None:
        self._pipeline.close()


def create_client(model_config: ModelConfig, port: int = 8000):
    """Factory to create the appropriate client for a model."""
    if model_config.backend == "allgaznie":
        return AllgaznieClient(model_config, port=port)
    elif model_config.backend == "deepseek_offline":
        return DeepSeekOCR2Client(
            model_id=model_config.model_id,
            max_tokens=MODEL_MAX_TOKENS.get(model_config.name, 8192),
        )
    elif model_config.backend == "vllm":
        return VLLMOCRClient(
            base_url=f"http://localhost:{port}/v1",
            model_name=model_config.model_id,
        )
    elif model_config.backend == "glmocr_pipeline":
        return GLMOCRPipelineClient(vllm_port=port)
    elif model_config.backend == "mineru_optimized":
        return MinerUOptimizedClient(port=port)
    elif model_config.backend == "paddleocr_pipeline":
        return PaddleOCRVLPipelineClient()
    elif model_config.backend == "upstage_api":
        mode = model_config.env.get("UPSTAGE_MODE")
        return UpstageDocParserClient(model=model_config.model_id, mode=mode)
    elif model_config.backend == "azure_api":
        return AzureLayoutClient()
    else:
        return MinerUClient()


MAX_IMAGE_PIXELS = 4096 * 4096  # ~16M pixels (relaxed to preserve detail)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64-encoded JPEG string, resizing if too large."""
    w, h = image.size
    if w * h > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    # Convert RGBA to RGB for JPEG compatibility
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
