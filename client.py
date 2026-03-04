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

# Default max_tokens per model (override the global 4096 default)
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

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> tuple[str, float]:
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
        max_tokens: int = 4096,
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
        mineru_path = "/home/ubuntu/junghoon/miner_test/MinerU"
        if mineru_path not in sys.path:
            sys.path.insert(0, mineru_path)
        self._parse_fn = None

    def _get_parse_fn(self):
        if self._parse_fn is None:
            from demo.demo import do_parse  # type: ignore
            self._parse_fn = do_parse
        return self._parse_fn

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> tuple[str, float]:
        """Save image as temp file, run MinerU parse, return (markdown, latency_ms)."""
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.png"
            image.save(str(img_path))
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            try:
                parse_fn = self._get_parse_fn()
                parse_fn(
                    str(img_path),
                    str(out_dir),
                    backend="pipeline",
                    parse_method="auto",
                )
                latency_ms = (time.time() - t0) * 1000
                # Find the markdown output
                md_files = list(out_dir.rglob("*.md"))
                if md_files:
                    return md_files[0].read_text(encoding="utf-8"), latency_ms
                # Fall back to any text output
                json_files = list(out_dir.rglob("*content_list.json"))
                if json_files:
                    return json_files[0].read_text(encoding="utf-8"), latency_ms
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
        default_cfg = Path("/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml")
        self._config_dir = Path(tempfile.mkdtemp())
        config_path = self._config_dir / "config.yaml"
        shutil.copy(default_cfg, config_path)

        # Patch the config file: correct vLLM port + single worker.
        # request_timeout (60s) per region; PIPELINE_HARD_TIMEOUT (300s)
        # catches multi-region accumulation.
        text = config_path.read_text()
        text = text.replace("api_port: 8080", f"api_port: {vllm_port}")
        text = text.replace("level: INFO", "level: WARNING")
        text = text.replace("max_workers: 32", "max_workers: 1")
        text = text.replace("request_timeout: 120", "request_timeout: 60")
        text = text.replace("retry_max_attempts: 2", "retry_max_attempts: 1")
        config_path.write_text(text)

        self._config_path = str(config_path)
        self._parser = GlmOcr(config_path=self._config_path)
        self._GlmOcr = GlmOcr  # keep reference for re-init
        self._worker_thread: threading.Thread | None = None

    # Hard timeout for a single page (seconds).
    PIPELINE_HARD_TIMEOUT = 300

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> tuple[str, float]:
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

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 4096) -> tuple[str, float]:
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
    """Offline vLLM client for DeepSeek-OCR2 using official inference pipeline.

    Uses LLM.generate() (not OpenAI API) with:
    - Custom model class (DeepseekOCR2ForCausalLM)
    - Custom image processor with dynamic tiling (1024x1024 global + 768x768 local crops)
    - NoRepeatNGramLogitsProcessor (ngram_size=40, window_size=90)
    - skip_special_tokens=False
    - VLLM_USE_V1=0
    """

    def __init__(self, model_id: str, max_tokens: int = 8192) -> None:
        os.environ['VLLM_USE_V1'] = '0'

        from deepseek_vllm.deepseek_ocr2 import DeepseekOCR2ForCausalLM
        from deepseek_vllm.ngram_norepeat import NoRepeatNGramLogitsProcessor
        from deepseek_vllm.image_process import DeepseekOCR2Processor
        from transformers import AutoTokenizer
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm import LLM, SamplingParams

        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

        self.llm = LLM(
            model=model_id,
            hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=True,
            max_model_len=8192,
            swap_space=0,
            max_num_seqs=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
        )

        logits_processors = [NoRepeatNGramLogitsProcessor(
            ngram_size=40, window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        )]

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.processor = DeepseekOCR2Processor(tokenizer=tokenizer)

    def infer(self, image: Image.Image, prompt: str, max_tokens: int = 8192) -> tuple[str, float]:
        """Run inference using offline LLM.generate(). Returns (text, latency_ms)."""
        from PIL import ExifTags

        t0 = time.time()

        # Correct EXIF orientation (official protocol)
        try:
            exif = image._getexif()
            if exif is not None:
                orientation_key = None
                for tag, value in ExifTags.TAGS.items():
                    if value == 'Orientation':
                        orientation_key = tag
                        break
                if orientation_key:
                    orientation = exif.get(orientation_key, 1)
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        except Exception:
            pass

        image = image.convert('RGB')

        # Build full prompt with <image> placeholder
        full_prompt = f"<image>\n{prompt}"

        # Process image with official tiling processor
        processed = self.processor.tokenize_with_images(
            images=[image],
            prompt=full_prompt,
            bos=True, eos=True, cropping=True,
        )

        inputs = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": processed},
        }

        try:
            outputs = self.llm.generate([inputs], self.sampling_params)
            latency_ms = (time.time() - t0) * 1000
            text = outputs[0].outputs[0].text if outputs else ""
            return text, latency_ms
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            print(f"[DeepSeek-OCR2] Inference error ({latency_ms:.0f}ms): {type(e).__name__}: {e}")
            return "", latency_ms

    def close(self) -> None:
        """Release GPU resources."""
        if hasattr(self, 'llm'):
            del self.llm
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_client(model_config: ModelConfig, port: int = 8000):
    """Factory to create the appropriate client for a model."""
    if model_config.backend == "deepseek_offline":
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
    elif model_config.backend == "paddleocr_pipeline":
        return PaddleOCRVLPipelineClient()
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
