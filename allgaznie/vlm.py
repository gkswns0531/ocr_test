"""VLM HTTP client for vLLM-served OCR models with concurrent per-region inference."""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI
from PIL import Image

from allgaznie.preprocess import smart_resize


# Per-region prompts by model family
REGION_PROMPTS: dict[str, dict[str, str]] = {
    "GLM-OCR": {
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    },
    "PaddleOCR-VL": {
        "text": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    },
    "DeepSeek-OCR2": {
        "text": "Free OCR.",
        "table": "Free OCR.",
        "formula": "Free OCR.",
    },
    "MinerU-VL": {
        "text": "\nText Recognition:",
        "table": "\nTable Recognition:",
        "formula": "\nFormula Recognition:",
    },
}

# VLM model_id → display name mapping for prompt lookup
_MODEL_ID_TO_DISPLAY: dict[str, str] = {
    "zai-org/GLM-OCR": "GLM-OCR",
    "PaddlePaddle/PaddleOCR-VL": "PaddleOCR-VL",
    "deepseek-ai/DeepSeek-OCR-2": "DeepSeek-OCR2",
    "opendatalab/MinerU2.5-2509-1.2B": "MinerU-VL",
}

_MINERU_END_TOKEN = "<|im_end|>"

# Model-specific pixel limits from official preprocessor_config.json
_MODEL_PIXEL_DEFAULTS: dict[str, tuple[int, int]] = {
    "GLM-OCR": (12544, 9633792),       # preprocessor: shortest_edge=12544, longest_edge=9633792
    "PaddleOCR-VL": (147384, 2822400), # preprocessor: min_pixels=147384, max_pixels=2822400
    "MinerU-VL": (3136, 1605632),      # preprocessor: min_pixels=3136, max_pixels=1605632
    "DeepSeek-OCR2": (12544, 1003520), # DeepSeek-VL-V2 arch; vLLM server re-processes to 1024×1024 tiles
}
_DEFAULT_PIXELS = (12544, 1003520)  # Qwen2VL default fallback


def _image_to_base64_jpeg(
    image: Image.Image,
    min_pixels: int = 12544,
    max_pixels: int = 1003520,
) -> str:
    """Smart resize + single-pass JPEG encode + base64.

    Applies SDK-compatible smart_resize to ensure image dimensions are
    aligned to patch size (28) and within pixel limits before encoding.
    """
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    w, h = image.size
    new_h, new_w = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
    if (new_h, new_w) != (h, w):
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _try_otsl_to_html(text: str) -> str:
    """Try to convert MinerU OTSL table output to HTML. Returns original on failure."""
    if not text or text.startswith("<table"):
        return text
    try:
        from mineru_vl_utils.post_process import convert_otsl_to_html
        result = convert_otsl_to_html(text)
        # Only use conversion if it produced valid HTML
        if result.startswith("<table"):
            return result
    except Exception:
        pass
    return text


class VLMClient:
    """Concurrent HTTP client for vLLM-served VLM models."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        model_display_name: str | None = None,
        max_tokens: int = 8192,
        max_workers: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        min_pixels: int = 12544,
        max_pixels: int = 1003520,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        # Resolve display name for prompt selection
        if model_display_name:
            self.display_name = model_display_name
        else:
            self.display_name = _MODEL_ID_TO_DISPLAY.get(model_name, "GLM-OCR")

        # Resolve pixel limits: 0 = use model-specific defaults from preprocessor_config
        default_min, default_max = _MODEL_PIXEL_DEFAULTS.get(self.display_name, _DEFAULT_PIXELS)
        self.min_pixels = min_pixels if min_pixels > 0 else default_min
        self.max_pixels = max_pixels if max_pixels > 0 else default_max

        self.is_mineru = self.display_name == "MinerU-VL"
        self.prompts = REGION_PROMPTS.get(self.display_name, REGION_PROMPTS["GLM-OCR"])

        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
        )

    def _infer_one(self, crop: Image.Image, task: str) -> str:
        """Send a single region to the VLM server. Returns recognized text."""
        prompt = self.prompts.get(task, self.prompts["text"])
        b64 = _image_to_base64_jpeg(crop, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        messages = []
        if self.is_mineru:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        })
        # All models use greedy decoding (temperature=0.0).
        # Penalties modify logits BEFORE argmax, so they affect output even with greedy.
        temperature = self.temperature
        top_p = self.top_p
        extra: dict = {
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

        # Model-specific penalties from official configs
        # presence_penalty/frequency_penalty are OpenAI API top-level params.
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        if self.is_mineru:
            extra["skip_special_tokens"] = False
            extra["vllm_xargs"] = {"no_repeat_ngram_size": 100}
            presence_penalty = 1.0
            frequency_penalty = 0.005 if task == "table" else 0.05
        elif self.display_name == "GLM-OCR":
            extra["repetition_penalty"] = 1.1
        elif self.display_name == "DeepSeek-OCR2":
            # Official: no_repeat_ngram_size=35 (eval mode), modeling_deepseekocr2.py:936
            extra["vllm_xargs"] = {"no_repeat_ngram_size": 35}

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                extra_body=extra,
            )
            text = resp.choices[0].message.content or ""

            # MinerU-VL post-processing
            if self.is_mineru:
                if text.endswith(_MINERU_END_TOKEN):
                    text = text[: -len(_MINERU_END_TOKEN)]
                if task == "table":
                    text = _try_otsl_to_html(text)

            return text
        except Exception as e:
            print(f"[VLMClient] Error for task={task}: {type(e).__name__}: {e}")
            return ""

    def infer_regions(self, crops: list[Image.Image], tasks: list[str]) -> list[str]:
        """Run concurrent VLM inference on a batch of cropped regions.

        Args:
            crops: List of cropped PIL images.
            tasks: Corresponding task types ("text", "table", "formula").

        Returns:
            List of recognized text strings (order preserved).
        """
        n = len(crops)
        if n == 0:
            return []

        results: list[str | None] = [None] * n

        def _do(idx: int) -> tuple[int, str]:
            text = self._infer_one(crops[idx], tasks[idx])
            return idx, text

        with ThreadPoolExecutor(max_workers=min(self.max_workers, n)) as pool:
            for idx, text in pool.map(lambda i: _do(i), range(n)):
                results[idx] = text

        return [r if r is not None else "" for r in results]
