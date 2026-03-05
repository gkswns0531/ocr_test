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
}

# VLM model_id → display name mapping for prompt lookup
_MODEL_ID_TO_DISPLAY: dict[str, str] = {
    "zai-org/GLM-OCR": "GLM-OCR",
    "PaddlePaddle/PaddleOCR-VL": "PaddleOCR-VL",
    "deepseek-ai/DeepSeek-OCR-2": "DeepSeek-OCR2",
}


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


class VLMClient:
    """Concurrent HTTP client for vLLM-served VLM models."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        model_display_name: str | None = None,
        max_tokens: int = 8192,
        max_workers: int = 16,
        temperature: float = 0.1,
        top_p: float = 0.1,
        top_k: int = 1,
        repetition_penalty: float = 1.1,
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
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Resolve display name for prompt selection
        if model_display_name:
            self.display_name = model_display_name
        else:
            self.display_name = _MODEL_ID_TO_DISPLAY.get(model_name, "GLM-OCR")

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                extra_body={
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty,
                },
            )
            return resp.choices[0].message.content or ""
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
