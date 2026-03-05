"""VLM HTTP client for vLLM-served OCR models with concurrent per-region inference."""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI
from PIL import Image


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


def _image_to_base64_jpeg(image: Image.Image) -> str:
    """Single-pass JPEG encode + base64 (OPT-008)."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
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
        max_tokens: int = 4096,
        max_workers: int = 16,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_workers = max_workers

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
        b64 = _image_to_base64_jpeg(crop)

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
                temperature=0.0,
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
