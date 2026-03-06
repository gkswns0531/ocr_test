"""Allgaznie OCR SDK — unified 2-stage hybrid pipeline.

Layout Detection (PP-DocLayoutV3) → Region Crop → Per-Region VLM → Markdown Assembly.

Usage:
    from allgaznie import AllgaznieOCR, AllgaznieConfig

    config = AllgaznieConfig(vlm="glm-ocr")
    ocr = AllgaznieOCR(config=config)
    result = ocr.parse("/path/to/document.jpg")
    print(result.markdown)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image

from allgaznie.layout import Detection, LayoutDetector
from allgaznie.postprocess import assemble_markdown
from allgaznie.preprocess import crop_regions
from allgaznie.vlm import VLMClient

VLM_MODEL_IDS: dict[str, str] = {
    "glm-ocr": "zai-org/GLM-OCR",
    "paddleocr-vl": "PaddlePaddle/PaddleOCR-VL",
    "deepseek-ocr2": "deepseek-ai/DeepSeek-OCR-2",
    "mineru-vl": "opendatalab/MinerU2.5-2509-1.2B",
}

VLM_DISPLAY_NAMES: dict[str, str] = {
    "glm-ocr": "GLM-OCR",
    "paddleocr-vl": "PaddleOCR-VL",
    "deepseek-ocr2": "DeepSeek-OCR2",
    "mineru-vl": "MinerU-VL",
}


@dataclass
class AllgaznieConfig:
    """Configuration for the Allgaznie OCR pipeline."""

    vlm: str = "glm-ocr"  # "glm-ocr" | "paddleocr-vl" | "deepseek-ocr2"
    vlm_model_id: str = ""  # Auto-resolved from vlm key if empty
    vlm_port: int = 8000
    vlm_max_tokens: int = 8192
    vlm_max_workers: int = 16
    vlm_temperature: float = 0.0
    vlm_top_p: float = 1.0
    vlm_top_k: int = 1
    vlm_repetition_penalty: float = 1.0
    vlm_min_pixels: int = 0  # 0 = use model-specific default
    vlm_max_pixels: int = 0  # 0 = use model-specific default
    layout_model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors"
    layout_threshold: float = 0.3
    layout_device: str = "cuda"
    use_gpu_decode: bool = True


@dataclass
class PageResult:
    """Result of parsing a single page/image."""

    markdown: str
    regions: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


class AllgaznieOCR:
    """Unified OCR pipeline: layout detection → region crop → VLM → markdown.

    Args:
        config: Pipeline configuration. If None, uses default (GLM-OCR).
    """

    def __init__(self, config: AllgaznieConfig | None = None) -> None:
        if config is None:
            config = AllgaznieConfig()
        self.config = config

        # Resolve VLM model ID
        vlm_model_id = config.vlm_model_id
        if not vlm_model_id:
            vlm_model_id = VLM_MODEL_IDS.get(config.vlm)
            if not vlm_model_id:
                raise ValueError(
                    f"Unknown VLM key: {config.vlm}. "
                    f"Available: {list(VLM_MODEL_IDS.keys())}"
                )

        display_name = VLM_DISPLAY_NAMES.get(config.vlm, "GLM-OCR")

        # Initialize layout detector
        self.layout = LayoutDetector(
            model_id=config.layout_model_id,
            device=config.layout_device,
            threshold=config.layout_threshold,
        )

        # Initialize VLM client (HTTP — server must be running externally)
        self.vlm = VLMClient(
            base_url=f"http://localhost:{config.vlm_port}/v1",
            model_name=vlm_model_id,
            model_display_name=display_name,
            max_tokens=config.vlm_max_tokens,
            max_workers=config.vlm_max_workers,
            temperature=config.vlm_temperature,
            top_p=config.vlm_top_p,
            top_k=config.vlm_top_k,
            repetition_penalty=config.vlm_repetition_penalty,
            min_pixels=config.vlm_min_pixels,
            max_pixels=config.vlm_max_pixels,
        )

    def parse(self, image_path: str) -> PageResult:
        """Parse a document image from file path.

        Args:
            image_path: Path to the image file.

        Returns:
            PageResult with markdown text and region details.
        """
        t0 = time.time()

        image = Image.open(image_path).convert("RGB")

        # Layout detection (GPU decode from path)
        detections = self.layout.detect(
            images=[image],
            image_paths=[image_path],
            use_gpu_decode=self.config.use_gpu_decode,
        )[0]

        return self._process_detections(image, detections, image_path, t0)

    def parse_image(self, image: Image.Image) -> PageResult:
        """Parse a document from a PIL Image (no file path available).

        Args:
            image: PIL Image to parse.

        Returns:
            PageResult with markdown text and region details.
        """
        t0 = time.time()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Layout detection (CPU path — no file for GPU decode)
        detections = self.layout.detect(
            images=[image],
            use_gpu_decode=False,
        )[0]

        return self._process_detections(image, detections, None, t0)

    def _process_detections(
        self,
        image: Image.Image,
        detections: list[Detection],
        image_path: str | None,
        t0: float,
    ) -> PageResult:
        """Shared logic: crop → VLM → assemble."""
        # Separate VLM-processable regions from skip regions
        vlm_dets: list[Detection] = []
        skip_dets: list[Detection] = []
        for d in detections:
            if d.task in ("text", "table", "formula"):
                vlm_dets.append(d)
            elif d.task == "skip":
                skip_dets.append(d)

        # Crop regions for VLM (skip empty crops to avoid 1x1 placeholder errors)
        if vlm_dets:
            raw_crops = crop_regions(image, vlm_dets, image_path)
            crops = []
            tasks = []
            valid_indices: list[int] = []
            for i, crop in enumerate(raw_crops):
                if crop is not None:
                    crops.append(crop)
                    tasks.append(vlm_dets[i].task)
                    valid_indices.append(i)
            if crops:
                raw_texts = self.vlm.infer_regions(crops, tasks)
            else:
                raw_texts = []
            # Map back: fill empty string for skipped crops
            texts = [""] * len(vlm_dets)
            for vi, txt in zip(valid_indices, raw_texts):
                texts[vi] = txt
        else:
            texts = []

        # Build regions list in original detection order (reading order from layout)
        regions: list[dict] = []
        vlm_idx = 0
        for d in detections:
            if d.task in ("text", "table", "formula"):
                region = {
                    "label": d.label,
                    "task": d.task,
                    "bbox": d.bbox,
                    "score": d.score,
                    "text": texts[vlm_idx] if vlm_idx < len(texts) else "",
                }
                vlm_idx += 1
            elif d.task == "skip":
                region = {
                    "label": d.label,
                    "task": d.task,
                    "bbox": d.bbox,
                    "score": d.score,
                    "text": "",
                }
            else:
                continue
            regions.append(region)

        markdown = assemble_markdown(regions)
        latency_ms = (time.time() - t0) * 1000

        return PageResult(markdown=markdown, regions=regions, latency_ms=latency_ms)

    def close(self) -> None:
        """Release layout model GPU memory."""
        self.layout.close()
