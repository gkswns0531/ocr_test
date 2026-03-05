"""PP-DocLayoutV3 layout detector with BF16 + torch.compile optimizations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from allgaznie.postprocess import spatial_reading_order, vectorized_containment, vectorized_nms
from allgaznie.preprocess import cpu_decode_and_resize, gpu_decode_and_resize

logger = logging.getLogger(__name__)

# 25 PP-DocLayoutV3 categories → task mapping
LABEL_TO_TASK: dict[str, str] = {
    "abstract": "text",
    "algorithm": "text",
    "content": "text",
    "doc_title": "text",
    "figure_title": "text",
    "formula_number": "text",
    "paragraph_title": "text",
    "reference_content": "text",
    "seal": "text",
    "text": "text",
    "vertical_text": "text",
    "vision_footnote": "text",
    "table": "table",
    "display_formula": "formula",
    "inline_formula": "formula",
    "chart": "skip",
    "image": "skip",
    "aside_text": "abandon",
    "footer": "abandon",
    "footer_image": "abandon",
    "footnote": "abandon",
    "header": "abandon",
    "header_image": "abandon",
    "number": "abandon",
    "reference": "abandon",
}


@dataclass
class Detection:
    """A single layout detection result."""

    label: str
    task: str  # "text" | "table" | "formula" | "skip" | "abandon"
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    score: float


class LayoutDetector:
    """PP-DocLayoutV3 with BF16, torch.compile, and GPU preprocessing."""

    def __init__(
        self,
        model_id: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        device: str = "cuda",
        threshold: float = 0.3,
    ) -> None:
        from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessorFast

        self.device = device
        self.threshold = threshold

        self.processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(model_id)
        proc_size = getattr(self.processor, "size", {})
        self.target_h = proc_size.get("height", 800) if isinstance(proc_size, dict) else 800
        self.target_w = proc_size.get("width", 800) if isinstance(proc_size, dict) else 800

        model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_id)
        model.eval()

        # BF16 on Ampere+ GPUs (OPT-007)
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            model = model.bfloat16()
            logger.info("Layout model cast to bfloat16")

        model = model.to(device)
        self.model_dtype = next(model.parameters()).dtype

        # torch.compile for graph-level fusion (OPT-007)
        if device.startswith("cuda"):
            try:
                model = torch.compile(model)
                logger.info("torch.compile enabled for layout model")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode: %s", e)

        self.model = model
        self.id2label: dict[int, str] = model.config.id2label

    def detect(
        self,
        images: list | None = None,
        image_paths: list[str] | None = None,
        use_gpu_decode: bool = True,
    ) -> list[list[Detection]]:
        """Run layout detection on a batch of images.

        Args:
            images: List of PIL Images (used if image_paths is None or GPU decode disabled).
            image_paths: File paths for fast GPU decode (JPEG via nvJPEG).
            use_gpu_decode: Whether to attempt GPU JPEG decode.

        Returns:
            Per-image list of Detection objects in reading order (abandon filtered out).
        """
        target_size = (self.target_h, self.target_w)

        # Decide decode path
        can_gpu_decode = (
            use_gpu_decode
            and image_paths is not None
            and self.device.startswith("cuda")
        )

        if can_gpu_decode:
            pixel_values, original_sizes = gpu_decode_and_resize(
                image_paths, target_size, self.device, self.model_dtype
            )
        else:
            if images is None:
                raise ValueError("Either images or image_paths must be provided")
            pixel_values, original_sizes = cpu_decode_and_resize(
                images, target_size, self.device, self.model_dtype
            )

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)

        # BF16 → FP32 for post_process compatibility (OPT-010)
        self._outputs_to_float32(outputs)

        # Post-process: map boxes to original image coordinates
        target_sizes = torch.tensor(
            [(h, w) for w, h in original_sizes], device=self.device
        )
        raw_results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )

        # Convert to Detection lists with NMS + containment + reading order
        all_detections: list[list[Detection]] = []
        for img_idx, raw in enumerate(raw_results):
            scores = raw["scores"].cpu().numpy()
            labels = raw["labels"].cpu().numpy()
            boxes = raw["boxes"].cpu().numpy()  # (N, 4) float [x1,y1,x2,y2]

            n = len(scores)
            if n == 0:
                all_detections.append([])
                continue

            # Map label IDs to names and tasks
            label_names = [self.id2label.get(int(l), "unknown") for l in labels]
            tasks = [LABEL_TO_TASK.get(name, "abandon") for name in label_names]

            # Integer class IDs for NMS
            class_ids = labels.astype(np.float64)

            # Vectorized NMS (OPT-006)
            kept = vectorized_nms(boxes, scores, class_ids)
            if not kept:
                all_detections.append([])
                continue

            kept_boxes = boxes[kept]
            kept_scores = scores[kept]
            kept_names = [label_names[i] for i in kept]
            kept_tasks = [tasks[i] for i in kept]

            # Vectorized containment (OPT-006)
            contained = vectorized_containment(kept_boxes)
            keep_mask = ~contained

            kept_boxes = kept_boxes[keep_mask]
            kept_scores = kept_scores[keep_mask]
            kept_names = [n for n, k in zip(kept_names, keep_mask) if k]
            kept_tasks = [t for t, k in zip(kept_tasks, keep_mask) if k]

            # Filter abandon categories
            final_indices = [i for i, t in enumerate(kept_tasks) if t != "abandon"]
            if not final_indices:
                all_detections.append([])
                continue

            final_boxes = kept_boxes[final_indices]
            final_scores = kept_scores[final_indices]
            final_names = [kept_names[i] for i in final_indices]
            final_tasks = [kept_tasks[i] for i in final_indices]

            # Reading order
            order = spatial_reading_order(final_boxes)

            detections: list[Detection] = []
            for i in order:
                bx = final_boxes[i]
                detections.append(Detection(
                    label=final_names[i],
                    task=final_tasks[i],
                    bbox=(int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                    score=float(final_scores[i]),
                ))
            all_detections.append(detections)

        return all_detections

    @staticmethod
    def _outputs_to_float32(outputs: object) -> None:
        """Cast model outputs to float32 in-place for HF post_process compatibility."""
        for attr in ("logits", "pred_boxes", "pred_masks", "out_masks"):
            t = getattr(outputs, attr, None)
            if t is not None and t.dtype != torch.float32:
                setattr(outputs, attr, t.float())

    def close(self) -> None:
        """Release GPU memory."""
        del self.model
        del self.processor
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
