"""PP-DocLayoutV3 layout detector with BF16 + torch.compile optimizations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from allgaznie.postprocess import (
    filter_large_images,
    spatial_reading_order,
    vectorized_containment,
    vectorized_nms,
)
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
    "formula": "formula",  # HF model merges display/inline into "formula"
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
    order: int = -1  # model-predicted reading order (-1 = unknown)
    polygon: list[list[int]] = field(default_factory=list)  # polygon points [[x,y], ...]


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

        # Tiny box pre-filter: mask logits for boxes smaller than 1 mask pixel (SDK layout_detector.py:258-275)
        self._prefilter_tiny_boxes(outputs)

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

            # Extract order_seq if available (model-predicted reading order)
            order_seq = None
            if "order_seq" in raw:
                order_seq = raw["order_seq"].cpu().numpy()

            # Extract polygon_points if available
            polygon_points = raw.get("polygon_points", [])

            n = len(scores)
            if n == 0:
                all_detections.append([])
                continue

            # Map label IDs to names and tasks
            label_names = [self.id2label.get(int(l), "unknown") for l in labels]
            tasks = [LABEL_TO_TASK.get(name, "abandon") for name in label_names]

            # Integer class IDs for NMS
            class_ids = labels.astype(np.float64)

            # Vectorized NMS with SDK params (iou_diff=0.98)
            kept = vectorized_nms(boxes, scores, class_ids)
            if not kept:
                all_detections.append([])
                continue

            kept_boxes = boxes[kept]
            kept_scores = scores[kept]
            kept_labels = labels[kept]
            kept_names = [label_names[i] for i in kept]
            kept_tasks = [tasks[i] for i in kept]
            kept_order = order_seq[kept] if order_seq is not None else None
            kept_polygons = [polygon_points[i] if i < len(polygon_points) else None for i in kept]

            # Filter large images (SDK layout_postprocess_utils.py:242-264)
            img_w, img_h = original_sizes[img_idx]
            large_keep = filter_large_images(kept_boxes, kept_labels, kept_names, (img_w, img_h))
            if not large_keep.all():
                idx = np.where(large_keep)[0]
                kept_boxes = kept_boxes[idx]
                kept_scores = kept_scores[idx]
                kept_labels = kept_labels[idx]
                kept_names = [kept_names[i] for i in idx]
                kept_tasks = [kept_tasks[i] for i in idx]
                kept_order = kept_order[idx] if kept_order is not None else None
                kept_polygons = [kept_polygons[i] for i in idx]

            # Vectorized containment with per-class mode and preserve classes
            if len(kept_boxes) > 0:
                contained = vectorized_containment(
                    kept_boxes,
                    threshold=0.8,
                    label_names=kept_names,
                    class_ids=kept_labels,
                )
                keep_mask = ~contained

                kept_boxes = kept_boxes[keep_mask]
                kept_scores = kept_scores[keep_mask]
                kept_labels = kept_labels[keep_mask]
                kept_names = [n for n, k in zip(kept_names, keep_mask) if k]
                kept_tasks = [t for t, k in zip(kept_tasks, keep_mask) if k]
                kept_order = kept_order[keep_mask] if kept_order is not None else None
                kept_polygons = [p for p, k in zip(kept_polygons, keep_mask) if k]

            # Filter abandon categories
            final_indices = [i for i, t in enumerate(kept_tasks) if t != "abandon"]
            if not final_indices:
                all_detections.append([])
                continue

            final_boxes = kept_boxes[final_indices]
            final_scores = kept_scores[final_indices]
            final_names = [kept_names[i] for i in final_indices]
            final_tasks = [kept_tasks[i] for i in final_indices]
            final_order = kept_order[final_indices] if kept_order is not None else None
            final_polygons = [kept_polygons[i] for i in final_indices]

            # Reading order: prefer model's order_seq, fallback to spatial
            if final_order is not None:
                order = np.argsort(final_order).tolist()
            else:
                order = spatial_reading_order(final_boxes)

            detections: list[Detection] = []
            for i in order:
                bx = final_boxes[i]
                # Convert polygon to list format
                poly = []
                if final_polygons[i] is not None:
                    try:
                        poly_arr = np.array(final_polygons[i])
                        if poly_arr.ndim == 2 and poly_arr.shape[1] == 2:
                            poly = [[int(p[0]), int(p[1])] for p in poly_arr]
                    except (ValueError, TypeError):
                        pass

                detections.append(Detection(
                    label=final_names[i],
                    task=final_tasks[i],
                    bbox=(int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                    score=float(final_scores[i]),
                    order=int(final_order[i]) if final_order is not None else -1,
                    polygon=poly,
                ))
            all_detections.append(detections)

        return all_detections

    def _prefilter_tiny_boxes(self, outputs: object) -> None:
        """Mask logits for boxes smaller than 1 mask pixel (SDK layout_detector.py:258-275)."""
        try:
            pred_boxes = getattr(outputs, "pred_boxes", None)
            if pred_boxes is None:
                return
            out_masks = getattr(outputs, "out_masks", None)
            if out_masks is not None:
                mask_h, mask_w = out_masks.shape[-2:]
            else:
                mask_h, mask_w = 200, 200
            min_norm_w = 1.0 / mask_w
            min_norm_h = 1.0 / mask_h
            box_wh = pred_boxes[..., 2:4]
            valid_mask = (box_wh[..., 0] > min_norm_w) & (box_wh[..., 1] > min_norm_h)
            logits = getattr(outputs, "logits", None)
            if logits is not None:
                invalid_mask = ~valid_mask
                if invalid_mask.any():
                    # Clone to avoid inplace update error under inference_mode + torch.compile
                    new_logits = logits.clone()
                    new_logits.masked_fill_(invalid_mask.unsqueeze(-1), -100.0)
                    outputs.logits = new_logits
        except Exception as e:
            logger.warning("Pre-filter failed (%s), continuing...", e)

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
