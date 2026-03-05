"""Vectorized NMS, containment filtering, reading order, and markdown assembly.

Extracted from sdk_optimizations.patch — numpy-broadcasting versions of the
O(n²) Python loops used in the original glmocr SDK.
"""

from __future__ import annotations

import numpy as np


def vectorized_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_same: float = 0.6,
    iou_diff: float = 0.95,
) -> list[int]:
    """Greedy NMS with class-aware IoU thresholds.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] in pixel coordinates.
        scores: (N,) confidence scores.
        classes: (N,) integer class IDs.
        iou_same/iou_diff: IoU thresholds for same-class / cross-class suppression.

    Returns:
        List of kept indices into the original arrays.
    """
    n = len(boxes)
    if n == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Pairwise intersection: (N,1) op (1,N) → (N,N)
    xi1 = np.maximum(x1, boxes[:, 0])
    yi1 = np.maximum(y1, boxes[:, 1])
    xi2 = np.minimum(x2, boxes[:, 2])
    yi2 = np.minimum(y2, boxes[:, 3])
    inter = np.maximum(0, xi2 - xi1 + 1) * np.maximum(0, yi2 - yi1 + 1)

    union = areas + areas.T - inter
    iou_matrix = np.where(union > 0, inter / union, 0.0)

    same_class = classes[:, None] == classes[None, :]
    threshold_matrix = np.where(same_class, iou_same, iou_diff)

    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(n, dtype=bool)
    selected: list[int] = []

    for idx in order:
        if suppressed[idx]:
            continue
        selected.append(int(idx))
        suppress_mask = iou_matrix[idx] >= threshold_matrix[idx]
        suppressed |= suppress_mask
        suppressed[idx] = False  # keep self

    return selected


def vectorized_containment(
    boxes: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """Return boolean mask of boxes that are ≥threshold contained by another box.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        threshold: fraction of box_i area overlapping box_j to mark as contained.

    Returns:
        (N,) boolean — True for boxes that should be removed.
    """
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=bool)

    x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    areas = (x2 - x1) * (y2 - y1)

    xi1 = np.maximum(x1, boxes[:, 0])
    yi1 = np.maximum(y1, boxes[:, 1])
    xi2 = np.minimum(x2, boxes[:, 2])
    yi2 = np.minimum(y2, boxes[:, 3])
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)

    containment = np.where(areas > 0, inter / areas, 0.0)
    np.fill_diagonal(containment, 0.0)

    return (containment >= threshold).any(axis=1)


def spatial_reading_order(boxes: np.ndarray) -> list[int]:
    """Sort detections in reading order: top→bottom row groups, left→right within rows.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].

    Returns:
        Sorted index list.
    """
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Row clustering by y-center
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    median_h = max(np.median(heights), 1.0)

    # Sort by y first, then group into rows
    y_order = np.argsort(y_centers)
    rows: list[list[int]] = []
    current_row: list[int] = [int(y_order[0])]
    current_y = y_centers[y_order[0]]

    for i in range(1, n):
        idx = int(y_order[i])
        if abs(y_centers[idx] - current_y) < median_h * 0.5:
            current_row.append(idx)
        else:
            rows.append(current_row)
            current_row = [idx]
            current_y = y_centers[idx]
    rows.append(current_row)

    # Within each row, sort left→right
    result: list[int] = []
    for row in rows:
        row.sort(key=lambda i: boxes[i, 0])
        result.extend(row)
    return result


def assemble_markdown(regions: list[dict]) -> str:
    """Combine per-region VLM outputs into a single markdown string.

    Each region dict must have keys: task, text.
    - text → appended as-is
    - table → wrapped as HTML block
    - formula → wrapped in LaTeX display math
    - skip → "[image]" placeholder
    """
    parts: list[str] = []
    for r in regions:
        task = r["task"]
        text = r.get("text", "").strip()
        if not text and task != "skip":
            continue
        if task == "text":
            parts.append(text)
        elif task == "table":
            parts.append(text)
        elif task == "formula":
            parts.append(text)
        elif task == "skip":
            parts.append("[image]")
    return "\n\n".join(parts)
