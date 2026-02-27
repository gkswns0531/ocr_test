"""Benchmark: layout post-processing old (scalar loops) vs new (vectorized).

Compares NMS, check_containment, and full apply_layout_postprocess.
Also validates that both produce identical results.
"""

import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Set


# ============================================================================
# OLD implementation (original scalar Python loops)
# ============================================================================

def _old_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou_value = inter_area / float(box1_area + box2_area - inter_area)
    return iou_value


def _old_nms(boxes, iou_same=0.6, iou_diff=0.95):
    scores = boxes[:, 1]
    indices = list(np.argsort(scores)[::-1])
    selected_boxes = []
    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = current_box[2:]
        selected_boxes.append(current)
        indices = indices[1:]
        filtered_indices = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = _old_iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    return selected_boxes


def _old_is_contained(box1, box2):
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height
    iou = intersect_area / box1_area if box1_area > 0 else 0
    return iou >= 0.8


def _old_check_containment(boxes, preserve_indices=None, category_index=None, mode=None):
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if preserve_indices is not None and boxes[i][0] in preserve_indices:
                continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if _old_is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if _old_is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if _old_is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


def _old_apply_layout_postprocess(
    raw_results, id2label, img_sizes,
    layout_nms=True, layout_unclip_ratio=None, layout_merge_bboxes_mode=None,
):
    all_labels = list(id2label.values())
    paddle_format_results = []
    for img_idx, result in enumerate(raw_results):
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()
        order_seq = result["order_seq"].cpu().numpy()
        polygon_points = result.get("polygon_points", [])
        img_size = img_sizes[img_idx]
        boxes_with_order = []
        for i in range(len(scores)):
            cls_id = int(labels[i])
            score = float(scores[i])
            x1, y1, x2, y2 = boxes[i]
            order = int(order_seq[i])
            boxes_with_order.append([cls_id, score, x1, y1, x2, y2, order])
        if len(boxes_with_order) == 0:
            paddle_format_results.append([])
            continue
        boxes_array = np.array(boxes_with_order)
        if layout_nms:
            selected_indices = _old_nms(boxes_array[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes_array = boxes_array[selected_indices]
        filter_large_image = True
        if filter_large_image and len(boxes_array) > 1:
            area_thres = 0.82 if img_size[0] > img_size[1] else 0.93
            image_index = all_labels.index("image") if "image" in all_labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            for box in boxes_array:
                label_index, score, xmin, ymin, xmax, ymax = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                else:
                    filtered_boxes.append(box)
            if len(filtered_boxes) > 0:
                boxes_array = np.array(filtered_boxes)
        if layout_merge_bboxes_mode:
            preserve_labels = ["image", "seal", "chart"]
            preserve_indices = set()
            for label in preserve_labels:
                if label in all_labels:
                    preserve_indices.add(all_labels.index(label))
            if isinstance(layout_merge_bboxes_mode, str):
                if layout_merge_bboxes_mode != "union":
                    contains_other, contained_by_other = _old_check_containment(
                        boxes_array[:, :6], preserve_indices
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes_array = boxes_array[contained_by_other == 0]
                    elif layout_merge_bboxes_mode == "small":
                        boxes_array = boxes_array[
                            (contains_other == 0) | (contained_by_other == 1)
                        ]
        if len(boxes_array) == 0:
            paddle_format_results.append([])
            continue
        sorted_idx = np.argsort(boxes_array[:, 6])
        boxes_array = boxes_array[sorted_idx]
        img_width, img_height = img_size
        image_results = []
        for i, box_data in enumerate(boxes_array):
            cls_id = int(box_data[0])
            score = float(box_data[1])
            x1, y1, x2, y2 = box_data[2:6]
            order = int(box_data[6]) if box_data[6] > 0 else None
            label_name = id2label.get(cls_id, f"class_{cls_id}")
            x1 = max(0, min(float(x1), img_width))
            y1 = max(0, min(float(y1), img_height))
            x2 = max(0, min(float(x2), img_width))
            y2 = max(0, min(float(y2), img_height))
            if x1 >= x2 or y1 >= y2:
                continue
            poly = None
            if len(polygon_points) > 0:
                for orig_idx in range(len(boxes)):
                    if np.allclose(boxes[orig_idx], box_data[2:6], atol=1.0):
                        if orig_idx < len(polygon_points):
                            candidate_poly = polygon_points[orig_idx]
                            if candidate_poly is not None:
                                poly = candidate_poly.astype(np.float32)
                        break
            if poly is None:
                poly = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            else:
                poly[:, 0] = np.clip(poly[:, 0], 0, img_width)
                poly[:, 1] = np.clip(poly[:, 1], 0, img_height)
            image_results.append({
                "cls_id": cls_id, "label": label_name, "score": score,
                "coordinate": [int(x1), int(y1), int(x2), int(y2)],
                "order": order, "polygon_points": poly,
            })
        paddle_format_results.append(image_results)
    return paddle_format_results


# ============================================================================
# NEW implementation (import from SDK)
# ============================================================================

import sys
sys.path.insert(0, "/root/glm-ocr-sdk")
from glmocr.utils.layout_postprocess_utils import (
    nms as new_nms,
    check_containment as new_check_containment,
    apply_layout_postprocess as new_apply_layout_postprocess,
)


# ============================================================================
# Test data generation
# ============================================================================

def generate_realistic_detections(
    n_images: int = 64,
    boxes_per_image: int = 120,
    n_classes: int = 12,
    img_w: int = 2480,
    img_h: int = 3508,
    seed: int = 42,
) -> tuple:
    """Generate realistic layout detection results mimicking PP-DocLayoutV3 output."""
    rng = np.random.RandomState(seed)
    id2label = {i: name for i, name in enumerate([
        "text", "title", "figure", "table", "caption",
        "header", "footer", "page_number", "image", "seal", "chart", "formula",
    ][:n_classes])}

    raw_results = []
    img_sizes = []
    for _ in range(n_images):
        n = rng.randint(boxes_per_image - 30, boxes_per_image + 30)
        # Generate boxes with some overlap to exercise NMS/containment
        x1 = rng.uniform(0, img_w * 0.7, n)
        y1 = rng.uniform(0, img_h * 0.7, n)
        w = rng.uniform(50, img_w * 0.4, n)
        h = rng.uniform(30, img_h * 0.3, n)
        x2 = np.minimum(x1 + w, img_w)
        y2 = np.minimum(y1 + h, img_h)
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # Add some contained boxes (small boxes inside larger ones)
        n_contained = n // 10
        for ci in range(n_contained):
            parent = rng.randint(0, n)
            px1, py1, px2, py2 = boxes[parent]
            pw, ph = px2 - px1, py2 - py1
            if pw > 20 and ph > 20:
                cx1 = px1 + rng.uniform(0, pw * 0.1)
                cy1 = py1 + rng.uniform(0, ph * 0.1)
                cx2 = px2 - rng.uniform(0, pw * 0.1)
                cy2 = py2 - rng.uniform(0, ph * 0.1)
                target = rng.randint(0, n)
                boxes[target] = [cx1, cy1, cx2, cy2]

        labels = rng.randint(0, n_classes, n).astype(np.int64)
        scores = rng.uniform(0.3, 0.99, n).astype(np.float32)
        order_seq = np.arange(n, dtype=np.int64)

        # Generate polygon points (4-corner polygons with slight perturbation)
        polygon_points = []
        for i in range(n):
            bx1, by1, bx2, by2 = boxes[i]
            jitter = rng.uniform(-2, 2, (4, 2)).astype(np.float32)
            poly = np.array([
                [bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]
            ], dtype=np.float32) + jitter
            polygon_points.append(poly)

        raw_results.append({
            "scores": torch.from_numpy(scores),
            "labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(boxes),
            "order_seq": torch.from_numpy(order_seq),
            "polygon_points": polygon_points,
        })
        img_sizes.append((img_w, img_h))

    return raw_results, id2label, img_sizes


# ============================================================================
# Benchmarks
# ============================================================================

def bench_nms(n_runs: int = 5) -> None:
    """Benchmark NMS: old scalar vs new vectorized."""
    print("=" * 70)
    print("BENCHMARK: NMS (old scalar vs new vectorized)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    for n_boxes in [50, 100, 150, 200]:
        # Generate boxes: [cls_id, score, x1, y1, x2, y2]
        cls_ids = rng.randint(0, 10, n_boxes).astype(float)
        scores = rng.uniform(0.3, 0.99, n_boxes)
        x1 = rng.uniform(0, 2000, n_boxes)
        y1 = rng.uniform(0, 3000, n_boxes)
        x2 = x1 + rng.uniform(50, 500, n_boxes)
        y2 = y1 + rng.uniform(30, 400, n_boxes)
        boxes = np.column_stack([cls_ids, scores, x1, y1, x2, y2])

        # Warmup
        _old_nms(boxes, 0.6, 0.98)
        new_nms(boxes, 0.6, 0.98)

        # Old
        times_old = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            old_result = _old_nms(boxes, 0.6, 0.98)
            times_old.append(time.perf_counter() - t0)

        # New
        times_new = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            new_result = new_nms(boxes, 0.6, 0.98)
            times_new.append(time.perf_counter() - t0)

        old_ms = np.median(times_old) * 1000
        new_ms = np.median(times_new) * 1000
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")
        match = sorted(old_result) == sorted(new_result)

        print(f"  n={n_boxes:>3d}: old={old_ms:>8.2f}ms  new={new_ms:>8.2f}ms  "
              f"speedup={speedup:>5.1f}x  match={match}")


def bench_containment(n_runs: int = 5) -> None:
    """Benchmark containment checking: old scalar vs new vectorized."""
    print()
    print("=" * 70)
    print("BENCHMARK: check_containment (old scalar vs new vectorized)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    preserve = {8, 9, 10}  # image, seal, chart

    for n_boxes in [50, 100, 150, 200]:
        cls_ids = rng.randint(0, 12, n_boxes).astype(float)
        scores = rng.uniform(0.3, 0.99, n_boxes)
        x1 = rng.uniform(0, 2000, n_boxes)
        y1 = rng.uniform(0, 3000, n_boxes)
        x2 = x1 + rng.uniform(50, 500, n_boxes)
        y2 = y1 + rng.uniform(30, 400, n_boxes)
        # Add contained boxes
        for ci in range(n_boxes // 5):
            parent = rng.randint(0, n_boxes)
            px1, py1, px2, py2 = x1[parent], y1[parent], x2[parent], y2[parent]
            target = rng.randint(0, n_boxes)
            x1[target] = px1 + 1
            y1[target] = py1 + 1
            x2[target] = px2 - 1
            y2[target] = py2 - 1

        boxes = np.column_stack([cls_ids, scores, x1, y1, x2, y2])

        # Warmup
        _old_check_containment(boxes, preserve)
        new_check_containment(boxes, preserve)

        # Old
        times_old = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            old_co, old_cb = _old_check_containment(boxes, preserve)
            times_old.append(time.perf_counter() - t0)

        # New
        times_new = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            new_co, new_cb = new_check_containment(boxes, preserve)
            times_new.append(time.perf_counter() - t0)

        old_ms = np.median(times_old) * 1000
        new_ms = np.median(times_new) * 1000
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")
        match_co = np.array_equal(old_co, new_co)
        match_cb = np.array_equal(old_cb, new_cb)

        print(f"  n={n_boxes:>3d}: old={old_ms:>8.2f}ms  new={new_ms:>8.2f}ms  "
              f"speedup={speedup:>5.1f}x  match_contains={match_co}  match_contained={match_cb}")


def bench_full_postprocess(n_runs: int = 3) -> None:
    """Benchmark full apply_layout_postprocess: old vs new."""
    print()
    print("=" * 70)
    print("BENCHMARK: apply_layout_postprocess (full pipeline, 64 images)")
    print("=" * 70)

    raw_results, id2label, img_sizes = generate_realistic_detections(
        n_images=64, boxes_per_image=120
    )

    # Warmup
    _old_apply_layout_postprocess(raw_results[:2], id2label, img_sizes[:2],
                                  layout_nms=True, layout_merge_bboxes_mode="large")
    new_apply_layout_postprocess(raw_results[:2], id2label, img_sizes[:2],
                                 layout_nms=True, layout_merge_bboxes_mode="large")

    for merge_mode in [None, "large"]:
        label = f"merge={merge_mode or 'none'}"
        print(f"\n  [{label}]")

        # Old
        times_old = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            old_results = _old_apply_layout_postprocess(
                raw_results, id2label, img_sizes,
                layout_nms=True, layout_merge_bboxes_mode=merge_mode,
            )
            times_old.append(time.perf_counter() - t0)

        # New
        times_new = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            new_results = new_apply_layout_postprocess(
                raw_results, id2label, img_sizes,
                layout_nms=True, layout_merge_bboxes_mode=merge_mode,
            )
            times_new.append(time.perf_counter() - t0)

        old_ms = np.median(times_old) * 1000
        new_ms = np.median(times_new) * 1000
        speedup = old_ms / new_ms if new_ms > 0 else float("inf")

        # Validate: same number of detections per image, same labels/coordinates
        n_match = 0
        n_total = 0
        for img_i in range(len(old_results)):
            old_dets = old_results[img_i]
            new_dets = new_results[img_i]
            n_total += 1
            if len(old_dets) == len(new_dets):
                coords_match = all(
                    o["coordinate"] == n["coordinate"] and o["cls_id"] == n["cls_id"]
                    for o, n in zip(old_dets, new_dets)
                )
                if coords_match:
                    n_match += 1

        print(f"    old:     {old_ms:>8.1f}ms  ({old_ms/64:>5.1f}ms/img)")
        print(f"    new:     {new_ms:>8.1f}ms  ({new_ms/64:>5.1f}ms/img)")
        print(f"    speedup: {speedup:>5.1f}x")
        print(f"    match:   {n_match}/{n_total} images identical")


if __name__ == "__main__":
    bench_nms()
    bench_containment()
    bench_full_postprocess()
