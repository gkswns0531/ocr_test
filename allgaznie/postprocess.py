"""Vectorized NMS, containment filtering, reading order, and markdown assembly.

Includes full post-processing pipeline matching the GLM-OCR SDK:
- Content cleaning (repeated punctuation, \t, long content dedup)
- Title formatting (doc_title → #, paragraph_title → ##)
- Formula formatting ($$\n...\n$$ wrapping, \tag{} merging)
- Text formatting (bullet normalization, numbered lists, newline normalization)
- Hyphenated word merging (Zipf frequency validation)
- Bullet point auto-detection (alignment-based)
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Optional

import numpy as np
from wordfreq import zipf_frequency


# =============================================================================
# NMS & Containment
# =============================================================================

# Label indices that should never be removed by containment
_PRESERVE_LABELS: set[str] = {"image", "seal", "chart"}

# Per-class containment mode (from SDK config.yaml)
# All "large" except reference(18) which is "small"
_CONTAINMENT_MODE: dict[int, str] = {
    0: "large", 1: "large", 2: "large", 3: "large", 4: "large",
    5: "large", 6: "large", 7: "large", 8: "large", 9: "large",
    10: "large", 11: "large", 12: "large", 13: "large", 14: "large",
    15: "large", 16: "large", 17: "large", 18: "small", 19: "large",
    20: "large", 21: "large", 22: "large", 23: "large", 24: "large",
}


def vectorized_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_same: float = 0.6,
    iou_diff: float = 0.98,
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


def filter_large_images(
    boxes: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    img_size: tuple[int, int],
) -> np.ndarray:
    """Remove image-class detections that cover most of the page.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        labels: (N,) integer class IDs.
        label_names: list of label name strings (same order as boxes).
        img_size: (width, height) of the original image.

    Returns:
        Boolean keep mask (N,).
    """
    n = len(boxes)
    if n <= 1:
        return np.ones(n, dtype=bool)

    img_w, img_h = img_size
    img_area = img_w * img_h
    area_thres = 0.82 if img_w > img_h else 0.93

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if label_names[i] == "image":
            x1 = max(0, boxes[i, 0])
            y1 = max(0, boxes[i, 1])
            x2 = min(img_w, boxes[i, 2])
            y2 = min(img_h, boxes[i, 3])
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > area_thres * img_area:
                keep[i] = False
    return keep


def vectorized_containment(
    boxes: np.ndarray,
    threshold: float = 0.8,
    label_names: list[str] | None = None,
    class_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask of boxes that are contained by another box.

    Supports per-class containment mode and preserve classes (image, seal, chart).

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        threshold: fraction of box_i area overlapping box_j to mark as contained.
        label_names: label name per box (for preserve logic).
        class_ids: integer class IDs per box (for per-class mode).

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

    # containment[i,j] = fraction of box_i's area overlapping box_j
    containment = np.where(areas > 0, inter / areas, 0.0)
    np.fill_diagonal(containment, 0.0)

    # Preserve classes: never mark them as contained
    preserve_mask = np.zeros(n, dtype=bool)
    if label_names is not None:
        for i, name in enumerate(label_names):
            if name in _PRESERVE_LABELS:
                preserve_mask[i] = True

    # Per-class containment mode
    if class_ids is not None:
        contained = np.zeros(n, dtype=bool)
        for i in range(n):
            if preserve_mask[i]:
                continue
            cls_mode = _CONTAINMENT_MODE.get(int(class_ids[i]), "large")
            if cls_mode == "large":
                # "large" mode: remove box_i if it's contained by ANY box_j
                if (containment[i] >= threshold).any():
                    contained[i] = True
            elif cls_mode == "small":
                # "small" mode: remove box_i if it's contained by box_j
                # AND box_i is the smaller one (class_ids[i] matches)
                if (containment[i] >= threshold).any():
                    contained[i] = True
        return contained

    # Simple mode (fallback)
    is_contained = (containment >= threshold).any(axis=1)
    is_contained[preserve_mask] = False
    return is_contained


def spatial_reading_order(boxes: np.ndarray) -> list[int]:
    """Sort detections in reading order: top->bottom row groups, left->right within rows.

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

    # Within each row, sort left->right
    result: list[int] = []
    for row in rows:
        row.sort(key=lambda i: boxes[i, 0])
        result.extend(row)
    return result


# =============================================================================
# Content Cleaning (from SDK result_formatter.py)
# =============================================================================

def _find_consecutive_repeat(s: str, min_unit_len: int = 10, min_repeats: int = 10) -> Optional[str]:
    """Find and remove consecutive repeated patterns."""
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{" + str(min_unit_len) + "," + str(max_unit_len) + r"}?)\1{" + str(min_repeats - 1) + ",}",
        re.DOTALL,
    )
    match = pattern.search(s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def _clean_repeated_content(content: str, min_len: int = 10, min_repeats: int = 10, line_threshold: int = 10) -> str:
    """Remove repeated content (both consecutive and line-level)."""
    stripped = content.strip()
    if not stripped:
        return content
    if len(stripped) > min_len * min_repeats:
        result = _find_consecutive_repeat(stripped, min_unit_len=min_len, min_repeats=min_repeats)
        if result is not None:
            return result
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total = len(lines)
    if total >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / total) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(1 for j in range(i, min(i + 3, len(lines))) if lines[j] == common)
                    if consecutive >= 3:
                        original_lines = content.split("\n")
                        non_empty_count = 0
                        for idx, orig_line in enumerate(original_lines):
                            if orig_line.strip():
                                non_empty_count += 1
                                if non_empty_count == i + 1:
                                    return "\n".join(original_lines[: idx + 1])
                        break
    return content


def clean_content(content: str) -> str:
    """Clean OCR output content (SDK _clean_content equivalent)."""
    if content is None:
        return ""
    # Remove leading/trailing literal \t
    content = re.sub(r"^(\\t)+", "", content).lstrip()
    content = re.sub(r"(\\t)+$", "", content).rstrip()
    # Remove repeated punctuation (max 3)
    content = re.sub(r"(\.)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(·)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(_)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(\\_)\1{2,}", r"\1\1\1", content)
    # Remove repeated substrings for long content
    if len(content) >= 2048:
        content = _clean_repeated_content(content)
    return content.strip()


def _clean_formula_number(number_content: str) -> str:
    """Clean formula number by removing parentheses."""
    s = number_content.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1]
    if s.startswith("\uff08") and s.endswith("\uff09"):  # （）
        return s[1:-1]
    return s


# =============================================================================
# Content Formatting (from SDK result_formatter.py)
# =============================================================================

def format_content(content: str, label: str, task: str) -> str:
    """Format a region's content based on its label and task.

    Args:
        content: Raw VLM output text.
        label: Original detection label (e.g., "doc_title", "paragraph_title").
        task: Mapped task type ("text", "table", "formula").

    Returns:
        Formatted content string.
    """
    if content is None:
        return ""
    content = clean_content(str(content))

    # Title formatting
    if label == "doc_title":
        content = re.sub(r"^#+\s*", "", content)
        content = "# " + content
    elif label == "paragraph_title":
        if content.startswith("- ") or content.startswith("* "):
            content = content[2:].lstrip()
        content = re.sub(r"^#+\s*", "", content)
        content = "## " + content.lstrip()

    # Formula formatting
    if task == "formula":
        if content.startswith("$$") and content.endswith("$$"):
            content = content[2:-2].strip()
            content = "$$\n" + content + "\n$$"
        elif content.startswith("\\[") and content.endswith("\\]"):
            content = content[2:-2].strip()
            content = "$$\n" + content + "\n$$"
        elif content.startswith("\\(") and content.endswith("\\)"):
            content = content[2:-2].strip()
            content = "$$\n" + content + "\n$$"
        else:
            content = "$$\n" + content + "\n$$"

    # Text formatting
    if task == "text":
        # Bullet points
        if content.startswith("·") or content.startswith("•") or content.startswith("* "):
            content = "- " + content[1:].lstrip()

        # Parenthesized numbers/letters: (1), (A)
        match = re.match(r"^(\(|\uff08)(\d+|[A-Za-z])(\)|\uff09)(.*)$", content)
        if match:
            _, symbol, _, rest = match.groups()
            content = f"({symbol}) {rest.lstrip()}"

        # Dot/paren numbers/letters: 1., 1), A.
        match = re.match(r"^(\d+|[A-Za-z])(\.|\)|\uff09)(.*)$", content)
        if match:
            symbol, sep, rest = match.groups()
            sep = ")" if sep == "\uff09" else sep
            content = f"{symbol}{sep} {rest.lstrip()}"

        # Replace single newlines with double newlines
        content = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content)

    return content


# =============================================================================
# Block-level Post-processing
# =============================================================================

def merge_formula_numbers(regions: list[dict]) -> list[dict]:
    """Merge formula_number into adjacent formula block using \\tag{}.

    Handles:
    1. formula_number followed by formula
    2. formula followed by formula_number
    """
    if not regions:
        return regions

    merged: list[dict] = []
    skip_indices: set[int] = set()

    for i, block in enumerate(regions):
        if i in skip_indices:
            continue

        label = block.get("label", "")

        # Case 1: formula_number followed by formula
        if label == "formula_number":
            if i + 1 < len(regions):
                next_block = regions[i + 1]
                if next_block.get("task") == "formula":
                    number_content = block.get("text", "").strip()
                    number_clean = _clean_formula_number(number_content)
                    formula_text = next_block.get("text", "")
                    merged_block = deepcopy(next_block)
                    if formula_text.endswith("\n$$"):
                        merged_block["text"] = formula_text[:-3] + f" \\tag{{{number_clean}}}\n$$"
                    merged.append(merged_block)
                    skip_indices.add(i + 1)
                    continue
            # No formula follows, skip formula_number
            continue

        # Case 2: formula followed by formula_number
        if block.get("task") == "formula":
            if i + 1 < len(regions):
                next_block = regions[i + 1]
                if next_block.get("label") == "formula_number":
                    number_content = next_block.get("text", "").strip()
                    number_clean = _clean_formula_number(number_content)
                    formula_text = block.get("text", "")
                    merged_block = deepcopy(block)
                    if formula_text.endswith("\n$$"):
                        merged_block["text"] = formula_text[:-3] + f" \\tag{{{number_clean}}}\n$$"
                    merged.append(merged_block)
                    skip_indices.add(i + 1)
                    continue
            merged.append(block)
            continue

        merged.append(block)

    return merged


def merge_hyphenated_blocks(regions: list[dict]) -> list[dict]:
    """Merge text blocks separated by hyphens if the combined word is valid."""
    if not regions:
        return regions

    merged: list[dict] = []
    skip_indices: set[int] = set()

    for i, block in enumerate(regions):
        if i in skip_indices:
            continue

        if block.get("task") != "text":
            merged.append(block)
            continue

        text = block.get("text", "")
        if not isinstance(text, str):
            merged.append(block)
            continue

        stripped = text.rstrip()
        if not stripped or not stripped.endswith("-"):
            merged.append(block)
            continue

        # Look for next text block starting with lowercase
        did_merge = False
        for j in range(i + 1, len(regions)):
            if regions[j].get("task") == "text":
                next_text = regions[j].get("text", "")
                if isinstance(next_text, str):
                    next_stripped = next_text.lstrip()
                    if next_stripped and next_stripped[0].islower():
                        words_before = stripped[:-1].split()
                        next_words = next_stripped.split()
                        if words_before and next_words:
                            merged_word = words_before[-1] + next_words[0]
                            if zipf_frequency(merged_word.lower(), "en") >= 2.5:
                                merged_block = deepcopy(block)
                                merged_block["text"] = stripped[:-1] + next_text.lstrip()
                                merged.append(merged_block)
                                skip_indices.add(j)
                                did_merge = True
                break

        if not did_merge:
            merged.append(block)

    return merged


def format_bullet_points(regions: list[dict], left_align_threshold: float = 10.0) -> list[dict]:
    """Add missing bullet points to list items between two bullet-pointed blocks."""
    if len(regions) < 3:
        return regions

    for i in range(1, len(regions) - 1):
        curr = regions[i]
        prev = regions[i - 1]
        nxt = regions[i + 1]

        if curr.get("task") != "text" or prev.get("task") != "text" or nxt.get("task") != "text":
            continue

        curr_text = curr.get("text", "")
        if curr_text.startswith("- "):
            continue

        prev_text = prev.get("text", "")
        nxt_text = nxt.get("text", "")
        if not (prev_text.startswith("- ") and nxt_text.startswith("- ")):
            continue

        # Check left alignment via bbox
        curr_bbox = curr.get("bbox")
        prev_bbox = prev.get("bbox")
        nxt_bbox = nxt.get("bbox")
        if not (curr_bbox and prev_bbox and nxt_bbox):
            continue

        curr_left = curr_bbox[0]
        prev_left = prev_bbox[0]
        nxt_left = nxt_bbox[0]
        if abs(curr_left - prev_left) <= left_align_threshold and abs(curr_left - nxt_left) <= left_align_threshold:
            curr["text"] = "- " + curr_text

    return regions


# =============================================================================
# Markdown Assembly (full pipeline)
# =============================================================================

def assemble_markdown(regions: list[dict]) -> str:
    """Combine per-region VLM outputs into a single markdown string.

    Applies the full SDK post-processing pipeline:
    1. Content cleaning + formatting per region
    2. Skip empty content
    3. Formula number merging
    4. Hyphenated text block merging
    5. Bullet point auto-detection
    6. Final join

    Each region dict must have keys: task, text, label, bbox.
    """
    # Step 1: Clean and format each region's content
    processed: list[dict] = []
    for r in regions:
        task = r.get("task", "text")
        text = r.get("text", "")
        label = r.get("label", "text")

        if task == "skip":
            processed.append({**r, "text": "[image]"})
            continue

        # Format content (clean + title/formula/text formatting)
        formatted = format_content(text, label, task)

        # Skip empty content after formatting
        if isinstance(formatted, str) and formatted.strip() == "":
            continue

        processed.append({**r, "text": formatted})

    # Step 2: Merge formula numbers
    processed = merge_formula_numbers(processed)

    # Step 3: Merge hyphenated text blocks
    processed = merge_hyphenated_blocks(processed)

    # Step 4: Bullet point auto-detection
    processed = format_bullet_points(processed)

    # Step 5: Join
    parts: list[str] = []
    for r in processed:
        text = r.get("text", "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
