"""Recompute TEDS/TEDS-structure scores for existing PubTabNet and TEDS_TEST results.

The original runs failed to compute scores due to a broken import
(metrics.py shadowed the OmniDocBench metrics package). Predictions are
already stored in result/checkpoint files — this script reloads GT from
prepared_datasets/ and recomputes the TEDS metrics in-place.

Usage:
    python3 recompute_teds_metrics.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from metrics import compute_teds

RESULTS_DIR = Path("/home/ubuntu/ocr_test/results")
PREPARED_DIR = Path("/home/ubuntu/ocr_test/prepared_datasets")


def _load_gt_map(benchmark_key: str) -> dict[str, str]:
    """Load sample_id → ground_truth mapping from prepared metadata."""
    meta_path = PREPARED_DIR / benchmark_key / "metadata.jsonl"
    gt_map: dict[str, str] = {}
    with open(meta_path) as f:
        for line in f:
            rec = json.loads(line)
            gt_map[rec["sample_id"]] = rec["ground_truth"]
    return gt_map


def _convert_paddle_table_to_html(text: str) -> str:
    """Convert PaddleOCR-VL table format (<fcel>, <ecel>, <lcel>, <ucel>, <nl>) to HTML.

    Markers:
    - <fcel>: filled cell (has content)
    - <ecel>: empty cell
    - <lcel>: left-merge (colspan with the cell to the left)
    - <ucel>: up-merge (rowspan with the cell above)
    - <nl>: new row
    """
    rows = text.split("<nl>")
    # First pass: parse into a grid of (type, content) tuples
    grid: list[list[tuple[str, str]]] = []
    for row in rows:
        row_cells: list[tuple[str, str]] = []
        tokens = re.split(r"(<fcel>|<ecel>|<lcel>|<ucel>)", row)
        marker = None
        for token in tokens:
            if token in ("<fcel>", "<ecel>", "<lcel>", "<ucel>"):
                marker = token
            elif marker is not None:
                row_cells.append((marker, token.strip()))
                marker = None
        if row_cells:
            grid.append(row_cells)

    if not grid:
        return text

    # Second pass: build HTML with colspan/rowspan
    num_cols = max(len(r) for r in grid)
    rowspan_left: list[int] = [0] * num_cols

    html_rows = []
    for row_cells in grid:
        cells_html: list[str] = []
        col_idx = 0
        cell_idx = 0
        while cell_idx < len(row_cells):
            while col_idx < num_cols and rowspan_left[col_idx] > 0:
                rowspan_left[col_idx] -= 1
                col_idx += 1

            marker, content = row_cells[cell_idx]
            cell_idx += 1

            if marker == "<ucel>":
                col_idx += 1
                continue
            if marker == "<lcel>":
                col_idx += 1
                continue

            colspan = 1
            while cell_idx < len(row_cells) and row_cells[cell_idx][0] == "<lcel>":
                colspan += 1
                cell_idx += 1

            rowspan = 1
            cur_row_idx = grid.index(row_cells)
            for below_row in grid[cur_row_idx + 1:]:
                if col_idx < len(below_row) and below_row[col_idx][0] == "<ucel>":
                    rowspan += 1
                else:
                    break

            if rowspan > 1:
                for c in range(col_idx, min(col_idx + colspan, num_cols)):
                    rowspan_left[c] = rowspan - 1

            attrs = ""
            if colspan > 1:
                attrs += f' colspan="{colspan}"'
            if rowspan > 1:
                attrs += f' rowspan="{rowspan}"'
            cells_html.append(f"<td{attrs}>{content}</td>")
            col_idx += colspan

        if cells_html:
            html_rows.append("<tr>" + "".join(cells_html) + "</tr>")

    return "<table>" + "".join(html_rows) + "</table>"


def _convert_markdown_table_to_html(text: str) -> str:
    """Convert Markdown pipe-table to HTML <table>."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    data_lines = [l for l in lines if not re.match(r"^\|[\s\-:|]+\|$", l)]
    if not data_lines:
        return text
    html_rows = []
    for i, line in enumerate(data_lines):
        line = line.strip("|")
        cells = [c.strip() for c in line.split("|")]
        tag = "td"
        row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
        html_rows.append(f"<tr>{row}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


def recompute_file(result_path: Path, gt_map: dict[str, str]) -> None:
    """Recompute TEDS scores for a single result/checkpoint file."""
    with open(result_path) as f:
        data = json.load(f)

    results = data.get("per_sample_results", [])
    if not results:
        print(f"  No samples in {result_path.name}, skipping")
        return

    recomputed = 0
    for r in results:
        sid = r["sample_id"]
        gt = gt_map.get(sid)
        if gt is None:
            print(f"  WARNING: No GT for {sid}")
            continue

        pred = r.get("prediction", "")
        # Apply format conversion as needed
        if "<fcel>" in pred:
            pred_html = _convert_paddle_table_to_html(pred)
        elif "|" in pred and "<table" not in pred.lower():
            pred_html = _convert_markdown_table_to_html(pred)
        else:
            pred_html = pred
        gt_str = gt if isinstance(gt, str) else str(gt)

        teds = compute_teds(pred_html, gt_str)
        teds_s = compute_teds(pred_html, gt_str, structure_only=True)

        r["scores"] = {"teds": teds, "teds_structure": teds_s}
        r["error"] = ""  # Clear the old import error
        recomputed += 1

    # Re-aggregate metrics
    all_scores = [r["scores"] for r in results if r["scores"]]
    aggregate: dict[str, float] = {}
    if all_scores:
        keys = set()
        for s in all_scores:
            keys.update(s.keys())
        for key in sorted(keys):
            vals = [s[key] for s in all_scores if key in s and isinstance(s[key], (int, float))]
            if vals:
                aggregate[f"mean_{key}"] = sum(vals) / len(vals)

    error_count = sum(1 for r in results if r.get("error"))
    aggregate["error_rate"] = error_count / len(results)

    # Update top-level metrics
    if "metrics" in data:
        data["metrics"] = aggregate
    if "n_completed" in data:
        data["n_completed"] = len(results)

    with open(result_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Recomputed {recomputed} samples in {result_path.name}")
    for k, v in aggregate.items():
        print(f"    {k}: {v:.4f}")


def main() -> None:
    for benchmark_key in ("pubtabnet", "teds_test"):
        print(f"\n=== {benchmark_key} ===")
        meta_path = PREPARED_DIR / benchmark_key / "metadata.jsonl"
        if not meta_path.exists():
            print(f"  No prepared data at {meta_path}, skipping")
            continue
        gt_map = _load_gt_map(benchmark_key)
        print(f"  Loaded {len(gt_map)} GT entries")

        # Find all result/checkpoint files for this benchmark
        patterns = [f"*_{benchmark_key}.json", f"*_{benchmark_key}_checkpoint.json"]
        for pattern in patterns:
            for result_path in sorted(RESULTS_DIR.glob(pattern)):
                print(f"\n  Processing {result_path.name}...")
                recompute_file(result_path, gt_map)


if __name__ == "__main__":
    main()
