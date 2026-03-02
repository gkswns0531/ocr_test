#!/usr/bin/env python3
"""Update HTML report v2: add visual summary + caption text per case."""
from __future__ import annotations

import html as html_mod
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset

DATA_DIR = Path("output_dl")
EMB_DIR = DATA_DIR / "embeddings_8b"
HTML_PATH = DATA_DIR / "true_visual_case_report.html"


def load_data():
    qa = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-QA",
        split="test", cache_dir="data",
    )
    ann = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-annotations",
        split="test", cache_dir="data",
    )
    idx_to_info: dict[int, tuple[str, int]] = {}
    for row in ann:
        for offset, ci in enumerate(row["page_indices"]):
            idx_to_info[ci] = (row["file_id"], offset)

    with open("/tmp/true_visual_40_detail.json") as f:
        tv_cases = json.load(f)

    # Load caption texts per page
    page_captions: dict[str, list[dict]] = {}
    with open(DATA_DIR / "region_captions.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            cap = rec.get("caption", "").strip()
            pid = rec["page_id"]
            page_captions.setdefault(pid, []).append({
                "caption": cap,
                "region_index": rec.get("region_index", ""),
                "label": rec.get("label", ""),
            })

    return qa, idx_to_info, tv_cases, page_captions


def get_rank(sim_row, pids, expected):
    sorted_idx = np.argsort(sim_row)[::-1]
    seen = set()
    rank = 0
    for idx in sorted_idx:
        pid = pids[idx]
        if pid not in seen:
            seen.add(pid)
            rank += 1
            if pid in expected:
                return rank
    return rank + 1


def build_summary_html(cases: list[dict]) -> str:
    """Build the entire visual summary section."""
    n = len(cases)

    # R@10 stats for 8B model
    pt_hits = sum(1 for c in cases if c["pt_rank"] <= 10)
    rg_hits = sum(1 for c in cases if c["rg_rank"] <= 10)
    cap_hits = sum(1 for c in cases if c["cap_rank"] <= 10)

    # Patterns
    all_hit = sum(1 for c in cases if c["pt_rank"] <= 10 and c["rg_rank"] <= 10 and c["cap_rank"] <= 10)
    rg_only = sum(1 for c in cases if c["pt_rank"] > 10 and c["rg_rank"] <= 10 and c["cap_rank"] > 10)
    cap_only = sum(1 for c in cases if c["pt_rank"] > 10 and c["rg_rank"] > 10 and c["cap_rank"] <= 10)
    none_hit = sum(1 for c in cases if c["pt_rank"] > 10 and c["rg_rank"] > 10 and c["cap_rank"] > 10)

    # R@K data for the 3x3 table
    recall_data = {
        "Qwen3-VL-2B": {"pure_text": [12.5, 22.5, 35.0, 42.5], "region": [47.5, 70.0, 72.5, 82.5], "caption": [17.5, 27.5, 32.5, 40.0]},
        "Qwen3-VL-8B": {"pure_text": [17.5, 30.0, 35.0, 40.0], "region": [62.5, 75.0, 75.0, 77.5], "caption": [25.0, 37.5, 40.0, 40.0]},
        "BGE-M3": {"pure_text": [25.0, 40.0, 50.0, 50.0], "region": [32.5, 37.5, 47.5, 47.5], "caption": [7.5, 20.0, 27.5, 30.0]},
    }
    ks = [1, 5, 10, 20]

    def bar(val: float, max_val: float = 100, color: str = "#3498db") -> str:
        w = val / max_val * 100
        return (
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="flex:1;background:#eee;border-radius:4px;height:22px;overflow:hidden;">'
            f'<div style="width:{w}%;height:100%;background:{color};border-radius:4px;'
            f'transition:width 0.3s;"></div></div>'
            f'<span style="font-weight:600;min-width:48px;text-align:right;">{val:.1f}%</span></div>'
        )

    def cell_color(val: float) -> str:
        if val >= 70:
            return "#27ae60"
        if val >= 50:
            return "#f39c12"
        if val >= 30:
            return "#e67e22"
        return "#e74c3c"

    # Build the visual summary
    s = []
    s.append('<div class="summary">')
    s.append('<h2>Summary</h2>')
    s.append('<p>SDS-KoPub-VDR-Benchmark 600개 QA 중 <code>type=visual</code> 161건을 정성 전수 검토하여 true_visual 58건 분류 → '
             'GT 페이지에 image/chart region이 존재하는 <strong>40건</strong>에 대한 상세 분석.</p>')
    s.append('<p><strong>Corpus</strong>: 40,781 pages | <strong>Regions</strong>: 21,052 | <strong>Caption Model</strong>: 4o</p>')

    # --- Visual bar chart: R@10 comparison (8B model) ---
    s.append('<h3 style="margin-top:25px;">Qwen3-VL-8B R@10 — 3가지 검색 방식 비교</h3>')
    s.append('<div style="max-width:500px;margin:10px 0;">')
    for label, val, color in [
        ("Region (multimodal)", 75.0, "#27ae60"),
        ("Caption (4o text)", 40.0, "#9b59b6"),
        ("Pure Text (OCR)", 35.0, "#3498db"),
    ]:
        s.append(f'<div style="margin:8px 0;"><div style="font-size:0.85em;color:#555;margin-bottom:3px;">{label}</div>')
        s.append(bar(val, 100, color))
        s.append('</div>')
    s.append('</div>')

    # --- Heatmap table: R@K × Model × Method ---
    s.append('<h3 style="margin-top:30px;">Recall@K — 3 Embedding Models × 3 Corpus Types</h3>')
    s.append('<div style="overflow-x:auto;">')
    s.append('<table style="border-collapse:collapse;width:100%;font-size:0.9em;">')
    s.append('<tr><th rowspan="2" style="padding:8px;border:1px solid #ddd;background:#16213e;color:white;">R@K</th>')
    for model in ["Qwen3-VL-2B", "Qwen3-VL-8B", "BGE-M3"]:
        s.append(f'<th colspan="3" style="padding:8px;border:1px solid #ddd;background:#16213e;color:white;">{model}</th>')
    s.append('</tr><tr>')
    for _ in range(3):
        for method in ["text", "region", "caption"]:
            s.append(f'<th style="padding:6px;border:1px solid #ddd;background:#2c3e6b;color:white;font-size:0.85em;">{method}</th>')
    s.append('</tr>')

    for ki, k in enumerate(ks):
        s.append(f'<tr><td style="padding:6px 8px;border:1px solid #ddd;font-weight:600;background:#f8f9fa;">R@{k}</td>')
        for model in ["Qwen3-VL-2B", "Qwen3-VL-8B", "BGE-M3"]:
            for method in ["pure_text", "region", "caption"]:
                v = recall_data[model][method][ki]
                bg = cell_color(v)
                opacity = 0.15 + (v / 100) * 0.85
                s.append(f'<td style="padding:6px;border:1px solid #ddd;text-align:center;'
                         f'background:rgba({int(bg[1:3],16)},{int(bg[3:5],16)},{int(bg[5:7],16)},{opacity:.2f});'
                         f'font-weight:{"700" if v >= 70 else "400"};">{v:.1f}%</td>')
        s.append('</tr>')
    s.append('</table></div>')

    # --- Per-case hit/miss matrix ---
    s.append('<h3 style="margin-top:30px;">케이스별 R@10 Hit/Miss 패턴 (Qwen3-VL-8B)</h3>')
    s.append('<div style="display:flex;gap:15px;align-items:center;margin:8px 0;font-size:0.85em;">')
    s.append('<span><span style="display:inline-block;width:14px;height:14px;background:#27ae60;border-radius:3px;vertical-align:middle;"></span> Hit (≤10)</span>')
    s.append('<span><span style="display:inline-block;width:14px;height:14px;background:#f39c12;border-radius:3px;vertical-align:middle;"></span> Near (11-50)</span>')
    s.append('<span><span style="display:inline-block;width:14px;height:14px;background:#e74c3c;border-radius:3px;vertical-align:middle;"></span> Miss (&gt;50)</span>')
    s.append('</div>')
    s.append('<div style="overflow-x:auto;">')
    s.append('<table style="border-collapse:collapse;font-size:0.78em;">')
    s.append('<tr><th style="padding:4px 6px;border:1px solid #ddd;background:#f8f9fa;"></th>')
    for c in cases:
        s.append(f'<th style="padding:4px 3px;border:1px solid #ddd;background:#f8f9fa;font-size:0.85em;">'
                 f'<a href="#case-{c["qi"]}" style="text-decoration:none;color:#16213e;">#{c["qi"]}</a></th>')
    s.append('</tr>')

    for label, key in [("text", "pt_rank"), ("region", "rg_rank"), ("caption", "cap_rank")]:
        s.append(f'<tr><td style="padding:4px 6px;border:1px solid #ddd;font-weight:600;background:#f8f9fa;">{label}</td>')
        for c in cases:
            r = c[key]
            if r <= 10:
                bg = "#27ae60"
                txt = str(r)
            elif r <= 50:
                bg = "#f39c12"
                txt = str(r)
            else:
                bg = "#e74c3c"
                txt = "×"
            s.append(f'<td style="padding:3px;border:1px solid #ddd;text-align:center;background:{bg};color:white;'
                     f'font-weight:600;font-size:0.9em;min-width:22px;" title="Rank {r:,}">{txt}</td>')
        s.append('</tr>')
    s.append('</table></div>')

    # --- Pattern summary ---
    s.append('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0;">')
    patterns = [
        (f"{all_hit}", "3방식 모두 Hit", "#27ae60"),
        (f"{rg_only}", "Region만 Hit", "#2ecc71"),
        (f"{cap_only}", "Caption만 Hit", "#9b59b6"),
        (f"{none_hit}", "모두 Miss", "#e74c3c"),
    ]
    for val, label, color in patterns:
        s.append(f'<div style="text-align:center;padding:15px;border-radius:8px;background:{color}15;border:2px solid {color};">'
                 f'<div style="font-size:2em;font-weight:700;color:{color};">{val}</div>'
                 f'<div style="font-size:0.82em;color:#555;margin-top:4px;">{label}</div></div>')
    s.append('</div>')

    s.append('</div>')
    return "\n".join(s)


def build_caption_section(page_captions: dict[str, list[dict]], gt_pids: list[str]) -> str:
    """Build a collapsible caption section for a case."""
    all_caps: list[tuple[str, str, str]] = []  # (region_index, label, caption)
    for pid in gt_pids:
        if pid in page_captions:
            for c in page_captions[pid]:
                cap = c["caption"]
                if cap and cap != "해당 없음":
                    all_caps.append((c["region_index"], c["label"], cap))

    if not all_caps:
        return ""

    lines = []
    lines.append(f'<details><summary>4o Caption ({len(all_caps)}건)</summary>')
    for ri, label, cap in all_caps:
        cap_escaped = html_mod.escape(cap)
        # Truncate very long captions for display
        if len(cap_escaped) > 800:
            cap_escaped = cap_escaped[:800] + "..."
        lines.append(f'<div class="caption-box"><strong>[{label} #{ri}]</strong> {cap_escaped}</div>')
    lines.append('</details>')
    return "\n".join(lines)


def main() -> None:
    print("Loading data...")
    qa, idx_to_info, tv_cases_raw, page_captions = load_data()
    tv_qi = [c["qi"] for c in tv_cases_raw]

    # Load 8B embeddings for ranks
    queries = np.load(str(EMB_DIR / "queries.npy"))
    pt_emb = np.load(str(EMB_DIR / "corpus_pure_text.npy"))
    with open(EMB_DIR / "pure_text_page_ids.json") as f:
        pt_pids = json.load(f)
    rg_emb = np.load(str(EMB_DIR / "corpus_regions.npy"))
    rg_meta = []
    with open(EMB_DIR / "region_metadata.jsonl") as f:
        for line in f:
            rg_meta.append(json.loads(line))
    rg_pids = [r["page_id"] for r in rg_meta]
    cap_emb = np.load(str(EMB_DIR / "corpus_caption.npy"))
    with open(EMB_DIR / "caption_page_ids.json") as f:
        cap_pids = json.load(f)

    pt_sim = queries @ pt_emb.T
    rg_sim = queries @ rg_emb.T
    cap_sim = queries @ cap_emb.T

    # Build case data
    cases = []
    for qi in tv_qi:
        gt = qa[qi]["ground_truth"]
        expected = set()
        for gi in gt:
            if gi in idx_to_info:
                fid, off = idx_to_info[gi]
                expected.add(f"{fid}_{off+1}")
        cases.append({
            "qi": qi,
            "gt_pids": sorted(expected),
            "pt_rank": get_rank(pt_sim[qi], pt_pids, expected),
            "rg_rank": get_rank(rg_sim[qi], rg_pids, expected),
            "cap_rank": get_rank(cap_sim[qi], cap_pids, expected),
        })

    # Read HTML
    print("Reading HTML...")
    content = HTML_PATH.read_text(encoding="utf-8")

    # 1. Replace summary section
    print("Replacing summary section...")
    summary_start = content.find('<div class="summary">')
    summary_end = content.find('</div>\n<div class="toc">')
    if summary_start == -1 or summary_end == -1:
        # Try alternate end marker
        summary_end = content.find('</div>\n<div class="note">')
        if summary_end == -1:
            print("ERROR: Could not find summary section boundaries!")
            return

    new_summary = build_summary_html(cases)
    content = content[:summary_start] + new_summary + content[summary_end + len('</div>'):]

    # 2. Add caption text section to each case (before Region Similarities)
    print("Adding caption sections to cases...")
    for c in cases:
        qi = c["qi"]
        cap_html = build_caption_section(page_captions, c["gt_pids"])
        if not cap_html:
            continue

        # Find the Region Similarities details section for this case
        case_pos = content.find(f'id="case-{qi}"')
        if case_pos == -1:
            continue

        # Find the first <details> after images section (which is Region Similarities)
        images_end = content.find('</div>\n<details>', case_pos)
        if images_end == -1:
            # Try without newline
            images_end = content.find('</div>\n<details><summary>Region', case_pos)
            if images_end == -1:
                continue

        insert_pos = images_end + len('</div>')
        content = content[:insert_pos] + "\n" + cap_html + "\n" + content[insert_pos:]

    # Write updated HTML
    HTML_PATH.write_text(content, encoding="utf-8")
    print(f"Done! HTML size: {len(content)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
