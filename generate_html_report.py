#!/usr/bin/env python3
"""Generate HTML case report for 40 true_visual queries.

Reads corrected embeddings, computes ranks, generates an interactive HTML report
with embedded page/crop images (base64).
"""
from __future__ import annotations

import base64
import html
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

DATA_DIR = Path("output_dl")
OCR_RESULTS_FILE = DATA_DIR / "ocr_results.jsonl"
EMBEDDINGS_8B_DIR = DATA_DIR / "embeddings_8b"
CAPTIONS_FILE = DATA_DIR / "region_captions.jsonl"
TRUE_VISUAL_FILE = Path("/tmp/true_visual_40_detail.json")
OUTPUT_HTML = DATA_DIR / "true_visual_case_report.html"


def img_to_base64(img: Image.Image, max_dim: int = 800, quality: int = 60) -> str:
    """Convert PIL Image to base64 JPEG string."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def load_page_images(page_ids: set[str]) -> dict[str, Image.Image]:
    """Load page images from Samsung SDS corpus."""
    ds = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-corpus",
        split="test", cache_dir="data",
    )
    ann = load_dataset(
        "SamsungSDS-Research/SDS-KoPub-VDR-Benchmark", "SDS-KoPub-annotations",
        split="test", cache_dir="data",
    )

    # Build corpus_index -> page_id mapping
    idx_to_pid: dict[int, str] = {}
    for row in ann:
        for offset, ci in enumerate(row["page_indices"]):
            idx_to_pid[ci] = f"{row['file_id']}_{offset+1}"

    # Find which corpus indices we need
    needed_indices: dict[int, str] = {}
    for ci, pid in idx_to_pid.items():
        if pid in page_ids:
            needed_indices[ci] = pid

    result: dict[str, Image.Image] = {}
    for ci, pid in needed_indices.items():
        try:
            row = ds[ci]
            img = row["image"]
            if img is not None:
                result[pid] = img.convert("RGB")
        except Exception:
            pass

    return result


def main() -> None:
    print("Loading data...")

    # Load QA dataset
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

    # Load OCR results indexed by page_id
    ocr_by_pid: dict[str, dict] = {}
    with open(OCR_RESULTS_FILE) as f:
        for line in f:
            rec = json.loads(line)
            ocr_by_pid[rec["page_id"]] = rec

    # Load captions indexed by crop_path
    captions_by_path: dict[str, str] = {}
    if CAPTIONS_FILE.exists():
        with open(CAPTIONS_FILE) as f:
            for line in f:
                rec = json.loads(line)
                cp = rec.get("crop_path", "")
                cap = rec.get("gpt5_caption", "")
                if cp and cap:
                    captions_by_path[cp] = cap

    # Load true_visual 40 cases
    with open(TRUE_VISUAL_FILE) as f:
        tv_cases = json.load(f)
    tv_qi = [c["qi"] for c in tv_cases]
    print(f"True visual cases: {len(tv_qi)}")

    # Load 8B embeddings
    queries = np.load(str(EMBEDDINGS_8B_DIR / "queries.npy"))

    # pure_text
    pt_emb = np.load(str(EMBEDDINGS_8B_DIR / "corpus_pure_text.npy"))
    with open(EMBEDDINGS_8B_DIR / "pure_text_page_ids.json") as f:
        pt_pids = json.load(f)

    # region
    rg_emb = np.load(str(EMBEDDINGS_8B_DIR / "corpus_regions.npy"))
    rg_meta = []
    with open(EMBEDDINGS_8B_DIR / "region_metadata.jsonl") as f:
        for line in f:
            rg_meta.append(json.loads(line))
    rg_pids = [r["page_id"] for r in rg_meta]

    print(f"Embeddings: queries={queries.shape}, pt={pt_emb.shape}, rg={rg_emb.shape}")

    # Compute similarity matrices
    pt_sim = queries @ pt_emb.T  # (600, 40781)
    rg_sim = queries @ rg_emb.T  # (600, N_regions)

    def get_rank(sim_row: np.ndarray, pids: list[str], expected_pids: set[str], pid_level: bool = True) -> int:
        """Get the rank of the first expected page_id (1-indexed)."""
        sorted_idx = np.argsort(sim_row)[::-1]
        if pid_level:
            seen: set[str] = set()
            rank = 0
            for idx in sorted_idx:
                pid = pids[idx]
                if pid not in seen:
                    seen.add(pid)
                    rank += 1
                    if pid in expected_pids:
                        return rank
            return rank + 1
        else:
            for rank, idx in enumerate(sorted_idx, 1):
                if pids[idx] in expected_pids:
                    return rank
            return len(sorted_idx) + 1

    def get_best_sim(sim_row: np.ndarray, pids: list[str], expected_pids: set[str]) -> float:
        """Get best similarity score for expected pages."""
        best = -1.0
        for idx, pid in enumerate(pids):
            if pid in expected_pids:
                best = max(best, float(sim_row[idx]))
        return best

    # Collect page images needed
    needed_page_ids: set[str] = set()
    case_data: list[dict] = []

    for qi in tv_qi:
        gt = qa[qi]["ground_truth"]
        expected = set()
        for gi in gt:
            if gi in idx_to_info:
                fid, off = idx_to_info[gi]
                expected.add(f"{fid}_{off+1}")
        needed_page_ids.update(expected)

    print(f"Loading {len(needed_page_ids)} page images...")
    page_images = load_page_images(needed_page_ids)
    print(f"Loaded {len(page_images)} page images")

    # Build case data with ranks
    for qi in tv_qi:
        gt = qa[qi]["ground_truth"]
        expected = set()
        for gi in gt:
            if gi in idx_to_info:
                fid, off = idx_to_info[gi]
                expected.add(f"{fid}_{off+1}")

        pt_rank = get_rank(pt_sim[qi], pt_pids, expected)
        rg_rank = get_rank(rg_sim[qi], rg_pids, expected)
        rg_best_sim = get_best_sim(rg_sim[qi], rg_pids, expected)

        # Get region sims for all GT regions
        gt_region_sims = []
        for idx, pid in enumerate(rg_pids):
            if pid in expected:
                gt_region_sims.append((pid, float(rg_sim[qi][idx]), rg_meta[idx]))

        # Get crop images and captions for GT pages
        crops_info: list[dict] = []
        for pid in sorted(expected):
            if pid in ocr_by_pid:
                rec = ocr_by_pid[pid]
                for crop in rec.get("image_crops", []):
                    if crop.get("saved"):
                        crop_path = crop.get("path", "")
                        caption = captions_by_path.get(crop_path, "")
                        crops_info.append({
                            "path": crop_path,
                            "index": crop.get("index", -1),
                            "caption": caption,
                            "page_id": pid,
                        })

        # OCR text preview
        text_preview = ""
        for pid in sorted(expected):
            if pid in ocr_by_pid:
                rec = ocr_by_pid[pid]
                regions = rec.get("regions", [])
                parts = []
                for r in regions:
                    c = (r.get("content") or "").strip()
                    if c:
                        parts.append(c[:200])
                text_preview += f"[{pid}]\n" + "\n".join(parts[:5]) + "\n\n"

        case_data.append({
            "qi": qi,
            "query": qa[qi]["query"],
            "answer": qa[qi]["answer"],
            "domain": qa[qi].get("domain", ""),
            "gt_pids": sorted(expected),
            "pt_rank": pt_rank,
            "rg_rank": rg_rank,
            "rg_best_sim": rg_best_sim,
            "gt_region_sims": sorted(gt_region_sims, key=lambda x: -x[1]),
            "crops": crops_info,
            "text_preview": text_preview.strip()[:2000],
            "n_image": len(crops_info),
            "n_regions": sum(len(ocr_by_pid.get(pid, {}).get("regions", [])) for pid in expected),
        })

    # Compute summary stats
    n = len(case_data)
    pt_hits = sum(1 for c in case_data if c["pt_rank"] <= 10)
    rg_hits = sum(1 for c in case_data if c["rg_rank"] <= 10)
    both_hits = sum(1 for c in case_data if c["pt_rank"] <= 10 and c["rg_rank"] <= 10)
    rg_only = sum(1 for c in case_data if c["pt_rank"] > 10 and c["rg_rank"] <= 10)
    pt_only = sum(1 for c in case_data if c["pt_rank"] <= 10 and c["rg_rank"] > 10)
    neither = sum(1 for c in case_data if c["pt_rank"] > 10 and c["rg_rank"] > 10)

    print(f"\nSummary:")
    print(f"  PT R@10: {pt_hits}/{n} ({100*pt_hits/n:.1f}%)")
    print(f"  RG R@10: {rg_hits}/{n} ({100*rg_hits/n:.1f}%)")
    print(f"  Both: {both_hits}, RG only: {rg_only}, PT only: {pt_only}, Neither: {neither}")

    # Generate HTML
    print("\nGenerating HTML report...")

    def badge(rank: int, label: str) -> str:
        if rank <= 10:
            cls = "badge-hit"
        elif rank <= 50:
            cls = "badge-near"
        else:
            cls = "badge-miss"
        return f'<span class="badge {cls}">{label}: Rank {rank:,}</span>'

    def rank_class(rank: int) -> str:
        if rank <= 10:
            return "rank-good"
        if rank <= 50:
            return "rank-mid"
        return "rank-bad"

    lines: list[str] = []
    lines.append("""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>true_visual 40건 케이스 분석 보고서 (Corrected)</title>
<style>
body { font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { text-align: center; color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 15px; }
h2 { color: #16213e; margin-top: 40px; }
.summary { background: #fff; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.summary table { width: 100%; border-collapse: collapse; margin: 10px 0; }
.summary th, .summary td { padding: 8px 12px; border: 1px solid #ddd; text-align: center; }
.summary th { background: #16213e; color: white; }
.case { background: #fff; margin: 30px 0; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.case-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.case-header h3 { margin: 0; color: #16213e; font-size: 1.3em; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: bold; color: white; margin-left: 5px; }
.badge-hit { background: #27ae60; }
.badge-miss { background: #e74c3c; }
.badge-near { background: #f39c12; }
.query-box { background: #eef2f7; padding: 12px 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #3498db; }
.answer-box { background: #fef9e7; padding: 12px 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #f1c40f; font-size: 0.9em; }
.metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 15px 0; }
.metric { text-align: center; padding: 12px; border-radius: 8px; }
.metric-label { font-size: 0.8em; color: #666; }
.metric-value { font-size: 1.6em; font-weight: bold; }
.metric-pt { background: #ebf5fb; }
.metric-rg { background: #eafaf1; }
.images { display: flex; flex-wrap: wrap; gap: 15px; margin: 15px 0; align-items: flex-start; }
.img-container { text-align: center; }
.img-container img { max-height: 450px; border: 1px solid #ddd; border-radius: 5px; }
.img-container .page-img { max-width: 350px; }
.img-container .crop-img { max-width: 280px; max-height: 280px; }
.img-label { font-size: 0.8em; color: #666; margin-top: 5px; }
.caption-box { background: #f9f0ff; padding: 10px 12px; border-radius: 8px; margin: 5px 0; font-size: 0.82em; max-height: 120px; overflow-y: auto; line-height: 1.5; }
.rank-good { color: #27ae60; }
.rank-bad { color: #e74c3c; }
.rank-mid { color: #f39c12; }
.toc { background: #fff; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
.toc-grid { columns: 2; column-gap: 20px; }
.toc a { display: block; padding: 4px 0; text-decoration: none; color: #16213e; font-size: 0.9em; break-inside: avoid; }
.toc a:hover { color: #3498db; }
details { margin: 10px 0; }
summary { cursor: pointer; font-weight: bold; color: #16213e; }
pre { background: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }
.note { background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 20px 0; }
</style>
</head>
<body>
""")

    lines.append("<h1>true_visual 40건 케이스 분석 보고서 (Corrected Crops)</h1>")

    # Note about correction
    lines.append("""<div class="note">
<strong>NOTE:</strong> 이 보고서는 수정된 crop 이미지로 재생성된 region 임베딩 기반입니다.
이전 보고서의 crop_path mismatch 버그(40.3%)가 해결되었으며, 모든 21,052개 crop이 corpus page image + bbox로부터 재생성되었습니다.
Caption 실험은 제외 (캡션이 잘못된 crop에서 생성되어 invalid).
</div>""")

    # Summary
    lines.append(f"""<div class="summary">
<h2>Summary</h2>
<p>SDS-KoPub-VDR-Benchmark 600개 QA 중 <code>type=visual</code> 161건을 정성 전수 검토하여 true_visual 58건 분류 →
GT 페이지에 image/chart region이 존재하는 <strong>40건</strong>에 대한 상세 분석.</p>
<p><strong>Embedding Model</strong>: Qwen3-VL-Embedding-8B-FP8 (dim=4096) | <strong>Corpus</strong>: 40,781 pages | <strong>Regions</strong>: {rg_emb.shape[0]:,}</p>

<table>
<tr><th></th><th>pure_text</th><th>region</th></tr>
<tr><td><strong>R@10 Hit</strong></td><td>{pt_hits}/{n} ({100*pt_hits/n:.1f}%)</td><td><strong>{rg_hits}/{n} ({100*rg_hits/n:.1f}%)</strong></td></tr>
</table>

<h3>검색 패턴 분포</h3>
<table>
<tr><th>패턴</th><th>건수</th><th>비율</th><th>의미</th></tr>
<tr><td>Both hit (pt ∩ rg)</td><td>{both_hits}</td><td>{100*both_hits/n:.1f}%</td><td>텍스트 키워드도 있고 이미지 매칭도 성공</td></tr>
<tr><td>Region only (rg - pt)</td><td>{rg_only}</td><td>{100*rg_only/n:.1f}%</td><td>이미지로만 검색 가능 — region 임베딩의 핵심 가치</td></tr>
<tr><td>Pure text only (pt - rg)</td><td>{pt_only}</td><td>{100*pt_only/n:.1f}%</td><td>텍스트 단서로 검색되지만 이미지 매칭은 실패</td></tr>
<tr><td>Neither</td><td>{neither}</td><td>{100*neither/n:.1f}%</td><td>어떤 단일 임베딩으로도 R@10 내 검색 불가</td></tr>
</table>
</div>""")

    # TOC
    lines.append('<div class="toc"><h3>목차</h3><div class="toc-grid">')
    for i, c in enumerate(case_data):
        qi = c["qi"]
        if c["rg_rank"] <= 10:
            icon = "🟢"
        elif c["rg_rank"] <= 50:
            icon = "🟡"
        else:
            icon = "🔴"
        q_short = c["query"][:60] + "..." if len(c["query"]) > 60 else c["query"]
        lines.append(f'<a href="#case-{qi}">{icon} #{qi}: {html.escape(q_short)}</a>')
    lines.append('</div></div>')

    # Cases
    for i, c in enumerate(case_data):
        qi = c["qi"]
        lines.append(f'<div class="case" id="case-{qi}">')
        lines.append(f'<div class="case-header"><h3>Case #{qi} ({i+1}/{n})</h3>')
        lines.append(f'<div>{badge(c["rg_rank"], "Region")}{badge(c["pt_rank"], "Text")}</div></div>')

        lines.append(f'<div class="query-box"><strong>Query:</strong> {html.escape(c["query"])}</div>')
        lines.append(f'<div class="answer-box"><strong>Answer:</strong> {html.escape(c["answer"])}</div>')

        # Metrics
        lines.append('<div class="metrics">')
        lines.append(f'<div class="metric metric-pt"><div class="metric-label">pure_text rank</div>'
                     f'<div class="metric-value {rank_class(c["pt_rank"])}">{c["pt_rank"]:,}</div></div>')
        lines.append(f'<div class="metric metric-rg"><div class="metric-label">region rank</div>'
                     f'<div class="metric-value {rank_class(c["rg_rank"])}">{c["rg_rank"]:,}</div></div>')
        lines.append('</div>')

        # Page info
        gt_str = ", ".join(f'<code style="font-size:0.85em">{html.escape(p)}</code>' for p in c["gt_pids"])
        lines.append(f'<p><strong>GT Page:</strong> {gt_str}<br>'
                     f'<strong>Image regions:</strong> {c["n_image"]} | '
                     f'<strong>Total regions:</strong> {c["n_regions"]} | '
                     f'<strong>Domain:</strong> {html.escape(c["domain"])}</p>')

        # Images
        lines.append('<div class="images">')
        for pid in c["gt_pids"]:
            if pid in page_images:
                b64 = img_to_base64(page_images[pid], max_dim=700, quality=50)
                lines.append(f'<div class="img-container"><img class="page-img" src="data:image/jpeg;base64,{b64}" alt="page">'
                             f'<div class="img-label">GT Page: {html.escape(pid[-30:])}</div></div>')

        for crop in c["crops"]:
            crop_path = crop["path"]
            if crop_path and Path(crop_path).exists():
                img = Image.open(crop_path).convert("RGB")
                b64 = img_to_base64(img, max_dim=400, quality=60)
                lines.append(f'<div class="img-container"><img class="crop-img" src="data:image/jpeg;base64,{b64}" alt="crop">'
                             f'<div class="img-label">image #{crop["index"]}</div></div>')
        lines.append('</div>')

        # Captions
        captions_for_case = [crop["caption"] for crop in c["crops"] if crop["caption"]]
        if captions_for_case:
            lines.append(f'<details><summary>GPT-5-mini Captions ({len(captions_for_case)}건, 잘못된 crop 기반 — 참고용)</summary>')
            for cap in captions_for_case:
                lines.append(f'<div class="caption-box">{html.escape(cap)}</div>')
            lines.append('</details>')

        # Region similarities
        if c["gt_region_sims"]:
            lines.append(f'<details><summary>Region Similarities ({len(c["gt_region_sims"])}개 GT region)</summary>')
            lines.append('<table style="width:100%;border-collapse:collapse;margin:5px 0;font-size:0.85em;">')
            lines.append('<tr><th style="text-align:left;padding:4px;border-bottom:1px solid #ddd;">Page</th>'
                         '<th style="padding:4px;border-bottom:1px solid #ddd;">Similarity</th></tr>')
            for pid, sim_val, meta in c["gt_region_sims"][:10]:
                lines.append(f'<tr><td style="padding:4px;">{html.escape(pid[-40:])}</td>'
                             f'<td style="padding:4px;text-align:center;">{sim_val:.4f}</td></tr>')
            lines.append('</table></details>')

        # OCR text
        if c["text_preview"]:
            lines.append(f'<details><summary>OCR Text Preview</summary>'
                         f'<pre style="white-space:pre-wrap;font-size:0.82em;max-height:200px;overflow-y:auto;">'
                         f'{html.escape(c["text_preview"])}</pre></details>')

        lines.append('</div>')

    lines.append('</body></html>')

    html_content = "\n".join(lines)
    OUTPUT_HTML.write_text(html_content, encoding="utf-8")
    print(f"\nHTML report saved to {OUTPUT_HTML} ({len(html_content)/1024/1024:.1f} MB)")

    # Also copy to /tmp for easy access
    Path("/tmp/true_visual_case_report_corrected.html").write_text(html_content, encoding="utf-8")
    print(f"Also saved to /tmp/true_visual_case_report_corrected.html")


if __name__ == "__main__":
    main()
