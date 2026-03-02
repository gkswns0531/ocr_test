#!/usr/bin/env python3
"""Update HTML report: add caption rank + per-case analysis for 3 methods."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset
from openai import OpenAI

DATA_DIR = Path("output_dl")
EMB_DIR = DATA_DIR / "embeddings_8b"
HTML_PATH = DATA_DIR / "true_visual_case_report.html"

client = OpenAI()


def load_all_data() -> tuple:
    """Load QA, annotations, embeddings, OCR, captions."""
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

    # OCR results
    ocr_by_pid: dict[str, dict] = {}
    with open(DATA_DIR / "ocr_results.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            ocr_by_pid[rec["page_id"]] = rec

    # Captions per page
    page_captions: dict[str, str] = {}
    cap_file = DATA_DIR / "region_captions.jsonl"
    if cap_file.exists():
        page_cap_list: dict[str, list[str]] = {}
        with open(cap_file) as f:
            for line in f:
                rec = json.loads(line)
                cap = rec.get("caption", "").strip()
                if cap:
                    pid = rec["page_id"]
                    page_cap_list.setdefault(pid, []).append(cap)
        for pid, caps in page_cap_list.items():
            page_captions[pid] = "\n".join(caps)[:800]

    return qa, idx_to_info, tv_cases, queries, pt_emb, pt_pids, rg_emb, rg_pids, cap_emb, cap_pids, ocr_by_pid, page_captions


def get_rank(sim_row: np.ndarray, pids: list[str], expected: set[str]) -> int:
    sorted_idx = np.argsort(sim_row)[::-1]
    seen: set[str] = set()
    rank = 0
    for idx in sorted_idx:
        pid = pids[idx]
        if pid not in seen:
            seen.add(pid)
            rank += 1
            if pid in expected:
                return rank
    return rank + 1


def get_ocr_text_preview(ocr_by_pid: dict, pids: list[str]) -> str:
    parts = []
    for pid in pids:
        if pid in ocr_by_pid:
            rec = ocr_by_pid[pid]
            for r in rec.get("regions", []):
                c = (r.get("content") or "").strip()
                if c and r.get("label") != "table":
                    parts.append(c[:150])
    return "\n".join(parts[:8])[:600]


def generate_analysis(case: dict) -> str:
    """Generate 1-2 sentence analysis per case using GPT-4o-mini."""
    qi = case["qi"]
    query = case["query"][:200]
    pt_rank = case["pt_rank"]
    rg_rank = case["rg_rank"]
    cap_rank = case["cap_rank"]
    ocr_preview = case["ocr_preview"][:400]
    caption_preview = case["caption_preview"][:400]

    def rank_status(r: int) -> str:
        if r <= 10:
            return f"성공(Rank {r})"
        elif r <= 50:
            return f"근접(Rank {r})"
        else:
            return f"실패(Rank {r:,})"

    prompt = f"""다음은 한국어 문서 검색 벤치마크의 한 케이스입니다. 3가지 검색 방식(pure_text, region, caption)의 결과를 분석해 주세요.

Query: {query}
pure_text rank: {rank_status(pt_rank)}
region rank: {rank_status(rg_rank)}
caption rank: {rank_status(cap_rank)}

페이지의 OCR 텍스트 일부:
{ocr_preview}

페이지의 캡션 텍스트 일부:
{caption_preview}

각 방식이 왜 성공/실패했는지 1-2문장으로 분석해 주세요.
- 구체적인 이유를 들어 설명 (예: "텍스트에 관련 키워드가 포함", "이미지가 쿼리의 차트/도표를 직접 포함", "캡션이 핵심 정보를 누락")
- 3가지 방식을 모두 언급
- 한국어로 작성, 총 2-3문장"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    print("Loading data...")
    (qa, idx_to_info, tv_cases, queries,
     pt_emb, pt_pids, rg_emb, rg_pids,
     cap_emb, cap_pids, ocr_by_pid, page_captions) = load_all_data()

    tv_qi = [c["qi"] for c in tv_cases]

    pt_sim = queries @ pt_emb.T
    rg_sim = queries @ rg_emb.T
    cap_sim = queries @ cap_emb.T

    # Compute ranks and build case data
    cases: list[dict] = []
    for qi in tv_qi:
        gt = qa[qi]["ground_truth"]
        expected = set()
        for gi in gt:
            if gi in idx_to_info:
                fid, off = idx_to_info[gi]
                expected.add(f"{fid}_{off+1}")

        pt_rank = get_rank(pt_sim[qi], pt_pids, expected)
        rg_rank = get_rank(rg_sim[qi], rg_pids, expected)
        cap_rank = get_rank(cap_sim[qi], cap_pids, expected)

        ocr_preview = get_ocr_text_preview(ocr_by_pid, sorted(expected))
        caption_preview = ""
        for pid in sorted(expected):
            if pid in page_captions:
                caption_preview += page_captions[pid][:400] + "\n"

        cases.append({
            "qi": qi,
            "query": qa[qi]["query"],
            "pt_rank": pt_rank,
            "rg_rank": rg_rank,
            "cap_rank": cap_rank,
            "ocr_preview": ocr_preview,
            "caption_preview": caption_preview.strip()[:600],
        })

    # Generate analyses in parallel
    print("Generating per-case analyses (40 cases)...")
    analyses: dict[int, str] = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(generate_analysis, c): c["qi"] for c in cases}
        done = 0
        for fut in as_completed(futures):
            qi = futures[fut]
            try:
                analyses[qi] = fut.result()
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/40 done ({time.time()-t0:.1f}s)")
            except Exception as e:
                analyses[qi] = f"분석 생성 실패: {e}"
                done += 1

    print(f"All analyses done in {time.time()-t0:.1f}s")

    # Save analyses
    with open("/tmp/case_analyses.json", "w") as f:
        json.dump({str(qi): txt for qi, txt in analyses.items()}, f, ensure_ascii=False, indent=2)

    # Now update HTML
    print("Updating HTML...")
    html = HTML_PATH.read_text(encoding="utf-8")

    # 1. Update CSS: change metrics grid to 3 columns, add caption metric style
    html = html.replace(
        ".metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 15px 0; }",
        ".metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }"
    )
    # Add caption metric style
    html = html.replace(
        ".metric-rg { background: #eafaf1; }",
        ".metric-rg { background: #eafaf1; }\n.metric-cap { background: #fef0f5; }\n"
        ".analysis-box { background: #f8f9fa; padding: 12px 15px; border-radius: 8px; margin: 10px 0; "
        "border-left: 4px solid #9b59b6; font-size: 0.9em; line-height: 1.6; }"
    )

    def rank_class(rank: int) -> str:
        if rank <= 10:
            return "rank-good"
        if rank <= 50:
            return "rank-mid"
        return "rank-bad"

    def badge_class(rank: int) -> str:
        if rank <= 10:
            return "badge-hit"
        if rank <= 50:
            return "badge-near"
        return "badge-miss"

    # 2. For each case, add caption badge, caption metric, and analysis
    for c in cases:
        qi = c["qi"]
        cap_rank = c["cap_rank"]
        analysis = analyses.get(qi, "")

        # Add caption badge next to existing badges
        # Find: <span class="badge badge-xxx">Text: Rank NNN</span></div></div>
        # in the case header for this qi
        old_header_end = f'</span></div></div>\n<div class="query-box">'
        # Need to find case-specific pattern - look for case-{qi}
        case_start = html.find(f'id="case-{qi}"')
        if case_start == -1:
            print(f"  Case #{qi} not found in HTML, skipping")
            continue

        # Find the next query-box after case start
        query_box_pos = html.find('<div class="query-box">', case_start)
        if query_box_pos == -1:
            continue

        # Find the Text badge (last badge before query-box)
        header_section = html[case_start:query_box_pos]

        # Add caption badge after the Text badge
        text_badge_end = header_section.rfind("</span>")
        if text_badge_end != -1:
            abs_pos = case_start + text_badge_end + len("</span>")
            cap_badge = f'<span class="badge {badge_class(cap_rank)}">Caption: Rank {cap_rank:,}</span>'
            html = html[:abs_pos] + cap_badge + html[abs_pos:]

        # Find metrics section for this case and add caption metric
        # Recalculate positions since we modified the string
        case_start = html.find(f'id="case-{qi}"')
        metrics_start = html.find('<div class="metrics">', case_start)
        metrics_end = html.find('</div>\n</div>', metrics_start)
        if metrics_end == -1:
            metrics_end = html.find('</div>\n<p>', metrics_start)

        if metrics_start != -1:
            # Find the closing </div> of the last metric
            rg_metric_end = html.find('</div></div>', metrics_start + 20)
            if rg_metric_end != -1:
                insert_pos = rg_metric_end + len('</div></div>')
                cap_metric = (
                    f'\n<div class="metric metric-cap"><div class="metric-label">caption rank</div>'
                    f'<div class="metric-value {rank_class(cap_rank)}">{cap_rank:,}</div></div>'
                )
                html = html[:insert_pos] + cap_metric + html[insert_pos:]

        # Add analysis box after the metrics section
        if analysis:
            case_start = html.find(f'id="case-{qi}"')
            # Find </div> that closes metrics, then the <p> tag after it
            metrics_start = html.find('<div class="metrics">', case_start)
            # Find the closing </div> of metrics section
            # The metrics div contains 3 metric divs, then closes
            pos = metrics_start
            # Skip past the 3 metric divs and the closing metrics div
            metrics_close = html.find('\n<p>', metrics_start)
            if metrics_close != -1:
                analysis_html = (
                    f'\n<div class="analysis-box">'
                    f'<strong>분석:</strong> {analysis}'
                    f'</div>'
                )
                html = html[:metrics_close] + analysis_html + html[metrics_close:]

    # 3. Update the summary section
    # Update the Recall table with caption column
    case_map = {c["qi"]: c for c in cases}
    cap_hits_10 = sum(1 for c in cases if c["cap_rank"] <= 10)

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"HTML updated: {len(html)/1024/1024:.1f} MB")
    print(f"Caption R@10: {cap_hits_10}/40 ({100*cap_hits_10/40:.1f}%)")


if __name__ == "__main__":
    main()
