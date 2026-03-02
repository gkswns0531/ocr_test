#!/usr/bin/env python3
"""Generate captions using GPT-4o-mini with production-grade exhaustive prompt.

Output: region_captions.jsonl
"""
from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

DATA_DIR = Path("output_dl")
OCR_RESULTS_FILE = DATA_DIR / "ocr_results.jsonl"
CAPTIONS_FILE = DATA_DIR / "region_captions.jsonl"

MODEL = "gpt-4o-mini"
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60

CAPTION_PROMPT = r"""Analyze this image with extreme thoroughness. You must produce a comprehensive, structured analysis covering EVERY applicable section below. Each section must contain multiple detailed sentences — not brief summaries. If a section does not apply, explicitly state "해당 없음" and briefly explain why.

## 출력 형식
각 섹션을 아래 번호와 제목으로 시작하고, 해당 내용을 상세하게 서술하시오.

---

### 1. 이미지 유형 및 전체 구성
- 이미지의 정확한 유형을 판별하시오 (사진, 차트, 그래프, 표, 다이어그램, 지도, 일러스트, 스크린샷, 문서 스캔, 인포그래픽, 플로차트, 조직도, 타임라인, 설계도 등)
- 전체 레이아웃의 구조를 설명하시오 (단일 이미지, 다중 패널, 격자형, 계층형 등)
- 이미지의 가로세로 비율, 해상도 품질, 전반적인 선명도를 평가하시오
- 주요 시각적 요소들의 배치와 비중을 설명하시오

### 2. 텍스트 완전 전사 및 서식 묘사
- 이미지에 보이는 모든 텍스트를 빠짐없이 전사하시오: 제목, 부제목, 본문, 캡션, 주석, 범례, 축 라벨, 헤더, 푸터, 워터마크, 페이지 번호, 날짜, 출처 표기
- 각 텍스트의 정확한 위치(상단, 하단, 좌측, 우측, 중앙 등)를 명시하시오
- 텍스트의 언어, 글꼴 스타일(굵게, 기울임, 밑줄, 취소선, 음영 처리), 글꼴 종류(고딕, 명조, 손글씨, 장식체 등)를 기술하시오
- 글자 크기의 상대적 차이를 구체적으로 기술하시오 (예: 제목은 본문의 약 2배 크기, 주석은 본문보다 작음 등)
- 글자 색상을 모두 기술하시오 (검정, 회색, 빨강, 파랑, 흰색 등). 배경색과의 대비도 언급하시오
- 텍스트 정렬 방식 (좌측 정렬, 중앙 정렬, 우측 정렬, 양쪽 정렬)을 기술하시오
- 텍스트가 박스, 말풍선, 배너, 리본, 원형 배지 등 특정 도형 안에 들어있는 경우 그 도형의 형태, 색상, 테두리를 함께 기술하시오
- 줄 간격, 자간, 들여쓰기 등 타이포그래피적 특성이 눈에 띄는 경우 기술하시오
- 숫자, 단위, 백분율, 날짜, 코드, 약어를 정확히 기록하시오
- 특수문자, 수식, 화학식 등이 있으면 가능한 정확히 전사하시오
- 글머리 기호(bullet), 번호 매김(numbering) 형식이 사용된 경우 그 스타일을 기술하시오

### 3. 데이터 시각화 상세 분석
- 차트/그래프 유형 (막대, 선, 원, 산점도, 히트맵, 박스플롯, 워터폴, 트리맵, 버블, 레이더, 영역 등)
- X축과 Y축의 라벨, 단위, 범위, 눈금 간격을 정확히 기술하시오
- 범례의 모든 항목과 해당 색상/패턴을 기술하시오
- 데이터의 최댓값, 최솟값, 평균적 추세, 변곡점, 이상치를 식별하시오
- 카테고리 간 비교, 시계열 추세, 비율 관계를 수치와 함께 분석하시오
- 표가 있는 경우: 행과 열의 구조, 헤더, 모든 셀 값을 가능한 전사하시오
- 데이터에서 도출할 수 있는 핵심 인사이트 2~3가지를 제시하시오

### 4. 인물 상세 묘사
- 각 인물의 추정 연령대, 성별, 인종/민족적 특징을 기술하시오
- 의복: 상의, 하의, 신발, 액세서리(안경, 모자, 시계, 가방 등)의 색상과 스타일
- 자세와 동작: 서 있음, 앉아 있음, 걷고 있음, 뛰고 있음, 점프, 손짓, 글쓰기 등 구체적으로
- 표정과 감정 상태: 미소, 진지함, 놀람, 집중, 대화 중 등
- 인물 간 상호작용: 대화, 협업, 발표, 악수, 포옹 등
- 인물의 시선 방향과 신체 언어가 암시하는 맥락

### 5. 사물 및 환경 요소
- 모든 주요 사물을 나열하고 각각의 색상, 크기(상대적), 재질, 상태를 기술하시오
- 가구, 장비, 도구, 차량, 음식, 식물, 동물 등 카테고리별로 분류
- 사물 간의 공간적 관계 (위에, 옆에, 앞에, 뒤에, 안에 등)
- 브랜드명이나 모델명이 식별 가능한 경우 기록하시오

### 6. 색채, 조명, 시각적 스타일
- 지배적인 색상 팔레트 (주요 3~5가지 색상과 그 역할)
- 조명의 유형: 자연광/인공조명, 방향, 강도, 색온도 (따뜻한/차가운)
- 명암 대비, 채도, 그림자의 특성
- 시각적 효과: 필터, 블러, 그라데이션, 테두리, 그림자 효과 등
- 전체적인 분위기와 톤 (밝고 경쾌한, 어둡고 진지한, 전문적인, 캐주얼한 등)

### 7. 공간 구성 및 원근
- 전경, 중경, 배경 각각에 위치한 요소들을 구분하여 기술하시오
- 촬영 각도/시점: 정면, 측면, 조감도, 클로즈업, 와이드샷 등
- 피사계 심도와 초점 영역
- 실내/실외 구분, 구체적인 장소 유형 (사무실, 회의실, 거리, 공원, 공장 등)
- 계절, 시간대, 날씨 등 환경적 단서

### 8. 로고, 심볼, 브랜드 요소
- 모든 로고, 상표, 아이콘, 심볼, 국기, 엠블럼을 식별하고 묘사하시오
- 각 로고의 위치, 크기, 색상을 기술하시오
- 식별 가능한 기관명, 회사명, 브랜드명을 기록하시오
- UI 요소(버튼, 메뉴, 아이콘)가 있는 경우 그 기능과 레이아웃을 설명하시오

### 9. 문서 구조 및 레이아웃 분석 (문서/보고서인 경우)
- 문서의 유형: 보고서, 논문, 발표자료, 법률문서, 재무제표, 뉴스기사 등
- 장/절/항의 구조와 번호 체계
- 각주, 미주, 참고문헌, 인용 표시
- 페이지 레이아웃: 단수/다단, 여백 크기(넓음/좁음), 머리글/바닥글 유무와 내용
- 단(column)의 수와 각 단의 폭 비율, 단 사이 구분선 유무
- 강조 표시(하이라이트, 밑줄, 박스, 색상 배경)된 부분과 그 내용
- 표의 시각적 서식: 셀 테두리 스타일(실선, 점선, 없음), 헤더 행/열의 배경색, 셀 병합, 줄무늬(zebra striping) 여부
- 구분선(수평선, 점선), 화살표, 괄호 등 구조적 시각 요소
- 이미지/도표가 본문 안에 삽입된 방식 (인라인, 플로팅, 전체 폭, 캡션 위치)
- 전체적인 디자인 톤: 공식적/비공식적, 미니멀/장식적, 컬러풀/모노톤

### 10. 맥락, 목적, 대상
- 이 이미지의 제작 목적을 추론하시오 (교육, 홍보, 보도, 과학, 예술, 정보 전달, 오락, 기록 등)
- 예상 대상 독자/시청자층
- 이미지가 전달하려는 핵심 메시지 또는 논지
- 시대적, 문화적, 산업적 맥락
- 관련될 수 있는 분야: 경제, 정치, 과학, 기술, 의학, 교육, 환경, 사회 등

### 11. 접근성 및 대체 텍스트
- 시각장애인을 위한 핵심 정보 요약 (3~4문장)
- 색맹 사용자에게 문제가 될 수 있는 색상 조합이 있는지 평가

### 12. 시각적 서식 및 디자인 요소 종합
- 전체 이미지에 사용된 색상 체계가 일관된 브랜드/테마 색상인지 판단하시오
- 그래프/차트의 데이터 계열별 색상, 패턴(빗금, 점, 격자), 투명도를 기술하시오
- 테두리, 그림자, 둥근 모서리 등 장식적 요소를 기술하시오
- 아이콘이나 픽토그램이 사용된 경우 그 스타일(라인, 솔리드, 컬러, 모노)과 크기를 기술하시오
- 여백(padding/margin)의 활용, 요소 간 간격의 균일성을 평가하시오
- 시각적 계층구조: 어떤 요소가 가장 먼저 시선을 끄는지, 정보의 우선순위가 디자인으로 어떻게 표현되는지 분석하시오

---

모든 출력은 반드시 한국어로 작성하시오. 각 섹션을 빠짐없이 작성하되, 해당 없는 섹션도 "해당 없음"으로 명시적으로 표기하시오. 최대한 상세하고 포괄적으로 기술하시오."""


def load_processed(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("caption"):
                        done.add(f"{rec['page_id']}_{rec['region_index']}")
    return done


def image_to_base64(img_path: str, max_pixels: int = 1_000_000) -> str:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def caption_one(client: OpenAI, item: dict) -> dict:
    b64 = image_to_base64(item["crop_path"])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }],
        max_tokens=16384,
    )
    caption = resp.choices[0].message.content or ""
    usage = resp.usage
    return {
        "page_id": item["page_id"],
        "region_index": item["region_index"],
        "crop_path": item["crop_path"],
        "label": item["label"],
        "caption": caption.strip(),
        "tokens_prompt": usage.prompt_tokens if usage else 0,
        "tokens_completion": usage.completion_tokens if usage else 0,
    }


def extract_region_items() -> list[dict]:
    items: list[dict] = []
    with open(OCR_RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_id = rec["page_id"]
            for crop in rec.get("image_crops", []):
                if not crop.get("saved"):
                    continue
                crop_path = crop.get("path", "")
                if not crop_path or not Path(crop_path).exists():
                    continue
                items.append({
                    "page_id": page_id,
                    "region_index": crop.get("index", -1),
                    "crop_path": crop_path,
                    "label": crop.get("label", ""),
                })
    return items


def main() -> None:
    print("=" * 60)
    print(f"Generate Captions — {MODEL} (production)")
    print("=" * 60)
    print(f"  Prompt length: {len(CAPTION_PROMPT)} chars")
    print(f"  Max tokens: 16384")

    items = extract_region_items()
    print(f"  Total: {len(items)}")

    done = load_processed(CAPTIONS_FILE)
    pending = [it for it in items if f"{it['page_id']}_{it['region_index']}" not in done]
    print(f"  Done: {len(done)}, Pending: {len(pending)}")

    if not pending:
        print("  Nothing to do!")
        return

    workers = 256
    print(f"  Workers: {workers}")

    client = OpenAI()
    t_start = time.time()
    tp = tc = errors = empty = 0

    out_f = open(CAPTIONS_FILE, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(caption_one, client, it): it for it in pending}
            pbar = tqdm(total=len(futures), desc=f"Captioning ({MODEL} caption)")
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    tp += result.get("tokens_prompt", 0)
                    tc += result.get("tokens_completion", 0)
                    if not result.get("caption"):
                        empty += 1
                except Exception as e:
                    item = futures[fut]
                    errors += 1
                    out_f.write(json.dumps({
                        "page_id": item["page_id"], "region_index": item["region_index"],
                        "crop_path": item.get("crop_path", ""), "label": item.get("label", ""),
                        "caption": "", "error": str(e),
                    }, ensure_ascii=False) + "\n")
                pbar.update(1)
                if pbar.n % 100 == 0:
                    out_f.flush()
                    el = time.time() - t_start
                    cost = (tp * PRICE_INPUT + tc * PRICE_OUTPUT) / 1_000_000
                    pbar.set_postfix(rate=f"{pbar.n/el:.1f}/s", cost=f"${cost:.2f}",
                                     est=f"${cost*len(pending)/max(pbar.n,1):.2f}", err=errors, empty=empty)
            pbar.close()
    finally:
        out_f.flush()
        out_f.close()

    el = time.time() - t_start
    cost = (tp * PRICE_INPUT + tc * PRICE_OUTPUT) / 1_000_000
    print(f"\n  Done in {el:.0f}s ({el/60:.1f}m)")
    print(f"  OK: {len(pending)-errors}, Errors: {errors}, Empty: {empty}")
    print(f"  Tokens: {tp:,} + {tc:,}, Cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
