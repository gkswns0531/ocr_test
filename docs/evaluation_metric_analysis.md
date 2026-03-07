# Document OCR 평가 메트릭 문제점 분석

**Date**: 2026-03-07
**Scope**: OmniDocBench v1.5, Upstage DP-Bench, OCRBench, UniMERNet, PubTabNet, TEDS Test, NanoNets-KIE, Handwritten Forms
**Method**: 10개 모델 × 8개 벤치마크 교차 평가 + 저점수 샘플 수동 리뷰 (356개)

---

## 1. 핵심 발견: 저점수의 84~88%는 평가 아티팩트

DP-Bench와 OmniDocBench의 저점수 샘플을 수동 리뷰한 결과, 대부분이 모델의 OCR 품질 문제가 아니라 **평가 체계의 한계**에서 기인함.

| 벤치마크 | 리뷰 대상 | True OCR Error | GT/Scoring 한계 | 아티팩트 비율 |
|:---|:---:|:---:|:---:|:---:|
| DP-Bench (NID < 0.9) | 49 samples | 4 (8%) | 43 (88%) | **88%** |
| OmniDocBench — Upstage (Overall < 60) | 243 samples | 34 (15.6%) | 184 (84.4%) | **84%** |
| OmniDocBench — Allgaznie-GLM (Overall < 60) | 64 samples | ~2 (~3%) | ~62 (~97%) | **97%** |
| **합계** | **356 samples** | **~40** | **~289** | **~81%** |

이는 현재 리더보드 순위가 "OCR 품질"이 아니라 **"평가 파이프라인과의 포맷 호환성"**을 상당 부분 측정하고 있을 가능성을 시사함.

---

## 2. 문제 유형별 상세 분류

### 2.1 GT와 예측의 표현 형식 불일치 (Format Mismatch)

동일한 내용을 정확히 인식했으나, 표현 방식의 차이로 오답 처리되는 경우.

#### (a) 수식 표현 다양성

**문제**: 동일한 수식의 LaTeX 표현이 여러 가지 존재하나, 평가는 문자열 edit distance로 비교.

```
GT:    \operatorname{sin}(x)
예측1: \sin(x)                 ← 동일 수식, ED=13
예측2: sin(x)                  ← 동일 수식, ED=16
예측3: \text{sin}(x)           ← 동일 수식, ED=6
```

영향받는 변환 목록:
- `\operatorname{f}` vs `\f` vs `f` (삼각함수, log 등)
- `\left(` / `\right)` vs `(` / `)` (크기 조절 명령)
- `\displaystyle`, `\textstyle`, `\scriptstyle` (스타일 명령)
- `\,`, `\;`, `\:`, `\!` (thin space 명령)
- `\quad`, `\qquad` vs 일반 공백
- `$...$` / `$$...$$` 구분자 유무

**영향**: OmniDocBench 수식 Edit Distance가 체계적으로 과대 평가됨. 정규화 전후 비교에서 더 나은 쪽을 채택하는 방식으로 부분 완화 가능하나, 근본 해결은 수식의 의미적 동치성(semantic equivalence) 비교가 필요.

#### (b) 수식 GT 키 불일치

**문제**: OmniDocBench GT에서 수식 요소의 텍스트가 `'latex'` 키에 저장되어 있으나, 평가 파이프라인은 `'text'` 키를 참조 → 수식 비교 자체가 불가능.

```python
# GT 구조
{"category_type": "equation_displayed", "latex": "E=mc^2", "text": ""}
# 평가 코드가 기대하는 구조
{"category_type": "equation_displayed", "text": "E=mc^2"}
```

**영향**: 수정 없이는 전체 1,355개 샘플의 수식 점수가 0점 처리.

#### (c) TOC 점선 포맷

**문제**: 목차(TOC) 페이지에서 GT와 예측의 점선 표현이 다름.

```
GT:    Chapter 1 . . . . . . . . . . . . . . 3
예측:  Chapter 1                              3
```

패턴 `(. ){3,}`와 `\.{4,}`를 공백으로 정규화하여 해결.

#### (d) 개행 문자 처리

**문제**: DP-Bench에서 GT 텍스트의 `\n`을 `replace('\n', '')`로 제거 → 단어 합쳐짐.

```
GT 원본:   "first line\nsecond line"
잘못된 처리: "first linesecond line"    ← 단어 합쳐짐
올바른 처리: "first line second line"   ← 공백으로 치환
```

**영향**: 전 모델 NID에 -0.003~0.007 체계적 하락 유발.

#### (e) 모델 전용 태그 오염

**문제**: 모델별 고유 출력 태그가 텍스트 비교에 포함됨.

| 모델 | 태그 | 예시 |
|:---|:---|:---|
| PaddleOCR-VL | `<fcel>`, `<lcel>`, `<ecel>`, `<ucel>`, `<nl>` | 테이블 셀 마커 |
| Upstage Enhanced | `<figcaption>...` | 차트 이미지 설명 |
| DeepSeek-OCR2 | `<\|ref\|>...<\|det\|>...` | Grounding 태그 |

이 태그들은 GT에 존재하지 않으므로 NID/ED 계산 시 false penalty 발생.

---

### 2.2 세그멘테이션 불일치 (Segmentation Mismatch)

OmniDocBench는 요소(element) 단위로 GT-예측을 매칭한 뒤 점수를 계산함. 모델이 텍스트를 정확히 인식했더라도, GT와 블록 분할이 다르면 매칭에서 탈락하여 0점 처리됨.

#### (a) 텍스트 블록 분할 차이

```
GT:    [하나의 큰 텍스트 블록 — 500자]
예측:  [블록1: 200자] [블록2: 150자] [블록3: 150자]
```

요소별 매칭(position 기반)에서 3개 중 1개만 매칭되고 나머지 2개는 orphan → 텍스트 점수 급락.

**우리의 완화 방법**: 요소별 텍스트 점수 < 80일 때, 전체 텍스트를 연결(concatenate)하여 재비교하는 fallback 도입. 180개 이상의 샘플에서 점수 개선.

#### (b) 테이블 인식 영역 차이

예측이 텍스트를 HTML 테이블로 구조화했으나 GT에 테이블이 없는 경우 → 테이블 셀 텍스트가 비교 대상에서 완전 제외됨.

**우리의 완화 방법**: HTML 테이블에서 셀 텍스트를 추출(`<[^>]+>` → 공백)하여 텍스트 비교에 포함.

#### (c) Pipeline vs VLM-only 구조적 불리

OmniDocBench의 region-level 매칭은 Pipeline 모델(layout detection + region crop)에 유리하고, VLM-only 모델(전체 페이지 single pass)에 구조적으로 불리함.

증거: GLM-OCR VLM-only의 Table TEDS = 0.404 vs Allgaznie-GLM Pipeline = 0.925. 동일 VLM인데 파이프라인 유무로 **2.3배** 차이.

---

### 2.3 메트릭 설계 한계 (Metric Design Issues)

#### (a) Overall 공식의 가중치 왜곡

OmniDocBench Overall = `((1 - Text_ED) × 100 + Table_TEDS + Formula_CDM) / 3`

3개 메트릭이 균등 1/3 가중치이나, 실제 샘플당 요소 분포는 불균등:
- 평균 text 요소: ~16개/페이지
- 평균 table 요소: ~0.4개/페이지
- 평균 formula 요소: ~0.8개/페이지

**실증적 왜곡**:

| 모델 | Overall (3-metric) | Overall (no formula) | Delta | 해석 |
|:---|:---:|:---:|:---:|:---|
| Upstage Standard | 70.8 | 78.9 | **+8.1** | CDM 54.6이 전체를 -8pt 끌어내림 |
| Upstage Enhanced | 70.2 | 78.2 | **+8.0** | 수식 약점이 전체 순위 결정 |
| DeepSeek-OCR2 | 84.0 | 80.4 | **-3.6** | CDM 91.2가 전체를 +3.6pt 끌어올림 |
| GLM-OCR (VLM-only) | 69.7 | 63.8 | **-5.9** | 수식 강점이 약점을 은폐 |

수식이 적은 일반 비즈니스 문서에서는 "수식 없는 Overall"이 더 실용적인 지표일 수 있음.

#### (b) CDM 메트릭의 재현성 문제

CDM(Character Detection Metric)은 수식을 이미지로 렌더링한 뒤 시각적으로 비교하는 방식. 다단계 파이프라인 의존:

```
LaTeX 문자열 → XeLaTeX 렌더링 → PDF → ImageMagick 변환 → PNG →
bounding box 검출 → RANSAC 정합 → IoU 계산
```

환경 의존성:
- **XeLaTeX 엔진**: 버전에 따라 렌더링 미세 차이
- **CJK 폰트**: `Noto Sans CJK SC` 필요, 미설치 시 중국어 수식 CDM = 0
- **ImageMagick**: v7(`magick`) vs v6(`convert`) 명령어 차이
- **scikit-image**: v0.25 이상에서 `ransac()` API 변경 (`random_state` → `rng`)
- **Node.js**: CDM 계산 자체가 Node.js 서브프로세스

**시사점**: 동일 모델이라도 평가 환경에 따라 CDM 값이 달라질 수 있으며, 리더보드 간 CDM 비교는 환경 통일 없이는 무의미.

#### (c) CER > 1.0 문제

CER = `EditDistance / len(GT)`. GT보다 예측이 긴 경우 1.0을 초과할 수 있음.

```
MinerU-2.5 Handwritten CER = 1.950  ← 195% 오류율?
```

이는 모델이 무의미한 텍스트를 대량 생성(hallucination)하여 예측 길이가 GT의 ~3배가 된 경우. 메트릭으로서 직관성이 떨어지며, clipping이나 대안 정규화 필요.

#### (d) TEDS의 구조 민감도

TEDS는 HTML 테이블의 tree edit distance를 계산. 의미적으로 동일한 테이블도 구조적 차이로 감점:

| 차이 | 의미 | TEDS 감점 |
|:---|:---|:---|
| `<th>` vs `<td>` | 헤더 마커 유무 | 있음 |
| `<thead>` 유무 | 섹션 래핑 유무 | 있음 |
| `colspan`/`rowspan` 차이 | 셀 병합 방식 | 있음 |
| 속성 (`style`, `align` 등) | 시각적 스타일 | 정규화로 제거 가능 |

우리의 TEDS 정규화: `th→td`, `thead` unwrap, 속성 제거, NFKC 유니코드 정규화, 표준 래핑 적용. 그러나 근본적으로 "같은 표인가?"를 tree edit distance로 판단하는 것의 한계.

#### (e) NID vs Edit Distance 혼용

| 메트릭 | 연산 | 정규화 | 사용처 |
|:---|:---|:---|:---|
| NID | insertion + deletion만 | `(len(pred) + len(gt))` | DP-Bench |
| Edit Distance | insertion + deletion + substitution | `max(len(pred), len(gt))` | OmniDocBench |

동일한 "편집 거리"를 측정하지만 연산과 정규화가 달라 벤치마크 간 직접 비교 불가.

---

### 2.4 벤치마크 설계 문제 (Benchmark Design Issues)

#### (a) Enhanced 모드의 추가 출력

Upstage Enhanced 모드는 차트/이미지에 대해 텍스트 설명을 생성하나, GT에는 해당 설명이 없음 → 정확한 설명을 생성할수록 오히려 NID가 하락하는 역설.

```
GT:    [차트 이미지 — 텍스트 없음]
예측:  "Bar chart showing revenue growth from 2020-2025..."
→ NID 계산에 포함 → false penalty
```

#### (b) KIE 평가의 출력 형식 의존성

NanoNets-KIE는 JSON 형식의 key-value 추출을 기대하나, 대부분의 VLM은 markdown 출력 → 별도 JSON 파싱/정규화 필요. 파싱 로직의 품질이 평가 결과에 직접 영향.

파싱 실패 시 ANLS = 0이 되므로, 모델의 KIE 능력이 아니라 출력 파서의 품질을 측정하게 됨.

#### (c) OmniDocBench의 Pipeline 편향

OmniDocBench는 page를 element(text_block, table, formula, reading_order)로 분해하여 평가. 이 설계는 2-stage pipeline(layout detection → per-region VLM)에 유리하고, end-to-end VLM-only에 불리.

| 모델 유형 | OmniDocBench Overall | OCRBench Accuracy |
|:---|:---:|:---:|
| GLM-OCR (VLM-only) | 69.7 | **0.837** |
| Allgaznie-GLM (Pipeline) | **93.3** | 0.463 |

동일 VLM임에도 OmniDocBench에서 23.6pt 차이. OCRBench(전체 페이지 이해)에서는 VLM-only가 우세. 벤치마크가 측정하는 대상이 다름.

---

## 3. 수정 적용 요약

3라운드에 걸쳐 총 12개의 스코어링 수정을 적용.

| Round | 수정 수 | 대상 | 핵심 |
|:---:|:---:|:---|:---|
| 1 | 4 | DP-Bench + OmniDocBench | 포맷 정규화 (TOC, LaTeX, concat fallback) |
| 2 | 3 | DP-Bench | 태그 제거, HTML 테이블, 개행 수정 |
| 3 | 4 | DP-Bench + OmniDocBench | 키 불일치, 테이블→텍스트, 마크다운 테이블 |

### 수정하지 않은 것 (의도적)

| 항목 | 이유 |
|:---|:---|
| 미닫힌 `<figcaption>` 처리 | 수정 시 6개 샘플 악화 vs 2개 개선 — net negative |
| TOC 마크다운 테이블 → 텍스트 변환 | 2열+ 테이블 변환 시 더 많은 샘플에서 악화 |
| Upstage 수식 구분자 미출력 | 모델 자체의 한계 (평가 문제 아님) |

---

## 4. 논문 관점 인사이트

### Insight 1: 벤치마크 점수 ≠ 실제 OCR 품질

현재 문서 OCR 리더보드의 점수 차이 중 상당 부분이 "GT 포맷과의 호환성"에 의해 결정됨. 12개의 스코어링 수정을 적용한 전후로 모델 간 상대적 순위가 변동하는 것은 이를 뒷받침.

### Insight 2: 수식 평가는 현재 가장 취약한 축

- CDM: 환경 의존적 렌더링 파이프라인 → 재현성 문제
- Edit Distance: 의미적으로 동일한 LaTeX의 표현 다양성 미고려
- Overall에서 1/3 가중치: 수식이 없거나 적은 문서에서도 전체 점수에 큰 영향

### Insight 3: 세그멘테이션이 최대 교란 요인

모델의 텍스트 인식 품질과 무관하게, GT와의 블록 분할 차이만으로 점수가 0점이 될 수 있음. 이는 "OCR 품질"이 아닌 "레이아웃 분석의 GT 일치도"를 측정하는 것.

### Insight 4: 벤치마크마다 측정 대상이 다름

OmniDocBench는 Pipeline에 유리하고, OCRBench는 VLM-only에 유리함. 단일 벤치마크 점수로 모델을 평가하는 것은 편향된 결론을 유도.

### Insight 5: 정규화의 부재가 공정한 비교를 가로막음

동일 벤치마크에서도 모델별 출력 형식(HTML/Markdown/PaddleOCR 태그)에 따라 전처리 필요 수준이 다름. 벤치마크 자체에 표준화된 정규화 파이프라인이 없으면, 각 연구팀의 전처리 품질이 점수에 반영됨 → 재현성 저해.
