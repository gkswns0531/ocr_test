# Epic D: 멀티모달 임베딩 실험

## 1. 배경 및 목적

OCR 텍스트 기반 검색은 시각적 정보(차트, 다이어그램, 지도 등)가 포함된 쿼리에서 성능이 저하. 멀티모달 임베딩(Qwen3-VL), 텍스트 전용 베이스라인(BGE-M3), GPT-4o 캡션 등 다양한 검색 전략을 실험하여 최적의 visual document retrieval 파이프라인을 탐색.

## 2. 임베딩 파이프라인 구축 (D-1)

### 2.1 임베딩 모델

| 모델 | Dimension | 양자화 | 특징 |
|------|-----------|--------|------|
| Qwen3-VL-Embedding-2B-FP8 | 2,048 | FP8 | 멀티모달 (이미지+텍스트) |
| Qwen3-VL-Embedding-8B-FP8 | 4,096 | FP8 | 멀티모달 (이미지+텍스트) |
| BAAI/bge-m3 | 1,024 | — | 텍스트 전용, 다국어 |

### 2.2 임베딩 유형 (3종)

| 유형 | 설명 | 건수 |
|------|------|------|
| **text** | OCR 전체 텍스트 | 40,781 pages |
| **region** | 멀티모달 (crop 이미지 + 캡션 텍스트) | 21,052 regions |

### 2.3 산출물 크기

| 모델 | corpus_text | corpus_regions | queries |
|------|----------------|----------------|---------|
| 2B | (40781, 2048) 637MB | (21052, 2048) 329MB | (600, 2048) |
| 8B | (40781, 4096) 1.27GB | (21052, 4096) 658MB | (600, 4096) |

## 3. Region crop 버그 수정 및 재계산 (D-2)

### 3.1 문제

crop 이미지 경로 매칭 로직에서 `page_id` 기반이 아닌 파일명 패턴 매칭을 사용하여, 21,052개 region 중 **40.3%가 잘못된 이미지와 매칭**. 멀티모달 임베딩이 엉뚱한 이미지를 인코딩.

### 3.2 수정

crop 경로 매칭 로직을 `page_id` 기반 정확한 매칭으로 수정 후, 전체 region 임베딩 재계산.

### 3.3 수정 효과 (Qwen3-VL-8B, Region standalone, true_visual 40건)

| R@K | Before (buggy) | After (corrected) | Δ |
|-----|:-:|:-:|:-:|
| R@1 | 40.0% | **62.5%** | **+22.5%p** |
| R@5 | 47.5% | **75.0%** | **+27.5%p** |
| R@10 | 55.0% | **75.0%** | **+20.0%p** |
| R@20 | 57.5% | **77.5%** | **+20.0%p** |

**교훈**: 데이터 정합성이 모델 성능보다 중요한 전형적 사례. E-2의 HTML 리포트를 통해 시각적으로 버그를 발견.

## 4. GPT-4o 캡션 생성 + 캡션 임베딩 (D-3)

### 4.1 캡션 생성

| 항목 | 값 |
|------|-----|
| 모델 | GPT-4o (vision API) |
| 프롬프트 | 12-section 한국어 프롬프트 |
| 대상 | 21,052 image/chart regions |
| 비용 | **$22** |
| 산출물 | `region_captions.jsonl` (54.9MB) |

프롬프트는 이미지 유형 분류, 구성 요소 설명, 수치 데이터 추출, 맥락 요약, 키워드 목록 등 12개 섹션으로 구성.

### 4.2 3방식 standalone 비교 (Qwen3-VL-8B, true_visual 40건)

| R@K | text | region | caption |
|-----|-----------|--------|---------|
| R@1 | 17.5% | **62.5%** | 25.0% |
| R@5 | 30.0% | **75.0%** | 37.5% |
| R@10 | 35.0% | **75.0%** | 40.0% |
| R@20 | 40.0% | **77.5%** | 40.0% |

Region이 압도적 1위. Caption은 Region 대비 낮지만 text보다 우위.

### 4.3 Region vs Caption 역할 분석

- **Region**: 이미지 자체의 visual feature → cross-modal matching. 구조도/프로세스 다이어그램에서 강점.
- **Caption**: 이미지 내용을 텍스트로 설명 → text-to-text matching. 지도/전략도/수치차트에서 강점.
- Region만 Hit 14건(35%), Caption만 Hit 3건(7.5%) → 서로 보완적.

## 5. BGE-M3 베이스라인 + 3×3 최종 평가 (D-4)

### 5.1 전체 600 쿼리: text+region (Qwen3-VL-2B vs 8B)

| R@K | Qwen3-VL-2B | Qwen3-VL-8B |
|-----|:-:|:-:|
| R@1 | 40.7% | **45.2%** |
| R@5 | 67.8% | **75.0%** |
| R@10 | 75.5% | **80.5%** |
| R@20 | 81.8% | **84.3%** |

### 5.2 쿼리 타입별 R@10 (text+region, Qwen3-VL-2B vs 8B)

| Type | n | Qwen3-VL-2B | Qwen3-VL-8B |
|------|---|:-:|:-:|
| cross | 336 | 76.8% | **82.7%** |
| text | 103 | 83.5% | **85.4%** |
| visual | 161 | 67.1% | **73.3%** |
| **Total** | **600** | **75.5%** | **80.5%** |

### 5.3 true_visual 40건: 3방식 standalone (Qwen3-VL-8B vs BGE-M3)

**Region standalone**

| R@K | Qwen3-VL-2B | Qwen3-VL-8B | BGE-M3 |
|-----|:-:|:-:|:-:|
| R@1 | 47.5% | **62.5%** | 32.5% |
| R@5 | 70.0% | **75.0%** | 37.5% |
| R@10 | 72.5% | **75.0%** | 47.5% |
| R@20 | 82.5% | **77.5%** | 47.5% |

멀티모달 임베딩이 텍스트 전용 대비 **+27.5%p** 우위 (8B 기준).

**Caption standalone**

| R@K | Qwen3-VL-2B | Qwen3-VL-8B | BGE-M3 |
|-----|:-:|:-:|:-:|
| R@1 | 17.5% | **25.0%** | 7.5% |
| R@5 | 27.5% | **37.5%** | 20.0% |
| R@10 | 32.5% | **40.0%** | 27.5% |
| R@20 | 40.0% | **40.0%** | 30.0% |

### 5.4 모델 크기 효과 (2B vs 8B)

| Metric | 2B | 8B | Δ |
|--------|:-:|:-:|:-:|
| 전체 R@10 (text+region) | 75.5% | **80.5%** | +5.0%p |
| true_visual R@10 (region) | 72.5% | **75.0%** | +2.5%p |

8B가 2B 대비 전 구간에서 우위. true_visual region에서도 격차 확인.

## 6. 산출물

- `run_b200_pipeline.py`, `run_8b_embeddings.py`, `run_comparison_embeddings.py`: 임베딩 파이프라인
- `recompute_region_embeddings.py`: Region 임베딩 재계산
- `compute_caption_embeddings_v11.py`: 캡션 임베딩 계산
- `compute_bge_m3_embeddings.py`: BGE-M3 임베딩
- `eval_final_comparison.py`: 3×3 최종 평가
- `output_dl/region_captions.jsonl` (54.9MB): 21,052건 캡션
- `output_dl/embeddings/`, `output_dl/embeddings_8b/`, `output_dl/embeddings_bge_m3/`: 임베딩 파일
