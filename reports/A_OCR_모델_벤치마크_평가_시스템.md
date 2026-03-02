# Epic A: OCR 모델 벤치마크 평가 시스템

## 1. 배경 및 목적

GLM-OCR, PaddleOCR-VL, DeepSeek-OCR2 등 최신 OCR/VLM 모델이 빠르게 등장하고 있으나, 동일 조건에서의 정량적 비교 체계가 부재. 모델 선정 근거를 마련하기 위한 벤치마크 프레임워크 구축 및 전체 평가를 수행.

## 2. 평가 프레임워크 구축 (A-1)

### 2.1 설계 원칙

- **추론/평가 완전 분리**: `infer.py`(추론)와 `eval_bench.py`(평가)를 독립 실행. 추론 결과를 텍스트 파일로 저장하여 재평가 가능.
- **Resume 지원**: 이미 완료된 샘플은 자동 건너뜀. 중단 후 재시작 시 안전.
- **공식 프로토콜 준수**: OmniDocBench는 공식 `pdf_validation.py`, TEDS는 IBM PubTabNet 구현 사용.

### 2.2 평가 대상 모델 (6개)

| 구분 | 모델 | 백엔드 |
|------|------|--------|
| Direct VLM | GLM-OCR | vLLM |
| Direct VLM | PaddleOCR-VL | vLLM |
| Direct VLM | DeepSeek-OCR2 | vLLM |
| Pipeline | GLM-OCR-Pipeline | GLM-OCR SDK + vLLM |
| Pipeline | PaddleOCR-VL-Pipeline | PaddleX 내장 |
| Pipeline | MinerU-2.5 | 자체 파이프라인 |

### 2.3 벤치마크 (8개)

| 벤치마크 | 카테고리 | 메트릭 | 샘플 수 |
|----------|----------|--------|---------|
| OmniDocBench | 문서 파싱 | Overall (Text ED + Table TEDS + Formula CDM) | 1,358 |
| DP-Bench | 문서 파싱 | NID, TEDS, TEDS-struct | ~200 |
| OCRBench | 텍스트 인식 | Accuracy (substring match) | 1,000 |
| IAM Handwritten | 손글씨 | CER, WER | 200 |
| PubTabNet | 테이블 인식 | TEDS, TEDS-struct | 200 |
| TEDS_TEST | 테이블 인식 | TEDS, TEDS-struct | 200 |
| UniMERNet | 수식 인식 | CDM F1, Edit Distance, BLEU | 200 |
| Nanonets-KIE | 정보 추출 | ANLS | 987/147 |

### 2.4 코드 구조

```
/root/ocr_test/
├── config.py          # 모델/벤치마크 설정, vLLM 파라미터
├── client.py          # 통합 inference client (VLLMOCRClient, GLMOCRPipelineClient 등)
├── server.py          # vLLM 서버 라이프사이클 관리 (자동 시작/종료)
├── infer.py           # 추론 실행 (resume 지원)
├── eval_bench.py      # 평가 실행 (모든 메트릭 통합)
├── benchmarks.py      # 벤치마크별 평가기
├── metrics.py         # 메트릭 구현 (TEDS, CDM, NID, ANLS, CER, WER)
├── prompts.py         # 벤치마크별/모델별 프롬프트 매핑
└── datasets_loader.py # HuggingFace 데이터셋 자동 다운로드 + 전처리
```

## 3. 전체 추론 실행 및 결과 (A-2)

### 3.1 Direct VLM 결과

| 벤치마크 | 메트릭 | 샘플 수 | GLM-OCR | PaddleOCR-VL | DeepSeek-OCR2 |
|----------|--------|---------|---------|--------------|---------------|
| OCRBench | Accuracy ↑ | 1,000 | **83.8%** | 71.3% | 48.4% |
| PubTabNet | TEDS ↑ | 200 | **86.1%** | 84.8% | 84.6% |
| PubTabNet | TEDS-struct ↑ | 200 | **92.3%** | 91.0% | 90.7% |
| TEDS_TEST | TEDS ↑ | 200 | **84.4%** | 84.4% | 81.9% |
| IAM | CER ↓ | 200 | **3.9%** | 4.1% | 18.5% |
| UniMERNet | CDM F1 ↑ | 200 | 94.1% | **96.9%** | 78.7% |

### 3.2 Pipeline 결과

| 벤치마크 | 메트릭 | GLM-OCR-P | PaddleOCR-VL-P | DeepSeek-OCR2 |
|----------|--------|-----------|----------------|---------------|
| OmniDocBench | Overall ↑ | 91.9 | 91.9 | 79.6 |
| Nanonets-KIE | ANLS ↑ | 58.4% | 80.8% | **84.0%** |
| DP-Bench | NID ↑ | 87.6% | 87.4% | **90.1%** |
| DP-Bench | TEDS ↑ | 86.7% | **95.2%** | 83.0% |

### 3.3 모델별 강점/약점

- **GLM-OCR**: OCR 정확도 1위(83.8%), 테이블 1위(TEDS 86.1%), 손글씨 1위(CER 3.9%). 수식에서 PaddleOCR-VL 대비 약간 뒤처짐.
- **PaddleOCR-VL**: 수식 인식 1위(CDM 96.9%), DP-Bench 테이블 1위(TEDS 95.2%).
- **DeepSeek-OCR2**: KIE 1위(ANLS 84.0%), DP-Bench NID 1위(90.1%). OCRBench/수식/손글씨 저조.

## 4. 평가 프로토콜 정합성 수정 — 15건 (A-3)

벤치마크 평가 시 프로토콜 불일치로 인한 점수 왜곡을 발견. 모델 변경 없이 eval 코드만 보정.

### 4.1 주요 수정 내역

| 영역 | 수정 내용 | 효과 |
|------|----------|------|
| **TEDS** | whitespace strip + `<th>`→`<td>` + lxml 정규화 | **45%→86% (+41pt)** |
| **TEDS** | DeepSeek 공백 구분 table → HTML 변환 | **+20pt** |
| **CDM** | xeCJK 제거, `\mathcolor` alias, ImageMagick fallback | CDM 정상 동작 |
| **KIE** | 한국어/인도 문서 regex 전면 재구현 | **0%→55%+** |
| **NID** | 공식 eval 프로토콜 맞춤, markdown strip 개선 | **+6~11pt** |

### 4.2 수정 원칙

1. **모델 출력은 일절 변경하지 않음** — eval 코드와 전처리만 수정
2. **공식 프로토콜 우선** — 각 벤치마크의 공식 평가 방법을 기준으로 정합성 확보
3. **전수 검증** — 수정 전후 전체 샘플의 점수 변화를 확인

## 5. 모델 아키텍처 비교 분석 (A-4)

### 5.1 3세대 OCR 아키텍처 분류

```
1세대: 전통 파이프라인 (단계별 전문 모델)
  └─ MinerU 1.x, Upstage Document Parse, Azure Document Intelligence

2세대: 하이브리드 (레이아웃 분리 + VLM 인식 통합)
  └─ MinerU 2.5+, GLM-OCR, PaddleOCR-VL  ← 현재 벤치마크 최고 성능

3세대: End-to-End (단일 모델)
  └─ DeepSeek-OCR 2  ← 아키텍처적으로 가장 진보, 정확도는 아직 하이브리드에 미달
```

### 5.2 종합 비교

| | GLM-OCR | PaddleOCR-VL | DeepSeek-OCR 2 |
|---|---------|-------------|----------------|
| 구조 | 하이브리드 | 하이브리드 | End-to-End |
| 총 파라미터 | 0.9B | 0.9B | 3B (500M 활성) |
| 비전 토큰 | ~6K | ~6K | **≤1,120** |
| OmniDocBench | **94.62** | 94.5 | 91.09 |

### 5.3 주요 관찰

1. **하이브리드가 현재 최적**: 레이아웃 감지만 분리하고 나머지를 VLM으로 통합하는 방식이 정확도/효율 균형점
2. **VLM 크기 0.9B 수렴**: GLM-OCR, PaddleOCR-VL 모두 0.9B — 문서 OCR에 최적화된 모델 크기
3. **End-to-End의 잠재력**: DeepSeek-OCR 2의 Visual Causal Flow는 혁신적이나, 하이브리드 대비 3~4점 차이

## 6. 산출물

- `config.py`, `client.py`, `server.py`, `infer.py`, `eval_bench.py`, `benchmarks.py`, `metrics.py`, `prompts.py`
- `predictions/`: 모델별 추론 결과
- `results/`: 평가 결과 JSON
- `RESULTS_SUMMARY.md`, `OCR_Model_Architecture_Comparison.md`
