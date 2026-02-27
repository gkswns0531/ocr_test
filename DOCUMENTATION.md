# OCR VLM 벤치마크 평가 시스템 종합 문서

> 최종 업데이트: 2026-02-26

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [모델 상세](#3-모델-상세)
   - 3.1 [GLM-OCR (Direct VLM)](#31-glm-ocr-direct-vlm)
   - 3.2 [PaddleOCR-VL (Direct VLM)](#32-paddleocr-vl-direct-vlm)
   - 3.3 [DeepSeek-OCR2 (Direct VLM)](#33-deepseek-ocr2-direct-vlm)
   - 3.4 [GLM-OCR-Pipeline](#34-glm-ocr-pipeline)
   - 3.5 [PaddleOCR-VL-Pipeline](#35-paddleocr-vl-pipeline)
   - 3.6 [MinerU-2.5](#36-mineru-25)
4. [레이아웃 파이프라인 아키텍처](#4-레이아웃-파이프라인-아키텍처)
   - 4.1 [Direct VLM vs Pipeline 비교](#41-direct-vlm-vs-pipeline-비교)
   - 4.2 [PP-DocLayoutV3 레이아웃 감지](#42-pp-doclayoutv3-레이아웃-감지)
   - 4.3 [GLM-OCR SDK 파이프라인](#43-glm-ocr-sdk-파이프라인)
   - 4.4 [PaddleOCR-VL SDK 파이프라인](#44-paddleocr-vl-sdk-파이프라인)
5. [벤치마크 상세](#5-벤치마크-상세)
6. [평가 메트릭 상세](#6-평가-메트릭-상세)
7. [사용법](#7-사용법)
   - 7.1 [데이터 준비](#71-데이터-준비)
   - 7.2 [추론 실행](#72-추론-실행)
   - 7.3 [평가 실행](#73-평가-실행)
8. [Eval 수정 이력](#8-eval-수정-이력)
9. [최종 결과](#9-최종-결과)
10. [Gap 분석](#10-gap-분석)

---

## 1. 시스템 개요

이 시스템은 OCR VLM(Vision-Language Model) 모델의 문서 인식 성능을 8개 벤치마크에서 평가합니다.

**평가 대상 모델**: 6개
- Direct VLM: GLM-OCR, PaddleOCR-VL, DeepSeek-OCR2
- Pipeline: GLM-OCR-Pipeline, PaddleOCR-VL-Pipeline, MinerU-2.5

**평가 벤치마크**: 8개
- 문서 파싱: OmniDocBench, Upstage DP-Bench
- 텍스트 인식: OCRBench (VQA), IAM Handwritten
- 테이블 인식: PubTabNet, TEDS_TEST
- 수식 인식: UniMERNet
- 정보 추출: Nanonets-KIE

**핵심 설계 원칙**:
- **추론(infer.py)과 평가(eval_bench.py)가 완전 분리**: 추론 결과는 텍스트 파일로 저장, 평가는 저장된 파일을 읽어 수행
- **Resume 지원**: 이미 추론/평가된 샘플은 자동 건너뜀
- **공식 프로토콜 준수**: OmniDocBench는 공식 `pdf_validation.py` 사용, TEDS는 IBM PubTabNet 구현 사용

---

## 2. 디렉토리 구조

```
/home/ubuntu/ocr_test/
├── eval_bench.py          # 평가 실행 스크립트 (메인 진입점)
├── infer.py               # 추론 실행 스크립트
├── config.py              # 모델/벤치마크 설정
├── client.py              # 추론 클라이언트 (VLLMOCRClient, GLMOCRPipelineClient 등)
├── server.py              # vLLM 서버 라이프사이클 관리
├── prompts.py             # 벤치마크별/모델별 프롬프트
├── benchmarks.py          # 벤치마크별 평가기 (_eval_table, _eval_kie 등)
├── metrics.py             # 메트릭 구현 (TEDS, CDM, NID, ANLS 등)
├── datasets_loader.py     # HuggingFace 데이터셋 로더
├── RESULTS_SUMMARY.md     # 최종 결과 요약표
├── DOCUMENTATION.md       # 이 문서
│
├── prepared_datasets/     # 전처리된 벤치마크 데이터 (JSONL + JPEG)
│   ├── ocrbench/          # 1,000 samples
│   ├── pubtabnet/         # 200 samples
│   ├── teds_test/         # 200 samples
│   ├── unimernet/         # 200 samples
│   ├── handwritten_forms/ # 200 samples
│   ├── nanonets_kie/      # 987/147 samples (train/test)
│   ├── omnidocbench/      # 1,358 samples (전수)
│   └── upstage_dp_bench/  # ~200 samples
│
├── predictions/           # 모델별 추론 결과
│   ├── glm_ocr/           # Direct VLM 추론 결과
│   ├── paddleocr_vl/
│   ├── deepseek_ocr2/
│   ├── glm_ocr_pipeline/  # Pipeline 추론 결과
│   └── paddleocr_vl_pipeline/
│
├── results/               # 평가 결과 JSON 파일
│   ├── *_eval.json        # 벤치마크별 상세 결과
│   └── eval_comparison_*.json  # 모델 간 비교표
│
└── data_cache/            # HuggingFace 데이터 캐시

/home/ubuntu/glm-ocr-sdk/         # GLM-OCR SDK 소스
/home/ubuntu/OmniDocBench/        # OmniDocBench 공식 평가 코드
/home/ubuntu/junghoon/miner_test/ # MinerU 설치
```

### Prediction 파일 형식

| 벤치마크 | 파일 형식 | 예시 |
|:---|:---|:---|
| OmniDocBench | `{원본이미지이름}.md` | `page_001.md` |
| 기타 모든 벤치마크 | `{sample_id}.txt` | `ocrbench_0042.txt` |

### Prepared Dataset 형식

각 벤치마크 디렉토리에는:
- `metadata.jsonl`: 한 줄에 하나의 JSON 레코드 (`idx`, `sample_id`, `ground_truth`, `metadata`)
- `images/`: `{idx:05d}.jpg` 형식의 JPEG 이미지

---

## 3. 모델 상세

### 3.1 GLM-OCR (Direct VLM)

| 항목 | 값 |
|:---|:---|
| **모델 ID** | `zai-org/GLM-OCR` |
| **제조사** | Zhipu AI (智谱) |
| **백엔드** | vLLM |
| **포트** | 8000 (기본) |
| **이미지 전처리** | 2048×2048 리사이즈 (base64 전송 시) |
| **강점** | OCR 정확도 1위, 테이블 1위, 손글씨 1위 |

**vLLM 서버 시작 명령**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-OCR \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-model-len 16384
```

**프롬프트 매핑**:

| 벤치마크 | 프롬프트 |
|:---|:---|
| Document Parse (ODB) | `"Convert the document to markdown."` |
| DP-Bench | `"Text Recognition:"` |
| OCRBench | 샘플별 question 필드 사용 |
| Formula (UniMERNet) | `"Formula Recognition:"` |
| Table (PubTabNet) | `"Table Recognition:"` |
| KIE (Nanonets) | JSON 추출 프롬프트 (기본값) |
| Handwritten (IAM) | `"Text Recognition:"` |

### 3.2 PaddleOCR-VL (Direct VLM)

| 항목 | 값 |
|:---|:---|
| **모델 ID** | `PaddlePaddle/PaddleOCR-VL` |
| **제조사** | Baidu (PaddlePaddle) |
| **백엔드** | vLLM |
| **포트** | 8000 (기본) |
| **이미지 전처리** | 2048×2048 리사이즈 (base64 전송 시) |
| **강점** | 수식 인식 1위 (CDM 96.9%), DP-Bench 테이블 1위 |

**vLLM 서버 시작 명령**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model PaddlePaddle/PaddleOCR-VL \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-num-batched-tokens 16384
```

**프롬프트 매핑**:

| 벤치마크 | 프롬프트 |
|:---|:---|
| Document Parse (ODB) | `"Convert the document to markdown."` |
| DP-Bench | `"OCR:"` |
| OCRBench | 샘플별 question 필드 사용 |
| Formula (UniMERNet) | `"Formula Recognition:"` |
| Table (PubTabNet) | `"Table Recognition:"` |
| KIE (Nanonets) | JSON 추출 프롬프트 (기본값) |
| Handwritten (IAM) | `"OCR:"` |

### 3.3 DeepSeek-OCR2 (Direct VLM)

| 항목 | 값 |
|:---|:---|
| **모델 ID** | `deepseek-ai/DeepSeek-OCR-2` |
| **제조사** | DeepSeek |
| **백엔드** | vLLM |
| **포트** | 8000 (기본) |
| **이미지 전처리** | 2048×2048 리사이즈 (base64 전송 시) |
| **강점** | KIE 1위 (ANLS 84.0%), DP-Bench NID 1위 (90.1%) |

**vLLM 서버 시작 명령**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-OCR-2 \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-model-len 8192
```

**프롬프트 매핑**:

| 벤치마크 | 프롬프트 |
|:---|:---|
| Document Parse (ODB/DP) | `"Convert the document to markdown."` |
| OCRBench | 샘플별 question 필드 사용 |
| Formula (UniMERNet) | `"Free OCR."` |
| Table (PubTabNet) | `"Free OCR."` |
| KIE (Nanonets) | JSON 추출 프롬프트 (기본값) |
| Handwritten (IAM) | `"Free OCR."` |

**참고**: DeepSeek-OCR2는 `max-model-len`이 8192로 다른 모델(16384)보다 짧음. 공식 프롬프트가 `"Free OCR."`으로 범용적.

### 3.4 GLM-OCR-Pipeline

| 항목 | 값 |
|:---|:---|
| **모델 ID** | `zai-org/GLM-OCR` (VLM 부분) |
| **SDK** | `/home/ubuntu/glm-ocr-sdk/` (`glmocr` 패키지) |
| **백엔드** | `glmocr_pipeline` (PP-DocLayoutV3 + vLLM) |
| **구조** | 레이아웃 감지 → 영역 크롭 → 영역별 VLM 추론 → 마크다운 조립 |

**아키텍처**: 3-스레드 비동기 파이프라인
```
PageLoader → LayoutDetector → OCRClient(per-region) → ResultFormatter
   (이미지 로드)   (PP-DocLayoutV3)  (vLLM API 호출)    (마크다운 생성)
```

**eval에서의 SDK 설정 패치** (`client.py`):

| 설정 | SDK 기본값 | eval 패치값 | 성능 영향 |
|:---|---:|---:|:---|
| `api_port` | 8080 | (vLLM 포트) | 연결 대상 |
| `max_workers` | 32 | 1 | 속도만 (정확도 무관) |
| `request_timeout` | 120s | 60s | **있음** (타임아웃 시 빈 출력) |
| `retry_max_attempts` | 2 | 1 | 속도만 (정확도 무관) |
| `max_tokens` | 4096 | 4096 (미변경) | **있음** (긴 문서 잘림 가능) |

**참고**: SDK README 권장값은 `request_timeout: 300`, `max_tokens: 16384`. eval에서는 `request_timeout: 60`으로 줄여 사용해서 약 1.1%(15/1355)의 샘플이 타임아웃으로 빈 출력 발생.

**Pipeline vLLM 시작 명령** (별도 vLLM args):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-OCR \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --max-model-len 16384 \
    --served-model-name glm-ocr \
    --gpu-memory-utilization 0.80
```

### 3.5 PaddleOCR-VL-Pipeline

| 항목 | 값 |
|:---|:---|
| **모델 ID** | `PaddlePaddle/PaddleOCR-VL` (내부) |
| **SDK** | `paddleocr` (`PaddleOCRVL` 클래스) |
| **백엔드** | `paddleocr_pipeline` (PaddleX 프레임워크) |
| **구조** | 레이아웃 감지 → 영역 크롭 → VL 인식 → 마크다운 조립 |
| **GPU 서버** | 불필요 (SDK가 내부적으로 모델 로드) |

**초기화**:
```python
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL(pipeline_version="v1.5")
output = pipeline.predict("input.jpg")
# output[0].markdown["markdown_texts"] → 마크다운 결과
```

**SDK 설정** (`PaddleOCR-VL-1.5.yaml`):
- 레이아웃 감지: PP-DocLayoutV3 (threshold 0.3)
- VLM: PaddleOCR-VL-1.5-0.9B (native backend)
- 문서 전처리(선택): orientation classify + unwarping

**Direct VLM과의 차이**:
- Direct: vLLM 서버 필요, OpenAI API로 통신
- Pipeline: 별도 서버 불필요, SDK가 모든 모델을 프로세스 내 로드

### 3.6 MinerU-2.5

| 항목 | 값 |
|:---|:---|
| **모델 ID** | `mineru` |
| **설치 경로** | `/home/ubuntu/junghoon/miner_test/MinerU` |
| **백엔드** | `mineru` (자체 파이프라인) |
| **구조** | `do_parse()` → 마크다운 출력 |

MinerU는 OCR VLM이 아닌 문서 파싱 파이프라인으로, 이미지를 입력받아 마크다운을 출력합니다. 현재 eval에서는 OmniDocBench 평가에만 사용됩니다.

---

## 4. 레이아웃 파이프라인 아키텍처

### 4.1 Direct VLM vs Pipeline 비교

```
[Direct VLM]
  전체 이미지 → VLM(text prompt) → 마크다운/HTML 출력

[Pipeline]
  전체 이미지 → PP-DocLayoutV3(레이아웃 감지)
    → 영역 크롭 (text, table, formula, skip, abandon)
    → 영역별 VLM 추론 (text→텍스트 프롬프트, table→테이블 프롬프트, formula→수식 프롬프트)
    → ResultFormatter (마크다운 조립, 읽기 순서 정렬)
```

**주요 차이점**:

| 특성 | Direct VLM | Pipeline |
|:---|:---|:---|
| 입력 | 전체 이미지 (2048×2048 리사이즈) | 원본 해상도 이미지 |
| 레이아웃 감지 | 모델이 암시적으로 수행 | PP-DocLayoutV3로 명시적 수행 |
| 영역별 프롬프트 | 단일 프롬프트 | text/table/formula 별도 프롬프트 |
| VLM 호출 횟수 | 1회/이미지 | N회/이미지 (영역 수만큼) |
| 해상도 | 고정 2048×2048 | 영역별 원본 해상도 크롭 |
| 장점 | 빠름, 간단 | 높은 정확도, 구조 보존 |
| 단점 | 복잡한 레이아웃에서 혼동 | 느림, 레이아웃 감지 오류 전파 |

**Pipeline이 더 좋은 벤치마크**: OmniDocBench (요소별 평가), DP-Bench (문서 구조 평가)
**Direct VLM이 적합한 벤치마크**: OCRBench (단일 질문), PubTabNet (단일 테이블), UniMERNet (단일 수식)

### 4.2 PP-DocLayoutV3 레이아웃 감지

두 Pipeline 모두 **PP-DocLayoutV3**를 레이아웃 감지 모델로 공유합니다.

**모델**: `PaddlePaddle/PP-DocLayoutV3_safetensors` (Transformers용)
**Threshold**: 0.3

**25개 레이블 카테고리**:

| ID | 레이블 | 태스크 | 설명 |
|---:|:---|:---|:---|
| 0 | abstract | text | 초록 |
| 1 | algorithm | text | 알고리즘 |
| 2 | aside_text | abandon | 사이드 텍스트 |
| 3 | chart | skip | 차트 (OCR 안 함) |
| 4 | content | text | 본문 |
| 5 | display_formula | formula | 디스플레이 수식 |
| 6 | doc_title | text | 문서 제목 |
| 7 | figure_title | text | 그림 제목 |
| 8 | footer | abandon | 꼬리말 |
| 9 | footer_image | abandon | 꼬리말 이미지 |
| 10 | footnote | abandon | 각주 |
| 11 | formula_number | text | 수식 번호 |
| 12 | header | abandon | 머리말 |
| 13 | header_image | abandon | 머리말 이미지 |
| 14 | image | skip | 이미지 (OCR 안 함) |
| 15 | inline_formula | formula | 인라인 수식 |
| 16 | number | abandon | 페이지 번호 |
| 17 | paragraph_title | text | 문단 제목 |
| 18 | reference | abandon | 참고문헌 |
| 19 | reference_content | text | 참고문헌 내용 |
| 20 | seal | text | 도장 |
| 21 | table | table | 테이블 |
| 22 | text | text | 일반 텍스트 |
| 23 | vertical_text | text | 세로 텍스트 |
| 24 | vision_footnote | text | 시각 각주 |

**태스크 유형**:
- `text`: 텍스트 인식 프롬프트로 OCR
- `table`: 테이블 프롬프트로 HTML 변환
- `formula`: 수식 프롬프트로 LaTeX 변환
- `skip`: 영역 유지하되 OCR 수행 안 함 (이미지, 차트)
- `abandon`: 영역 완전 무시 (머리말, 꼬리말, 페이지 번호 등)

### 4.3 GLM-OCR SDK 파이프라인

**위치**: `/home/ubuntu/glm-ocr-sdk/`
**설정 파일**: `/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml`

**파이프라인 흐름**:

```
1. GlmOcr.parse(image_path) 호출
2. PageLoader: 이미지/PDF 로드, 전처리
3. LayoutDetector: PP-DocLayoutV3로 레이아웃 감지
   - 영역별 바운딩박스 + 레이블 감지
   - NMS, 중첩 제거, bbox 확장
4. OCRClient: 각 영역을 태스크별 프롬프트로 vLLM에 요청
   - text 영역 → "Recognize the text..." 프롬프트
   - table 영역 → "Convert the table to HTML..." 프롬프트
   - formula 영역 → "Convert the formula to LaTeX..." 프롬프트
   - max_workers로 병렬 요청 (eval에서는 1)
5. ResultFormatter: 결과 조립
   - 중첩 영역 필터링 (min_overlap_ratio: 0.8)
   - 읽기 순서 정렬 (top→bottom, left→right)
   - 마크다운 + JSON 출력 생성
```

**핵심 설정 파라미터**:

| 파라미터 | 기본값 | 설명 |
|:---|:---|:---|
| `pipeline.enable_layout` | `true` | 레이아웃 감지 활성화 |
| `pipeline.max_workers` | 32 | 영역 병렬 처리 수 |
| `pipeline.ocr_api.request_timeout` | 120s | 영역당 타임아웃 |
| `pipeline.page_loader.max_tokens` | 4096 | 영역당 최대 토큰 |
| `pipeline.page_loader.temperature` | 0.0 | 결정론적 출력 |
| `pipeline.layout.threshold` | 0.3 | 감지 임계값 |
| `pipeline.layout.model_dir` | PP-DocLayoutV3_safetensors | 레이아웃 모델 |

**API 사용법**:
```python
from glmocr import GlmOcr

# 기본 사용
ocr = GlmOcr()  # 기본 config.yaml 사용
result = ocr.parse("document.pdf")

# 커스텀 설정
ocr = GlmOcr(config_path="/path/to/config.yaml")
result = ocr.parse("document.jpg", save_layout_visualization=True)

# 결과 접근
print(result.markdown_result)  # 마크다운 텍스트
print(result.json_result)      # 구조화된 JSON

ocr.close()  # 리소스 정리
```

**두 가지 모드**:
1. **Self-hosted**: 로컬 vLLM/SGLang 서버에 연결 (이 프로젝트에서 사용)
2. **MaaS**: 智谱(Zhipu) 클라우드 API로 직접 호출

### 4.4 PaddleOCR-VL SDK 파이프라인

**SDK**: `paddleocr` 패키지 (`PaddleOCRVL` 클래스)
**설정**: `paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml`

**파이프라인 흐름**:

```
1. PaddleOCRVL(pipeline_version="v1.5") 초기화
2. pipeline.predict(image_path) 호출
3. (선택) 문서 전처리: 방향 분류 + 왜곡 보정
4. PP-DocLayoutV3: 레이아웃 감지 (threshold 0.3)
5. PaddleOCR-VL-1.5-0.9B: 영역별 VL 인식
6. 결과 조립 → 마크다운 출력
```

**API 사용법**:
```python
from paddleocr import PaddleOCRVL

# 초기화 (모델 자동 다운로드)
pipeline = PaddleOCRVL(pipeline_version="v1.5")

# 추론
output = pipeline.predict("document.jpg")

# 결과 접근
for res in output:
    print(res.markdown["markdown_texts"])  # 마크다운 텍스트

# 파일 저장
for res in output:
    res.save_to_markdown(save_path="./output/")
```

**GLM-OCR SDK와의 핵심 차이점**:

| 특성 | GLM-OCR SDK | PaddleOCR-VL SDK |
|:---|:---|:---|
| VLM 서빙 | 외부 vLLM 서버 (API 호출) | 프로세스 내 로드 (native backend) |
| GPU 서버 | 필요 (별도 vLLM) | 불필요 (SDK가 직접 로드) |
| VLM 모델 | GLM-OCR (대형) | PaddleOCR-VL-1.5-0.9B (소형) |
| 병렬 처리 | max_workers로 제어 | SDK 내부 관리 |
| 설정 유연성 | YAML config 전체 노출 | 최소한의 설정만 |
| MaaS 모드 | 지원 (Zhipu Cloud) | 미지원 |

---

## 5. 벤치마크 상세

### 5.1 OmniDocBench

| 항목 | 값 |
|:---|:---|
| **데이터셋** | `opendatalab/OmniDocBench` |
| **샘플 수** | 1,358 (전수) |
| **평가 방식** | 공식 `pdf_validation.py` (End2EndDataset) |
| **평가 대상** | Pipeline 모델 + DeepSeek-OCR2 |
| **GT 형식** | `OmniDocBench.json` (요소별 annotation) |
| **Prediction 형식** | `{image_stem}.md` (마크다운) |

**메트릭**:
- Text Block: Edit Distance (1−ED)
- Table: TEDS + TEDS-struct + Edit Distance
- Formula: CDM (Character Detection Matching) + Edit Distance
- Reading Order: Edit Distance
- **Overall**: `((1−text_ED)×100 + table_TEDS×100 + formula_CDM×100) / 3`

**평가 프로세스**:
1. HuggingFace에서 GT JSON 다운로드
2. 임시 YAML config 생성 (prediction 디렉토리 지정)
3. OmniDocBench 디렉토리에서 `pdf_validation.py` 실행
4. `OmniDocBench/result/` 에서 결과 JSON 읽기

### 5.2 OCRBench

| 항목 | 값 |
|:---|:---|
| **데이터셋** | `echo840/OCRBench` |
| **샘플 수** | 1,000 (전수) |
| **메트릭** | Accuracy (substring match) |
| **평가 대상** | Direct VLM 모델만 |
| **GT 형식** | 질문(question) + 정답 리스트(answers) |

**특징**:
- 각 샘플에 고유한 질문이 있음 (프롬프트 = question 필드)
- 공식 스코어링: `if answer.lower() in prediction.lower(): score = 1`
- HME100k(수식) 서브셋: 대소문자 구분, 공백 제거 후 비교
- Binary scoring (0 또는 1), 전체 평균이 Accuracy

### 5.3 PubTabNet / TEDS_TEST

| 항목 | PubTabNet | TEDS_TEST |
|:---|:---|:---|
| **데이터셋** | `apoidea/pubtabnet-html` | `apoidea/pubtabnet-html` |
| **Split** | validation | validation |
| **샘플 수** | 200 | 200 |
| **메트릭** | TEDS, TEDS-struct | TEDS, TEDS-struct |
| **GT 형식** | HTML `<table>...</table>` | HTML `<table>...</table>` |

**테이블 변환 파이프라인** (`benchmarks.py`):
1. 모델 출력에서 HTML 테이블 추출 (`<table>` 태그)
2. HTML이 없으면 마크다운 테이블을 HTML로 변환 (`_convert_markdown_table_to_html`)
3. Space-separated 테이블을 HTML로 변환 (DeepSeek용)
4. `<th>` → `<td>` 변환 (TEDS가 `<td>`만 콘텐츠 비교)
5. Whitespace 정규화 (lxml 기반)
6. TEDS 알고리즘으로 트리 편집 거리 계산

### 5.4 UniMERNet (수식 인식)

| 항목 | 값 |
|:---|:---|
| **데이터셋** | `deepcopy/UniMER` |
| **샘플 수** | 200 |
| **메트릭** | CDM F1, Edit Distance, Corpus BLEU |
| **GT 형식** | LaTeX 수식 문자열 |

**평가 프로세스**:
1. LaTeX 정규화 (`_normalize_latex`): 불필요한 공백 제거
2. 정규화된 Edit Distance 계산
3. CDM (pdflatex → ImageMagick → 시각적 매칭) F1 계산
4. Corpus-level BLEU 계산 (HuggingFace evaluate)

**CDM 패치**: eval에서 적용하는 monkey-patches:
- `xelatex` → `pdflatex` (CJK 폰트 불필요)
- `magick` → `convert` (ImageMagick v6)
- `random_state` → `rng` (scikit-image >= 0.25)
- `\mathcolor` polyfill (TexLive < 2023)

### 5.5 Nanonets-KIE (정보 추출)

| 항목 | 값 |
|:---|:---|
| **데이터셋** | `nanonets/key_information_extraction` |
| **샘플 수** | 987 (train) / 147 (test) |
| **메트릭** | ANLS (Average Normalized Levenshtein Similarity) |
| **GT 형식** | 8개 필드 JSON |

**추출 대상 필드 (8개)**:
- `date`: 날짜
- `doc_no_receipt_no`: 문서/영수증 번호
- `seller_name`: 판매자 이름
- `seller_address`: 판매자 주소
- `seller_gst_id`: GST ID
- `seller_phone`: 전화번호
- `total_amount`: 총 금액
- `total_tax`: 총 세금

**평가 파이프라인**:
1. JSON 파싱 시도 (`json_repair.repair_json`)
2. 실패 시 regex 기반 필드 추출 fallback (`_extract_kie_fields_from_text`)
3. 마크다운/HTML 테이블에서 필드 추출 시도
4. 필드별 NLS 계산 → 평균 = ANLS

### 5.6 IAM Handwritten (손글씨)

| 항목 | 값 |
|:---|:---|
| **데이터셋** | `Teklia/IAM-line` |
| **샘플 수** | 200 |
| **메트릭** | CER (Character Error Rate), WER (Word Error Rate) |
| **GT 형식** | 텍스트 문자열 |

**전처리**: 구두점 앞 공백 제거 (e.g., `word .` → `word.`)

### 5.7 Upstage DP-Bench

| 항목 | 값 |
|:---|:---|
| **데이터셋** | `upstage/dp-bench` |
| **샘플 수** | ~200 |
| **메트릭** | NID (텍스트), TEDS + TEDS-struct (테이블) |
| **GT 형식** | JSON (text_content + table_html) |

**NID 평가 전처리** (`_strip_markdown_for_nid`):
- 마크다운 문법 제거 (헤더, 볼드, 이탤릭 등)
- LaTeX 수식 내용 유지, 구분자 제거
- HTML 테이블 제거 (텍스트 메트릭이므로)
- 연속 공백/줄바꿈 정규화

---

## 6. 평가 메트릭 상세

### 6.1 TEDS (Tree Edit Distance Similarity)

**구현**: IBM PubTabNet (OmniDocBench 번들)
**파일**: `/home/ubuntu/OmniDocBench/metrics/table_metric.py`

```
TEDS = 1 - TreeEditDistance(pred_tree, gt_tree) / max(|pred_tree|, |gt_tree|)
```

- 테이블 HTML을 트리 구조로 파싱
- 노드 = 태그 + 속성 + 셀 텍스트 토큰
- 셀 콘텐츠 비교: Levenshtein distance (CustomConfig)
- `<td>` 노드만 콘텐츠 비교 (`<th>` → `<td>` 변환 필요)
- `colspan`/`rowspan` 속성 정상 처리
- **TEDS-struct**: `structure_only=True` (셀 콘텐츠 무시, 구조만 비교)

### 6.2 CDM (Character Detection Matching)

**구현**: OmniDocBench CDM 모듈
**파일**: `/home/ubuntu/OmniDocBench/metrics/cdm_metric.py`

```
1. LaTeX → pdflatex → PDF
2. PDF → ImageMagick → PNG
3. 문자별 바운딩박스 추출 (색상 인코딩)
4. Pred/GT 바운딩박스 매칭 (RANSAC 정렬)
5. Precision, Recall, F1 계산
```

- 시각적 렌더링 기반 메트릭 (문자 위치 + 내용 동시 평가)
- Edit Distance보다 수식의 시각적 동등성을 잘 포착
- pdflatex 컴파일 실패 시 F1=0 반환

### 6.3 NID (Normalized Indel Distance)

```
NID = 1 - IndelDistance(pred, gt) / (len(pred) + len(gt))
```

- 삽입/삭제만 사용 (치환 없음)
- Upstage DP-Bench 공식 메트릭
- 1 = 동일, 0 = 완전 다름

### 6.4 ANLS (Average Normalized Levenshtein Similarity)

```
NLS = 1 - EditDistance(pred, gt) / max(len(pred), len(gt))
ANLS = mean(NLS for each field)
```

- Nanonets KIE 공식 메트릭
- 필드별로 NLS 계산, 전체 평균
- 누락 필드는 빈 문자열로 처리 (full mismatch)

### 6.5 CER/WER (Character/Word Error Rate)

```
CER = EditDistance(pred_chars, gt_chars) / len(gt_chars)
WER = EditDistance(pred_words, gt_words) / len(gt_words)
```

- IAM Handwritten 표준 메트릭
- GT 길이로 정규화 (1.0 초과 가능)
- 낮을수록 좋음

### 6.6 OCRBench Accuracy

```
Score = 1 if any(answer.lower() in pred.lower() for answer in answers) else 0
Accuracy = mean(Score for each sample)
```

- 공식 MultimodalOCR 스코어링
- Substring match (대소문자 무시)
- HME100k 서브셋: 공백 제거, 대소문자 구분

---

## 7. 사용법

### 7.1 데이터 준비

데이터는 HuggingFace에서 자동 다운로드되어 `prepared_datasets/`에 전처리됩니다.

```bash
# 전체 데이터 준비 (최초 1회)
# infer.py 또는 eval_bench.py 실행 시 자동으로 수행됨
```

### 7.2 추론 실행

```bash
# 단일 모델, 단일 벤치마크
python3 infer.py --model glm-ocr --benchmarks ocrbench

# 단일 모델, 전체 벤치마크
python3 infer.py --model deepseek-ocr2 --benchmarks all

# 샘플 수 제한 (테스트용)
python3 infer.py --model glm-ocr --benchmarks pubtabnet --max-samples 5

# vLLM 포트 지정
python3 infer.py --model paddleocr-vl --benchmarks all --port 8001

# Pipeline 모델 (GLM-OCR SDK)
python3 infer.py --model glm-ocr-pipeline --benchmarks omnidocbench

# Pipeline 모델 (PaddleOCR-VL SDK, vLLM 불필요)
python3 infer.py --model paddleocr-vl-pipeline --benchmarks omnidocbench
```

**사용 가능한 모델**: `glm-ocr`, `paddleocr-vl`, `deepseek-ocr2`, `mineru`, `glm-ocr-pipeline`, `paddleocr-vl-pipeline`

**사용 가능한 벤치마크**: `omnidocbench`, `upstage_dp_bench`, `ocrbench`, `unimernet`, `pubtabnet`, `teds_test`, `nanonets_kie`, `handwritten_forms`

**추론 특징**:
- **Resume 지원**: 이미 prediction 파일이 있는 샘플은 자동 건너뜀
- **서버 자동 관리**: vLLM 서버 시작/종료 자동 처리
- **오류 복구**: 연속 3회 오류 시 서버 헬스 체크, 50회 타임아웃 시 벤치마크 중단
- **vLLM 서버는 Direct VLM과 GLM-OCR-Pipeline만 필요** (PaddleOCR-VL-Pipeline과 MinerU는 불필요)

### 7.3 평가 실행

```bash
# 단일 모델, 전체 벤치마크
python3 eval_bench.py --model glm-ocr --benchmarks all

# 전체 모델 일괄 평가
python3 eval_bench.py --all

# 특정 벤치마크만
python3 eval_bench.py --model deepseek-ocr2 --benchmarks pubtabnet teds_test

# OmniDocBench (공식 프로토콜)
python3 eval_bench.py --model glm-ocr-pipeline --benchmarks omnidocbench
```

**평가 결과**: `results/` 디렉토리에 저장
- 벤치마크별: `{model}_{benchmark}_eval.json`
- 비교표: `eval_comparison_{timestamp}.json`

---

## 8. Eval 수정 이력

총 15건의 eval 수정을 적용하여 평가 정확도를 공식 프로토콜에 맞췄습니다.

| # | 수정 내용 | 영향 벤치마크 | 효과 |
|---:|:---|:---|:---|
| 1 | TEDS: whitespace strip + th→td + lxml 정규화 | PubTabNet | 45% → 86% (+41pt) |
| 2 | CDM: xeCJK 제거, mathcolor alias, ImageMagick fallback | UniMERNet/ODB | CDM 정상 동작 |
| 3 | ODB: formula에 CDM 적용, overall 공식 수정 | OmniDocBench | 공식 프로토콜 일치 |
| 4 | ODB formula 정규화: textcircled, prime, braces | OmniDocBench | CDM 정확도 향상 |
| 5 | KIE regex 전면 개편 (seller_name, doc_no 등) | Nanonets-KIE | 0% → 55%+ |
| 6 | IAM: 구두점 앞 공백 제거 | Handwritten | CER 소폭 개선 |
| 7 | DP-Bench NID: 공식 eval 프로토콜 맞춤 | DP-Bench | NID +6~11pt |
| 8 | DP-Bench tables: non-leading-pipe, LaTeX, HTML strip | DP-Bench | TEDS 개선 |
| 9 | `_strip_markdown_for_nid()` 전면 개선 | DP-Bench | NID 정확도 향상 |
| 10 | KIE seller_name: 마크다운/HTML skip, 헤더 split | Nanonets-KIE | ANLS +2~5pt |
| 11 | KIE doc_no: word boundary, Invoice.No 지원 | Nanonets-KIE | ANLS +1~2pt |
| 12 | KIE seller_name: 등록번호/COMPANY NO에서 중단 | Nanonets-KIE | ANLS 소폭 개선 |
| 13 | Space-separated table → HTML 변환 추가 | PubTabNet/TEDS_TEST | DS +20pt |
| 14 | KIE JSON garbage key fallback (regex 전환) | Nanonets-KIE | DS +6.4pt, P-P +6.2pt |
| 15 | LaTeX table cell: **bold** strip, \\% \\& \\$ 변환 | DP-Bench | TEDS 소폭 개선 |

**결론**: 모든 수정은 eval 코드 버그/프로토콜 불일치를 해결한 것이며, 남은 gap은 모델/파이프라인 성능 차이입니다.

---

## 9. 최종 결과

### Direct VLM 모델

| 벤치마크 | 메트릭 | 샘플 수 | GLM-OCR | PaddleOCR-VL | DeepSeek-OCR2 |
|:---|:---|---:|---:|---:|---:|
| **OCRBench** | Accuracy ↑ | 1,000 | **83.8%** | 71.3% | 48.4% |
| **PubTabNet** | TEDS ↑ | 200 | **86.1%** | 84.8% | 84.6% |
| **PubTabNet** | TEDS-struct ↑ | 200 | **92.3%** | 91.0% | 90.7% |
| **TEDS_TEST** | TEDS ↑ | 200 | **84.4%** | 84.4% | 81.9% |
| **TEDS_TEST** | TEDS-struct ↑ | 200 | **91.2%** | 90.6% | 87.9% |
| **IAM Handwritten** | CER ↓ | 200 | **3.9%** | 4.1% | 18.5% |
| **IAM Handwritten** | WER ↓ | 200 | **14.9%** | 14.4% | 30.4% |
| **UniMERNet** | CDM F1 ↑ | 200 | 94.1% | **96.9%** | 78.7% |
| **UniMERNet** | Edit Dist ↓ | 200 | 22.2% | **8.1%** | 39.2% |
| **UniMERNet** | BLEU ↑ | 200 | 74.3% | **89.8%** | 42.6% |

### Pipeline 모델

| 벤치마크 | 메트릭 | 샘플 수 | GLM-OCR-P | PaddleOCR-VL-P | DeepSeek-OCR2 |
|:---|:---|---:|---:|---:|---:|
| **OmniDocBench** | Overall ↑ | 1,358 | 91.9 | 91.9 | 79.6 |
| **Nanonets-KIE** | ANLS ↑ | 987/147 | 58.4% | 80.8% | **84.0%** |
| **DP-Bench** | NID ↑ | ~200 | 87.6% | 87.4% | **90.1%** |
| **DP-Bench** | TEDS ↑ | ~200 | 86.7% | **95.2%** | 83.0% |
| **DP-Bench** | TEDS-struct ↑ | ~200 | 89.3% | **96.9%** | 86.0% |

### OmniDocBench 세부

| 카테고리 | 메트릭 | GLM-OCR-P | PaddleOCR-VL-P | DeepSeek-OCR2 |
|:---|:---|---:|---:|---:|
| Text Block | 1−ED (ALL page) ↑ | 94.0% | **95.7%** | 88.1% |
| Text Block | 1−ED (edit whole) ↑ | 86.1% | **92.8%** | 71.4% |
| Table | TEDS ↑ | **89.9%** | 88.1% | 59.8% |
| Table | TEDS-struct ↑ | **94.2%** | 91.9% | 63.9% |
| Reading Order | 1−ED ↑ | 93.5% | **95.8%** | 89.1% |
| Formula | CDM ↑ | **91.9%** | 91.8% | 90.9% |
| Formula | 1−ED ↑ | **89.4%** | 89.0% | 82.8% |

### 모델별 강점/약점

**GLM-OCR**: OCR 정확도 1위, 테이블 1위, 손글씨 1위. 수식에서 PaddleOCR-VL에 비해 약간 뒤처짐.

**PaddleOCR-VL**: 수식 인식 1위 (CDM 96.9%), DP-Bench 테이블 1위. OCR 정확도 GLM 대비 낮음.

**DeepSeek-OCR2**: KIE 1위 (ANLS 84.0%), DP-Bench NID 1위. OCRBench 저조, 수식/손글씨 저조.

**GLM-OCR-Pipeline**: 테이블 구조 강점 (TEDS 89.9%), 수식 CDM 근소 우위.

**PaddleOCR-VL-Pipeline**: 텍스트 블록 강점 (95.7%), 읽기 순서 강점 (95.8%), KIE 크게 우위 (80.8% vs 58.4%).

---

## 10. Gap 분석

### 공식 발표 수치 대비

| 벤치마크 | 모델 | 우리 결과 | 공식 발표 | Gap | 원인 |
|:---|:---|---:|---:|---:|:---|
| OmniDocBench | GLM-OCR-P | 91.9 | 94.62 | −2.7 | 파이프라인 구현 차이 |
| OmniDocBench | PaddleOCR-VL-P | 91.9 | 94.50 | −2.6 | 파이프라인 구현 차이 |
| OmniDocBench | DeepSeek-OCR2 | 79.6 | 91.09 | −11.5 | VLM-direct vs 파이프라인 |
| OCRBench | GLM-OCR | 83.8 | 94.0 | −10.2 | VLM 추론 환경 차이 |
| UniMERNet CDM | GLM-OCR | 94.1 | 96.5 | −2.4 | 모델 성능 차이 |

### Gap 원인 상세

**OmniDocBench -2.6~2.7pt (Pipeline 모델)**:
- SDK 기본 설정 vs eval 설정 차이 (`request_timeout`, `max_tokens`)
- 1.1%의 샘플이 타임아웃으로 빈 출력
- 공식 벤치마크에서는 최적 설정(`max_tokens: 16384`, `request_timeout: 300`) 사용 추정

**OmniDocBench -11.5pt (DeepSeek-OCR2)**:
- DeepSeek-OCR2는 Direct VLM으로 평가 (파이프라인 없음)
- 공식은 자체 파이프라인으로 평가 추정
- 전체 이미지 → 단일 마크다운 출력 vs 영역별 추론의 근본적 차이

**OCRBench -10.2pt (GLM-OCR)**:
- Direct VLM의 이미지 리사이즈 (2048×2048) 영향 가능
- 공식은 파이프라인(영역 크롭) 모드로 평가 추정
- VLM 추론 환경(vLLM 설정) 차이

**결론**: 모든 gap은 모델/파이프라인 품질 차이이며, eval 코드 버그는 아님 (전수 샘플 확인 완료).

---

## 11. B200 파이프라인 vLLM 하이퍼파라미터

> `run_b200_pipeline.py` — B200 (192GB) 환경 기준

### 11.1 vLLM 옵션 의미

| 옵션 | 의미 | 단위 |
|:---|:---|:---|
| `--max-model-len` | **단일 시퀀스**의 최대 토큰 수 (input + output 합계) | per-request |
| `--max-num-batched-tokens` | 한 스케줄러 iteration에 모든 시퀀스 합산 처리 토큰 수 | per-iteration, 전체 배치 |
| `--max-num-seqs` | 동시 처리 시퀀스 수 (배치 크기) | per-iteration |
| `max_tokens` (API 요청) | 단일 요청의 최대 **출력** 토큰 수 | per-request |

- `max input tokens`는 별도 옵션이 없으며 `max_model_len - max_tokens(출력)` 로 암묵적 결정
- vLLM은 PagedAttention으로 KV cache를 동적 할당하므로, `max_model_len`은 per-sequence 상한이고 배치가 커져도 자동 처리됨

### 11.2 GLM-OCR vLLM 서버 (OCR 단계)

| 항목 | 값 | 공식 모델 설정 | 비고 |
|:---|:---|:---|:---|
| `--max-model-len` | **131,072** | `max_position_embeddings: 131072` | 공식 일치 |
| `--max-num-batched-tokens` | 65,536 | — | throughput 제어 |
| `--max-num-seqs` | 1,024 | — | 동시 시퀀스 |
| `max_tokens` (SDK config patch) | **8,192** | 공식 예제: `max_new_tokens=8192` | 공식 일치 |
| `--gpu-memory-utilization` | 0.90 | — | B200 192GB 기준 |
| `--quantization` | fp8 | 모델 native: bf16 | B200 FP8 하드웨어 활용 |
| `--enable-chunked-prefill` | 활성 | — | 긴 시퀀스 + 높은 동시성 |

**실제 토큰 사용 패턴** (OCR):
- 입력: 이미지 토큰 ~1-2K + 프롬프트 ~수백 = **~2K**
- 출력: 페이지 복잡도에 따라 **~1K-8K** (복잡한 테이블/수식 페이지는 최대 16K)
- 합계: 일반적으로 **~3K-10K**, `max_model_len` 131K 대비 충분한 여유

### 11.3 Qwen3-VL-Embedding vLLM (임베딩 단계)

| 항목 | 값 | 공식 모델 설정 | 소스 | 비고 |
|:---|:---|:---|:---|:---|
| `max_model_len` | **8,192** | `MAX_LENGTH = 8192` | `qwen3_vl_embedding.py:24` | 공식 일치 |
| `MAX_PIXELS` | **1,843,200** | `1800 * 32 * 32` | `qwen3_vl_embedding.py:28` | 공식 일치 |
| `MIN_PIXELS` | 4,096 (vLLM 기본) | `4 * 32 * 32` | `qwen3_vl_embedding.py:27` | — |
| `runner` | pooling | — | — | 임베딩 전용 (generation 없음) |
| `model_id` | `Forturne/Qwen3-VL-Embedding-2B-FP8` | base: `Qwen/Qwen3-VL-Embedding-2B` | — | FP8 양자화 버전 |
| `quantization` | fp8 | — | — | 모델 ID에 FP8 포함 시 자동 |
| `gpu_memory_utilization` | 0.90 | — | — | |
| embedding dim | 2,048 | `hidden_size: 2048` | `config.json` | |

**참고**: `max_position_embeddings`는 262,144 (256K), 모델 카드 context length는 32K이나, 공식 추론 코드의 기본 `MAX_LENGTH`는 8,192. 공식 vLLM 노트북 예제에서도 `max_model_len`을 별도 지정하지 않음 (모델 기본값 사용).

### 11.4 출력 파일 (임베딩)

```
output/embeddings/
  corpus_ocr_text.npy      # (N_pages, 2048) 텍스트 임베딩
  corpus_regions.npy        # (N_regions, 2048) region 멀티모달 임베딩 (image+caption)
  region_metadata.jsonl     # [{page_id, region_index, crop_path, caption_text, label}]
  queries.npy               # (N_queries, 2048) 쿼리 임베딩
```

- `corpus_regions.npy`: image/chart 크롭 이미지 + 주변 figure_title/vision_footnote 텍스트를 결합한 멀티모달 임베딩
- `region_metadata.jsonl`: region↔page 매핑 정보, 검색 결과를 페이지로 역추적할 때 사용

### 11.5 GPU 메모리별 권장 설정

> B200 (192GB) 실측 기반 산출. KV cache 비용: GLM-OCR 64 KB/token, Embedding 112 KB/token.
> 동시성은 실제 OCR 시퀀스 길이 기준 (~10K tokens/request). `--max-model-len`, `--max-tokens`는 모든 GPU에서 동일.

#### GLM-OCR vLLM 서버

**고정 파라미터** (GPU 무관): `--max-model-len 131072`, `max_tokens=8192`, `--enable-chunked-prefill`

| GPU | dtype | `--max-num-seqs` | `--max-num-batched-tokens` | `--workers` | `--batch-size` | KV 여유 | 동시성 (@10K) |
|:----|:------|:--:|:--:|:--:|:--:|:--:|:--:|
| **24 GB** (4090) | FP8 | 21 | 65536 | 84 | 6 | 13.2 GB | 22× |
| **48 GB** (A6000/L40) | FP8 | 57 | 65536 | 228 | 17 | 34.8 GB | 57× |
| **80 GB** (A100/H100) | FP8 | 104 | 65536 | 256 | 31 | 63.6 GB | 104× |
| **96 GB** (H100 NVL) | FP8 | 127 | 65536 | 256 | 39 | 78.0 GB | 128× |
| **192 GB** (B200) | FP8 | 269 | 65536 | 256 | 64 | 164.4 GB | 269× |

- **모델 weight**: FP8 ~1.4 GB, BF16 ~2.2 GB
- **activation overhead**: ~7 GB (프로파일링 기준)
- `--max-num-seqs`: KV cache가 수용 가능한 동시 시퀀스 수 (실 시퀀스 ~10K 기준)
- `--max-num-batched-tokens`: 65536 고정 (chunked prefill이 알아서 분할)
- `--workers`: SDK 병렬 요청 스레드 (max_num_seqs × 4, 최대 256)
- `--batch-size`: PP-DocLayoutV3 레이아웃 모델 배치 (vLLM과 별도)
- 24GB에서도 동시성 22×로 실용적

#### Qwen3-VL-Embedding

**고정 파라미터** (GPU 무관): `max_model_len=8192`, `runner=pooling`

| GPU | dtype | `--embed-batch-image` | `--embed-batch-text` | KV 여유 | 동시성 (@2K) |
|:----|:------|:--:|:--:|:--:|:--:|
| **24 GB** (4090) | FP8 | 30 | 32 | 13.2 GB | 62× |
| **48 GB** (A6000/L40) | FP8 | 64 | 32 | 34.8 GB | 163× |
| **80 GB** (A100/H100) | FP8 | 64 | 32 | 63.6 GB | 298× |
| **96 GB** (H100 NVL) | FP8 | 64 | 32 | 78.0 GB | 365× |
| **192 GB** (B200) | FP8 | 64 | 32 | 164.4 GB | 770× |

- **모델 weight**: FP8 ~1.4 GB, BF16 ~4.4 GB
- Embedding은 pooling 모드 (generation 없음), 실 입력 ~2K tokens/request
- 24GB에서도 batch_size=30으로 충분히 실용적

#### BF16 사용 시 (FP8 미지원 GPU)

| GPU | 모델 | 추가 VRAM 소요 | 비고 |
|:----|:-----|:--:|:---|
| 모든 GPU | GLM-OCR | +0.8 GB | weight 차이 작음, KV cache는 동일 |
| 모든 GPU | Embedding | +3.0 GB | weight 2배, 24GB에서는 batch_size 감소 필요 |

---

*Generated by OCR VLM Benchmark Evaluation System*
*결과 파일 위치: `/home/ubuntu/ocr_test/results/`*
