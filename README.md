# SDS-KoPub OCR + Embedding Pipeline

SamsungSDS KoPub VDR Benchmark (40,781 Korean document pages)에 대한 OCR 파싱 + VL 임베딩 파이프라인.

**GLM-OCR** (layout detection + VLM recognition) → **Qwen3-VL-Embedding** (image/text/query embeddings) → **HuggingFace 업로드**까지 단일 스크립트로 실행.

## Quick Start

```bash
# 1. 환경 설치 (B200 / 신규 머신)
bash setup_b200.sh

# 2. GLM-OCR SDK 패치 적용 (성능 최적화)
cd /root/glm-ocr-sdk && git apply /root/ocr_test/sdk_optimizations.patch

# 3. 전체 파이프라인 실행 (OCR → 임베딩 → HF 업로드)
python3 run_b200_pipeline.py \
  --hf-repo YOUR_USERNAME/sds-kopub-ocr-embeddings \
  --hf-token hf_xxxxx
```

## 파이프라인 단계

| Step | 설명 | 출력 |
|------|------|------|
| 1. Environment Check | CUDA, vLLM, GLM-OCR SDK 확인 | - |
| 2. Download Dataset | SDS-KoPub 40,781 페이지 다운로드 | `data/` |
| 3. Start vLLM Server | GLM-OCR VLM 서버 시작 (FP8) | - |
| 4. OCR Parsing | 레이아웃 감지 + VLM 인식 | `output/ocr_results.jsonl`, `output/crops/` |
| 5. Stop vLLM Server | OCR 서버 종료 | - |
| 6. Embeddings | 이미지/텍스트/쿼리 임베딩 계산 | `output/embeddings/*.npy` |
| 7. HF Upload | HuggingFace에 결과 업로드 | - |

## 개별 단계 실행

```bash
# OCR만 실행
python3 run_b200_pipeline.py --step ocr

# OCR 중단 후 재개
python3 run_b200_pipeline.py --step ocr --resume

# 첫 64페이지 테스트
python3 run_b200_pipeline.py --step ocr --limit 64

# 임베딩만 실행 (OCR 완료 후)
python3 run_b200_pipeline.py --step embed

# HF 업로드만
python3 run_b200_pipeline.py --step upload --hf-repo USER/REPO --hf-token hf_xxx
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--step` | 전체 | `ocr`, `embed`, `upload` 중 하나 |
| `--resume` | off | OCR 중단 후 이어서 처리 |
| `--ocr-dtype` | `fp8` | VLM 추론 dtype (`fp8`/`bf16`/`auto`) |
| `--gpu-mem-util` | `0.80` | vLLM GPU 메모리 사용률 |
| `--batch-size` | `64` | Layout detection 배치 크기 |
| `--workers` | `256` | VLM 병렬 요청 수 |
| `--embedding-model` | `Qwen/Qwen3-VL-Embedding-2B-FP8` | 임베딩 모델 |
| `--no-crops` | off | 이미지/차트 크롭 저장 건너뛰기 |
| `--limit` | 0 | 테스트용: 처음 N페이지만 처리 |

## GPU별 권장 설정

### B200 (192GB VRAM)

```bash
python3 run_b200_pipeline.py \
  --gpu-mem-util 0.75 \
  --batch-size 64 \
  --workers 256 \
  --ocr-dtype fp8
```

### L4 (23GB VRAM)

```bash
python3 run_b200_pipeline.py \
  --step ocr --resume \
  --gpu-mem-util 0.60 \
  --batch-size 8 \
  --workers 64 \
  --ocr-dtype fp8
```

---

## SDK 최적화 패치

`sdk_optimizations.patch`는 GLM-OCR SDK (`glmocr/`)에 적용하는 성능 패치입니다.

### 패치 적용 방법

```bash
cd /path/to/glm-ocr-sdk
git apply /path/to/sdk_optimizations.patch
```

### 패치 내용

#### 1. Layout Detector 최적화 (`glmocr/layout/layout_detector.py`)

**GPU JPEG 디코딩 (20x 빠른 전처리)**
- `torchvision.io.decode_jpeg(device='cuda')` + `F.interpolate`로 GPU에서 직접 JPEG 디코드+리사이즈
- 기존: PIL.Image.open → np.asarray → PIL.resize (101ms/img)
- 최적화: file read → GPU decode → GPU resize (5ms/img)
- JPEG 파일 경로가 제공되면 자동으로 GPU 경로 사용, 아니면 CPU fallback

**cv2 리사이즈 (CPU fallback 경로)**
- PIL.resize(BILINEAR) 대신 cv2.resize(INTER_LINEAR) 사용 (3.5x 빠름)
- `np.asarray()` 한 번만 호출하고 결과를 재사용

**직접 텐서 생성 (14x 빠른 전처리)**
- transformers `PPDocLayoutV3ImageProcessorFast` 전처리기 우회
- 프로세서 설정이 `mean=[0,0,0], std=[1,1,1], rescale=1/255`이므로 단순히 `tensor / 255.0`으로 대체
- numpy stack → torch permute → GPU transfer → float div

**단계별 타이밍 로깅**
- 전처리, 추론, 후처리 각 단계의 소요 시간을 INFO 레벨로 출력

#### 2. Pipeline 최적화 (`glmocr/pipeline/pipeline.py`)

**128 워커 제한 제거**
- `min(self.max_workers, 128)` → `self.max_workers`로 변경하여 B200에서 256+ 워커 사용 가능

**영역 크롭 최적화 (2x 빠름)**
- `crop_image_region()` 호출 시 매번 `np.asarray(image)` 변환하던 것을 이미지당 한 번만 변환
- JPEG 파일 경로가 있으면 `cv2.imread`로 직접 로드 (PIL lazy decode보다 3x 빠름)
- polygon 마스크 크롭 로직을 인라인하여 함수 호출 오버헤드 제거

**JPEG 경로 추적**
- 파이프라인 상태에 `image_paths_dict` 추가하여 이미지 인덱스→파일 경로 매핑
- 레이아웃 감지와 크롭에서 GPU 디코딩 및 빠른 로딩에 활용

**파이프라인 타이밍 계측**
- `PipelineTimings` dataclass로 데이터 로딩, 레이아웃, VLM 인식 각 단계 시간 측정
- `last_timings` 속성으로 외부에서 접근 가능

### 성능 결과 (B200, 64페이지 실측)

| 지표 | 최적화 전 | 최적화 후 | 개선 |
|------|----------|----------|------|
| 레이아웃 전처리 (64장) | 6.5s | 1.2s | **5.4x** |
| 레이아웃 모델 추론 | 1.3s | 1.2s | 1.1x |
| 레이아웃 process() 전체 | ~90s | 4.2s | **21x** |
| 영역 크롭 (64장) | ~18s | ~13s | 1.4x |
| 레이아웃 배치 전체 (크롭 포함) | ~90s | 17s | **5.3x** |
| VLM 인식 | 79s | 70s | 1.1x |
| 전체 OCR (64페이지) | 160s | 73s | **2.2x** |
| 처리 속도 | 0.4 pages/s | 0.9 pages/s | **2.25x** |

### 최적화 근거

프로파일링 (`profile_layout.py`, `benchmark_preprocess.py`)으로 확인한 병목:

```
PP-DocLayoutV3 레이아웃 감지 시간 내역 (64장, 최적화 전):
├── 이미지 전처리 (PIL resize + transformers processor)  90.5%  ← 병목!
├── 모델 추론                                            2.6%
├── 후처리                                               6.9%
└── 기타                                                 0.0%
```

- **transformers `PPDocLayoutV3ImageProcessorFast`**가 전체 시간의 90%를 차지
- 프로세서 설정 확인: `mean=[0,0,0], std=[1,1,1], rescale_factor=1/255` → 실질적으로 `pixel / 255.0`만 수행
- 800x800 리사이즈 + 255 나누기를 직접 구현하여 프로세서 완전 우회
- GPU JPEG 디코딩(`torchvision.io.decode_jpeg`)으로 CPU 디코딩 자체를 제거

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    run_b200_pipeline.py                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ Dataset   │───▶│ GLM-OCR SDK   │───▶│ OCR Results  │ │
│  │ (Arrow)   │    │  (patched)    │    │ (JSONL)      │ │
│  └──────────┘    │               │    └──────────────┘ │
│                  │ PP-DocLayout  │                      │
│                  │ + GPU JPEG    │                      │
│                  │   decode      │                      │
│                  │       ↓       │                      │
│                  │ vLLM GLM-OCR  │                      │
│                  │ (FP8 recog.)  │                      │
│                  └───────────────┘                      │
│                                                         │
│  ┌──────────────┐    ┌────────────────┐                │
│  │ OCR Results   │───▶│ Qwen3-VL-Emb   │──▶ .npy files │
│  │ + Images      │    │ (vLLM pooling) │                │
│  └──────────────┘    └────────────────┘                │
│                                                         │
│  ┌──────────────┐                                      │
│  │ HuggingFace  │◀── upload_folder()                   │
│  │ Hub          │                                      │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### 내부 파이프라인 (GLM-OCR SDK, patched)

```
Page Image (JPEG on disk)
    │
    ├── GPU path: torchvision.io.decode_jpeg(device='cuda')
    │   └── F.interpolate(800x800) → tensor / 255.0
    │
    └── CPU fallback: np.asarray → cv2.resize → np.stack → torch
    │
    ▼
PP-DocLayoutV3 (33M params, GPU)           ← Layout Detection (4.2s / 64 images)
    │ regions: text, table, formula, image, chart, ...
    ▼
cv2.imread + polygon crop (cached numpy)   ← Region Crop (~13s / 64 images)
    │
    ▼
vLLM GLM-OCR (0.9B, FP8)                  ← VLM Recognition (~70s / 64 images)
    │ text content, HTML tables, LaTeX formulas
    ▼
Result Formatter → JSONL
```

## vLLM 서버 설정 (B200 최적화)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `--max-model-len` | 32768 | 시퀀스 최대 길이 (메모리 절약) |
| `--max-num-batched-tokens` | 131072 | 동시 배치 토큰 수 (핵심! 이미지당 ~6K 토큰이므로 ~20장 동시 처리) |
| `--max-num-seqs` | 512 | 동시 시퀀스 수 |
| `--no-enable-chunked-prefill` | - | 프리필 분할 비활성화 (throughput 최적화) |
| `--gpu-memory-utilization` | 0.75 | GPU 메모리 75% 사용 (레이아웃 모델용 여유 확보) |
| `--quantization fp8` | - | FP8 동적 양자화 (B200 Blackwell 네이티브 지원) |

## GPU 모니터링

파이프라인 실행 중 pynvml로 실시간 GPU 사용량을 모니터링합니다:

- **SM Utilization**: GPU 코어 활용률 (%)
- **Memory Bandwidth**: 메모리 대역폭 활용률 (%)
- **VRAM Used**: 실제 VRAM 사용량 (GB)

각 shard 처리 후 요약 통계를 출력합니다.

## 모델 정보

| 모델 | 용도 | 크기 | dtype |
|------|------|------|-------|
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | VLM 텍스트 인식 | 0.9B | FP8 (dynamic quant) |
| [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) | 레이아웃 감지 | 33M | FP32 |
| [Qwen3-VL-Embedding-2B-FP8](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B-FP8) | 이미지/텍스트 임베딩 | 2B | FP8 |

## 파일 구조

```
ocr_test/
├── run_b200_pipeline.py        # 메인 파이프라인 (자체완결 단일 스크립트)
├── sdk_optimizations.patch     # GLM-OCR SDK 성능 패치
├── benchmark_preprocess.py     # 이미지 전처리 벤치마크 (SDK vs 네이티브 최적화)
├── profile_layout.py           # 레이아웃 감지 단계별 프로파일러
├── setup_b200.sh               # 환경 설치 스크립트
├── profile_pipeline.py         # 전체 파이프라인 레이턴시 프로파일러
├── eval_bench.py               # OCR 벤치마크 평가 (8개 벤치마크)
├── benchmarks.py               # 벤치마크별 평가기
├── metrics.py                  # 메트릭 (TEDS, CDM, ANLS, CER 등)
├── config.py                   # 모델/벤치마크 설정
└── README.md
```
