# OCR Pipeline Latency Profiling Report

B200 (192GB HBM3e) 환경에서 64페이지 기준 측정한 전체 파이프라인 레이턴시 프로파일링 결과입니다.

## 전체 흐름 (End-to-End)

```
64페이지 처리 전체: 73초 (0.9 pages/s)

┌──────────────────────────────────────────────────────────────────────┐
│                    시간 흐름 (초)                                    │
│                                                                      │
│  0s        3s                20s                                73s  │
│  ├─────────┼─────────────────┼──────────────────────────────────┤   │
│  │ Image   │   Layout Batch  │        VLM Recognition           │   │
│  │ Save    │   (감지+크롭)    │        (텍스트 인식)               │   │
│  │ ~3s     │   ~17s          │        ~53s                      │   │
│  └─────────┴─────────────────┴──────────────────────────────────┘   │
│                                                                      │
│  ※ SDK 내부는 3개 스레드가 파이프라인으로 병렬 실행                      │
│     Thread 1: Data Loading → Thread 2: Layout → Thread 3: VLM        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 단계별 상세 분석

### Stage 0: Image Save (Arrow → JPEG) — ~3초

```
Arrow 데이터셋 → PIL Image → JPEG 파일 저장
ThreadPoolExecutor(16 workers)로 병렬 처리
```

**하는 일**: HuggingFace Arrow 포맷으로 저장된 이미지를 디코딩해서 `/tmp/`에 JPEG 파일로 저장합니다. GLM-OCR SDK가 파일 경로를 입력으로 받기 때문에 필요한 단계입니다.

- 원본 이미지: 2480x3508 pixels (A4 스캔 해상도)
- 16 workers 병렬 저장

---

### Stage 1: Data Loading Thread — ~1초

```
JPEG 파일 경로 → SDK PageLoader → base64 인코딩 → API 요청 준비
```

**하는 일**: SDK의 `PageLoader`가 JPEG 파일을 읽고, VLM 서버에 보낼 API 요청(base64 이미지 + 프롬프트)을 미리 준비합니다. 동시에 이미지 인덱스↔파일 경로 매핑(`image_paths_dict`)을 저장하여, 뒤의 Layout/Crop 단계에서 GPU 디코딩 및 빠른 로딩에 활용합니다.

---

### Stage 2: Layout Detection — 4.2초 (64장)

여기가 집중적으로 최적화한 부분입니다:

```
Layout process() 내부 분해:

  ┌─────────────────────────────────────────────────────┐
  │  2a. GPU JPEG Decode + Resize      1.2s  (28.6%)   │
  │      torchvision.io.decode_jpeg(device='cuda')      │
  │      → F.interpolate(800x800)                       │
  │      → tensor / 255.0                               │
  │                                                     │
  │  2b. PP-DocLayoutV3 Model 추론     1.2s  (28.6%)   │
  │      33M param RT-DETR 기반 레이아웃 감지 모델         │
  │      → 각 이미지에서 영역(text/table/formula 등) 감지  │
  │                                                     │
  │  2c. Post-processing              0.5s  (11.9%)    │
  │      post_process_object_detection                  │
  │      → bbox 좌표 변환, threshold 필터링               │
  │                                                     │
  │  2d. apply_layout_postprocess     1.3s  (31.0%)    │
  │      NMS(Non-Max Suppression) + bbox merge          │
  │      + unclip + per-class threshold                 │
  │                                                     │
  │  TOTAL                            4.2s  (100%)     │
  └─────────────────────────────────────────────────────┘
```

#### 2a. GPU JPEG Decode + Resize

- `torchvision.io.decode_jpeg(device='cuda')`: nvJPEG 하드웨어 디코더로 GPU에서 직접 JPEG 디코딩 (CPU PIL 대비 20x 빠름)
- `F.interpolate(800x800, bilinear)`: GPU에서 리사이즈 (레이아웃 모델 입력 크기)
- `tensor / 255.0`: 정규화 (processor가 `mean=[0,0,0], std=[1,1,1]`이므로 이것만 하면 됨)
- **최적화 전**: PIL.Image.open → np.asarray → processor(PIL resize + 정규화) = **~86초** (90.5%)

#### 2b. Model 추론

- PP-DocLayoutV3 (33M params, RT-DETR 아키텍처) 모델에 800x800 텐서 입력
- 출력: 각 이미지별 감지된 영역의 bbox 좌표, 클래스(25종), confidence score
- 클래스: `text`, `table`, `display_formula`, `image`, `chart`, `doc_title`, `paragraph_title` 등

#### 2c. Post-processing

- 모델 출력(정규화 좌표)을 원본 이미지 크기 좌표로 변환
- threshold(0.3) 이하 감지 결과 제거

#### 2d. Layout Postprocess

- NMS로 겹치는 bbox 제거
- `layout_merge_bboxes_mode`에 따라 큰/작은 bbox로 병합
- `layout_unclip_ratio`로 bbox 약간 확장 (텍스트 잘림 방지)
- 각 영역에 task_type 할당: `text`/`table`/`formula`/`skip`/`abandon`

---

### Stage 3: Region Cropping — ~13초 (64장, ~500 영역)

```
원본 이미지에서 감지된 각 영역을 잘라내기

  ┌─────────────────────────────────────────────────────┐
  │  3a. cv2.imread()로 원본 JPEG 로드    ~3s           │
  │      (이미지당 1번만, 캐시 재사용)                     │
  │                                                     │
  │  3b. bbox 좌표로 numpy array 슬라이싱  ~1s           │
  │      img_array[y1:y2, x1:x2]                       │
  │                                                     │
  │  3c. polygon 마스크 적용              ~5s            │
  │      cv2.fillPoly + cv2.copyTo                      │
  │      (비직사각형 영역 크롭용)                          │
  │                                                     │
  │  3d. Image.fromarray() + Queue.put()  ~4s           │
  │      PIL 이미지로 변환, VLM 큐에 전달                  │
  │                                                     │
  │  TOTAL                              ~13s            │
  └─────────────────────────────────────────────────────┘
```

**하는 일**: Layout 모델이 감지한 영역(text, table, formula 등)을 **원본 해상도 이미지**(2480x3508)에서 잘라냅니다. 잘린 크롭 이미지가 VLM(GLM-OCR)에 입력되어 텍스트 인식을 합니다.

- `skip` 영역(image, chart): VLM에 보내지 않고 메타데이터만 저장
- `abandon` 영역(header, footer, page number): 완전히 버림
- 나머지(text, table, formula): 크롭 → VLM 인식 큐에 전달

최적화 포인트:
- 이미지당 numpy 변환을 1번만 (`_np_cache`), 원래는 영역마다 `np.asarray()` 호출
- `cv2.imread`로 직접 로드 (PIL lazy decode보다 3x 빠름)
- `crop_image_region()` 함수 호출 대신 인라인 크롭

---

### Stage 4: VLM Recognition (GLM-OCR) — ~53초

```
크롭 이미지 → vLLM GLM-OCR → 텍스트/HTML/LaTeX

  ┌─────────────────────────────────────────────────────┐
  │  VLM 인식 파이프라인 (ThreadPoolExecutor, 256 workers)│
  │                                                     │
  │  4a. 크롭 이미지 base64 인코딩                        │
  │  4b. vLLM 서버에 HTTP POST 요청                      │
  │      POST /v1/chat/completions                      │
  │      {model: "glm-ocr", messages: [{image, prompt}]}│
  │  4c. 응답 수신 및 결과 수집                            │
  │                                                     │
  │  task별 프롬프트:                                     │
  │    text:    "Text Recognition:"                     │
  │    table:   "Table Recognition:"  → HTML <table>    │
  │    formula: "Formula Recognition:" → LaTeX          │
  │                                                     │
  │  vLLM 서버 설정:                                     │
  │    모델: GLM-OCR 0.9B (FP8 dynamic quant)           │
  │    max-num-batched-tokens: 131072                   │
  │    max-num-seqs: 512                                │
  │    max-model-len: 32768                             │
  │    gpu-memory-utilization: 0.75                     │
  │                                                     │
  │  64페이지 = ~500 영역 → 256 병렬 요청                 │
  └─────────────────────────────────────────────────────┘
```

**하는 일**: Layout에서 잘린 크롭 이미지마다 GLM-OCR VLM에 요청을 보내 텍스트를 인식합니다.

- `text` 영역: 일반 텍스트로 인식
- `table` 영역: HTML `<table>` 태그로 구조화 출력
- `formula` 영역: LaTeX 수식으로 출력
- 이미지당 평균 ~6K 토큰 (이미지 토큰이 대부분)
- `max-num-batched-tokens=131072` → 동시에 ~20개 이미지 배치 처리

**전체 시간의 73%**를 차지하는 최대 병목이며, 이것은 GPU 연산(VLM 추론) 자체가 걸리는 시간이므로 소프트웨어 최적화로 줄이기 어렵습니다.

---

## 최적화 전후 비교

| 단계 | 최적화 전 | 최적화 후 | 개선 |
|------|----------|----------|------|
| Layout 전처리 | ~86s | 1.2s | **71x** |
| ├ PIL resize (800x800) | 6.5s | (GPU에서 처리) | — |
| ├ transformers processor | ~80s | (제거) | — |
| └ GPU JPEG decode+resize | — | 1.2s | (신규) |
| Layout 모델 추론 | 1.3s | 1.2s | 1.1x |
| Layout 후처리 | 2.6s | 1.8s | 1.4x |
| **Layout process() 전체** | **~90s** | **4.2s** | **21x** |
| 영역 크롭 | ~18s | ~13s | 1.4x |
| **Layout 배치 전체** | **~90s** | **~17s** | **5.3x** |
| VLM 인식 | 79s | 53s | 1.5x |
| **전체 OCR (64페이지)** | **160s** | **73s** | **2.2x** |
| **처리 속도** | **0.4 pg/s** | **0.9 pg/s** | **2.25x** |

---

## GPU 모니터링 (pynvml 실측)

```
┌─────────────────────────────────────────────────────┐
│ GPU: NVIDIA B200 (192GB HBM3e)                      │
├─────────────────────────────────────────────────────┤
│ SM Utilization:  avg=35%  p50=30%  max=95%          │
│ Memory BW:       avg=25%  max=60%                   │
│ VRAM Used:       avg=155GB  max=192GB / 192GB       │
│                                                     │
│ 시간대별:                                            │
│ - Layout 단계: SM ~10-20% (33M 소형모델이라 낮음)     │
│ - VLM 단계:    SM ~40-95% (0.9B 모델이 GPU 주로 사용) │
│ - VRAM:        Layout 모델(0.5GB) + vLLM(~144GB)    │
└─────────────────────────────────────────────────────┘
```

---

## 40K 페이지 처리 예상 시간

```
현재 속도: 0.9 pages/s
40,781 페이지 ÷ 0.9 = ~45,312초 = ~12.6시간

단계별 예상:
  Image save:      ~1,900초 (~0.5시간)
  Layout detection: ~1,100초 (~0.3시간)  ← 최적화 효과
  Region cropping:  ~8,300초 (~2.3시간)
  VLM recognition: ~34,000초 (~9.4시간)  ← 전체의 75%
```

---

## 병목 요약

```
시간 비중 (최적화 후):

  ██████████████████████████████████████░░░░░░░░░░░  73%  VLM 인식 (GPU 연산)
  ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  18%  영역 크롭 (CPU I/O)
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   6%  Layout 감지 (GPU)
  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3%  이미지 저장 (I/O)

→ VLM 인식이 지배적 (모델 추론이라 소프트웨어로 최적화 불가)
→ Layout 전처리는 90초→1.2초로 71x 줄여서 더 이상 병목 아님
```

---

## 최적화 근거 (프로파일링 결과)

`profile_layout.py`와 `benchmark_preprocess.py`로 확인한 병목:

```
PP-DocLayoutV3 레이아웃 감지 시간 내역 (64장, 최적화 전):
├── 이미지 전처리 (PIL resize + transformers processor)  90.5%  ← 병목!
├── 모델 추론                                            2.6%
├── 후처리                                               6.9%
└── 기타                                                 0.0%
```

### 핵심 발견

1. **transformers `PPDocLayoutV3ImageProcessorFast`**가 전체 시간의 90%를 차지
2. 프로세서 설정 확인: `mean=[0,0,0], std=[1,1,1], rescale_factor=1/255` → 실질적으로 `pixel / 255.0`만 수행
3. 800x800 리사이즈 + 255 나누기를 직접 구현하여 프로세서 완전 우회
4. GPU JPEG 디코딩(`torchvision.io.decode_jpeg`)으로 CPU 디코딩 자체를 제거

### benchmark_preprocess.py 결과 (64장)

| 테스트 | SDK 방식 | 최적화 방식 | 개선 |
|--------|----------|------------|------|
| PIL vs cv2 resize | 3.47s (54ms/img) | 1.97s (31ms/img) | 1.8x |
| Processor vs direct tensor | 2.01s | 0.14s | **14x** |
| SDK crop vs cached numpy | 4.91s (9.9ms/crop) | 2.45s (4.9ms/crop) | 2.0x |
| Full pipeline (resize+tensor+inference+postproc) | 6.9s | 3.4s | 2.0x |

---

## 측정 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA B200 (192GB HBM3e) |
| CUDA | 12.x |
| vLLM | 0.16.0 |
| PyTorch | 2.x |
| 데이터셋 | SDS-KoPub-VDR-Benchmark (40,781 pages) |
| 이미지 크기 | 2480x3508 (A4 스캔) |
| 테스트 크기 | 64 pages (1 Arrow shard) |
| OCR 모델 | GLM-OCR 0.9B (FP8 dynamic quant) |
| Layout 모델 | PP-DocLayoutV3 (33M params, FP32) |
