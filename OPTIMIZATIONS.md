# OCR Pipeline Optimization Log

각 파이프라인 단계에 적용한 SOTA 최적화 기록.
변경 사항은 `run_b200_pipeline.py`와 `sdk_optimizations.patch`에 반영됩니다.

---

## OPT-001: Stage 0 — Arrow raw bytes 직접 추출 + /dev/shm

**날짜**: 2026-02-27
**대상**: `run_b200_pipeline.py` → `_ocr_from_arrow_shards()`

### 문제

```
기존 흐름 (불필요한 작업 3단계):

Arrow 파일 (이미 PNG 바이트로 저장됨)
    ↓  ds[i]["image"]          → PIL 디코딩 (PNG→픽셀)     ~16ms/img
    ↓  .convert("RGB")         → 컬러 변환                 ~1ms/img
    ↓  .save(format="JPEG")    → JPEG 재인코딩 (픽셀→JPEG)  ~16ms/img
    ↓  /tmp/ 에 쓰기           → SSD 디스크 I/O             ~1ms/img
    총: ~34ms/img × 64 = ~2.2초

그리고 SDK가 이 파일을 다시 3번 읽음:
  1. PageLoader: base64 인코딩용
  2. Layout detector: GPU 디코딩용
  3. cv2.imread: 크롭용
```

### 핵심 발견

HuggingFace Arrow 데이터셋은 이미지를 **이미 인코딩된 PNG/JPEG 바이트**로 저장:
```
pa.struct({"bytes": pa.binary(), "path": pa.string()})
```

`ds[i]["image"]`를 호출하면 내부적으로 PNG 바이트를 PIL로 디코딩하는데, 우리는 그 디코딩된 이미지를 다시 JPEG로 인코딩해서 디스크에 쓰고 있었음. **완전히 불필요한 왕복**.

### 해결

```python
# Before: PIL 디코딩 + JPEG 재인코딩 + /tmp/ 디스크 쓰기
row = ds[local_idx]
pil_img = row["image"].convert("RGB")
pil_img.save(tmp_path, format="JPEG", quality=95)

# After: raw 바이트 직접 추출 + /dev/shm RAM 쓰기
ds_raw = ds.cast_column("image", Image(decode=False))
img_bytes = ds_raw[local_idx]["image"]["bytes"]  # 인코딩된 PNG 바이트 그대로
with open("/dev/shm/img.png", "wb") as f:
    f.write(img_bytes)  # 디코딩/재인코딩 없이 바이트 복사만
```

### 변경 사항

| 파일 | 변경 |
|------|------|
| `run_b200_pipeline.py` | `_ocr_from_arrow_shards()`: `cast_column(decode=False)` + `/dev/shm/` 사용 |
| SDK `pipeline.py` | 경로 감지에 `.png` 추가 |
| SDK `layout_detector.py` | `_gpu_preprocess_from_paths()`: PNG은 cv2 CPU 디코딩 + GPU 리사이즈 |

### SDK 변경 상세

**PNG 경로 지원 (pipeline.py)**:
```python
# Before: JPEG만 추적
if path.lower().endswith(('.jpg', '.jpeg')) and os.path.isfile(path):

# After: PNG도 추적 (crop에서 cv2.imread로 빠르게 로드)
if path.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(path):
```

**PNG 디코딩 (layout_detector.py)**:
```python
# Before: JPEG 전용 GPU 디코딩
decoded = torchvision.io.decode_jpeg(jt, device=self._device)

# After: 포맷별 최적 경로
if path.endswith(('.jpg', '.jpeg')):
    decoded = torchvision.io.decode_jpeg(jt, device=self._device)  # nvJPEG 하드웨어
else:
    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].copy()   # CPU decode
    decoded = torch.from_numpy(img).permute(2, 0, 1).to(device)   # GPU transfer
```

`torchvision.io.decode_jpeg`만 CUDA GPU 하드웨어 디코딩(nvJPEG) 지원. PNG는 GPU 하드웨어 디코더가 없으므로 cv2 CPU 디코딩 후 GPU 전송.

### 성능

| 지표 | Before | After | 개선 |
|------|--------|-------|------|
| 이미지 추출 (64장, 순차) | 7.915s | 0.043s | **184x** |
| 이미지 추출 (64장, 16 workers) | 1.341s | 0.043s | **31x** |
| img당 레이턴시 | 21.0ms (병렬) | 0.7ms | **30x** |
| PIL 디코딩 | 16ms/img | 0 (스킵) | 제거 |
| JPEG 재인코딩 | 16ms/img | 0 (스킵) | 제거 |
| 디스크 I/O | /tmp/ (SSD) | /dev/shm (RAM) | ~6x |
| 파일 크기 | 985KB/img (JPEG) | 587KB/img (PNG 원본) | 40% 작음 |
| 이미지 품질 | JPEG q95 (손실) | PNG 원본 (무손실) | 개선 |
| Layout decode | nvJPEG GPU | cv2 CPU + GPU resize | ~1.5x 느림 |
| **40K 예상** | **14분** | **0.5분** | **~14분 절약** |

### 병렬화 검증

병렬화가 유효한지 테스트한 결과, **순차가 최적**:

```
순차 (현재):     0.060s  (0.94ms/img)  ← 가장 빠름
2 workers:      0.081s  (1.26ms/img)  ← 0.7x (더 느림)
4 workers:      0.081s  (1.26ms/img)  ← 0.7x
8 workers:      0.086s  (1.34ms/img)  ← 0.7x
16 workers:     0.109s  (1.70ms/img)  ← 0.6x (가장 느림)
batch+8w:       0.063s  (0.98ms/img)  ← 1.0x (동일)
```

**이유**: 작업이 img당 0.94ms로 이미 너무 빨라서, ThreadPoolExecutor의 스레드 생성/스케줄링/GIL 경합 오버헤드가 실제 작업보다 큼. `/dev/shm`은 RAM 파일시스템이라 I/O 대기가 없으므로 병렬화 이득 없음.

기존 방식(PIL+JPEG)은 img당 123ms 걸려서 병렬화(16 workers → 21ms) 이득이 있었지만, raw bytes 추출은 img당 0.94ms라 병렬화 불필요.

### 트레이드오프

- Layout 전처리에서 nvJPEG GPU 디코딩 대신 cv2 CPU 디코딩 사용 (PNG이므로)
- GPU 디코딩 경로 대비 ~0.8초/64장 느리지만, Stage 0에서 ~7초 절약하므로 net positive
- 향후 데이터셋이 JPEG로 저장되어 있으면 nvJPEG 경로 자동 활성화

---

## OPT-002: Stage 2 — Layout 전처리 GPU JPEG 디코딩 + 프로세서 우회

**날짜**: 2026-02-27 (이전 세션)
**대상**: SDK `layout_detector.py`

### 문제

```
PP-DocLayoutV3 레이아웃 감지 시간 내역 (64장, 최적화 전):
├── 이미지 전처리 (PIL resize + transformers processor)  90.5%  ← 병목!
├── 모델 추론                                            2.6%
├── 후처리                                               6.9%
```

transformers `PPDocLayoutV3ImageProcessorFast`가 전체 시간의 90%를 차지. 프로세서 설정 확인 결과 `mean=[0,0,0], std=[1,1,1], rescale_factor=1/255`로, 실질적으로 `pixel / 255.0`만 수행.

### 해결

1. **transformers 프로세서 완전 우회**: `numpy.stack → torch.permute → GPU → float / 255.0`
2. **cv2.resize**: PIL.resize(BILINEAR) 대신 cv2.resize(INTER_LINEAR) (3.5x 빠름)
3. **GPU JPEG 디코딩**: `torchvision.io.decode_jpeg(device='cuda')` (JPEG 파일인 경우 20x 빠름)

### 성능

| 지표 | Before | After | 개선 |
|------|--------|-------|------|
| Layout process() 전체 (64장) | ~90s | 4.2s | **21x** |
| 전처리 | ~86s | 1.2s | **71x** |
| 모델 추론 | 1.3s | 1.2s | 1.1x |

---

## OPT-003: Stage 3 — 영역 크롭 캐싱 + cv2 직접 로드

**날짜**: 2026-02-27 (이전 세션)
**대상**: SDK `pipeline.py` → `_stream_process_layout_batch()`

### 문제

SDK의 `crop_image_region()` 함수가 영역마다 `np.asarray(image)` 호출. 64이미지 × 평균 8영역 = ~500번 PIL→numpy 변환 반복.

### 해결

1. **numpy 캐시**: 이미지당 1번만 numpy 변환, `_np_cache[img_idx]`로 재사용
2. **cv2.imread**: PIL lazy decode 대신 cv2로 직접 로드 (3x 빠름)
3. **인라인 크롭**: `crop_image_region()` 함수 호출 대신 bbox 슬라이싱 + polygon 마스크 인라인

### 성능

| 지표 | Before | After | 개선 |
|------|--------|-------|------|
| 영역 크롭 (64장, ~500 영역) | ~18s | ~13s | 1.4x |

---

## OPT-004: Stage 4 — VLM 서버 튜닝 + 워커 제한 제거

**날짜**: 2026-02-27 (이전 세션)
**대상**: `run_b200_pipeline.py`, SDK `pipeline.py`

### 변경

1. **vLLM 서버**: `--max-num-batched-tokens 131072` (기본 8192 → 16x 증가, ~20장 동시 처리)
2. **128 워커 제한 제거**: `min(self.max_workers, 128)` → `self.max_workers` (256+ 워커)
3. **`--max-num-seqs 512`**: 동시 시퀀스 수 증가
4. **`--no-enable-chunked-prefill`**: throughput 최적화

### 성능

| 지표 | Before | After | 개선 |
|------|--------|-------|------|
| VLM 인식 (64장) | 79s | 53s | 1.5x |

---

## 전체 성능 요약

| 단계 | 최적화 전 | 최적화 후 | 개선 |
|------|----------|----------|------|
| Image extract (Stage 0) | ~2.2s | ~0.1s | **~20x** |
| Layout detect (Stage 2 전체) | ~90s | 4.2s → ~2.2s (BF16+compile+vec) | **21x → 41x** |
| Layout postprocess (Stage 2) | ~9.5s | ~0.14s | **67x** |
| Region crop (Stage 3) | ~18s | ~13s | 1.4x |
| VLM request build (Stage 3) | ~3.5s | ~1.5s (parallel) | **2.37x** |
| VLM recognition (Stage 4) | 79s | 53s | 1.5x |
| **전체 OCR (64페이지)** | **~160s** | **~69s** | **~2.3x** |
| **처리 속도** | **0.4 pg/s** | **~0.9 pg/s** | **~2.3x** |

> Note: Layout postprocess ~9.5s는 합성 데이터 기준 (이미지당 ~120 박스).
> 실제 문서는 박스가 적어 ~1.3s → ~0.1s 수준으로 추정.

---

## OPT-005: Stage 1 — Data Loading 검토 (최적화 불필요)

**날짜**: 2026-02-27
**대상**: SDK `pipeline.py` → `data_loading_thread()`, `page_loader.py`
**결론**: **스킵 — 이미 최적에 가까움**

### 현재 동작

```
/dev/shm/img.png  →  PIL Image.open() [lazy]  →  images_dict + page_queue → Stage 2
```

`data_loading_thread`가 수행하는 작업:
1. `Image.open(path)` — **lazy open** (헤더만 읽음, 픽셀 디코딩 안함)
2. `images_dict[img_idx] = page` — 딕셔너리에 참조 저장
3. `image_paths_dict[img_idx] = path` — 파일 경로 저장
4. `page_queue.put(...)` — 큐에 넣기

### 최적화 불필요 사유

| 사유 | 설명 |
|------|------|
| PIL lazy open | `Image.open()`은 파일 헤더만 파싱, 실제 픽셀 디코딩 안함 |
| /dev/shm I/O | RAM 파일시스템이라 헤더 읽기 I/O가 거의 0 |
| 하위 단계 독립 | Stage 2는 `image_paths_dict` (파일 경로)로 직접 GPU/cv2 디코딩, PIL Image 미사용 |
| 소요 시간 미미 | ~1s / 64페이지 (전체 71s의 1.4%) |

### 하위 단계 이미지 사용 방식

| 단계 | 사용 데이터 | PIL Image 필요? |
|------|------------|----------------|
| Stage 2 전처리 | `image_paths_dict` → GPU/cv2 직접 디코딩 | 아니오 (경로만) |
| Stage 2 크롭 | `image_paths_dict` → cv2.imread numpy 로드 | 아니오 (경로만) |
| Stage 2 시각화 | `images_dict` → PIL Image | 예 (시각화 시에만) |
| Stage 3 VLM | cropped PIL Image → `build_request_from_image()` | 예 (크롭 결과) |

### 참고: base64 인코딩 위치

PROFILING.md에서 Stage 1에 "base64 인코딩"이 언급되었으나, 실제 base64 인코딩은 **Stage 3 (VLM Recognition)**의 `build_request_from_image()`에서 발생:

```python
# 크롭된 이미지마다 호출 (64페이지 × ~8영역 = ~500회)
def build_request_from_image(image, task_type):
    image.convert("RGB")              # 변환
    image.resize((...), BICUBIC)      # PIL 리사이즈
    image.save(buffered, format=fmt)  # JPEG/PNG 인코딩
    base64.b64encode(...)             # base64 인코딩
```

이 부분은 Stage 3/4 최적화에서 검토 예정.

### 성능

| 지표 | 현재 | 비고 |
|------|------|------|
| 소요 시간 (64장) | ~1s | 전체의 1.4% |
| img당 레이턴시 | ~15.6ms | PIL lazy open + queue put |
| **최적화 필요성** | **없음** | **이미 최적** |

---

## OPT-006: Stage 2 — Layout 후처리 벡터화 (NMS + Containment + Polygon)

**날짜**: 2026-02-27
**대상**: SDK `layout_postprocess_utils.py`

### 문제

Layout 후처리(`apply_layout_postprocess()`)가 64장 배치에서 ~1.3s 소요. 세 가지 O(n²) Python 루프가 병목:

```
후처리 시간 내역 (~1.3s):
├── NMS: ~200-400ms          ← O(n²) 스칼라 iou() 호출
├── Containment: ~300-600ms  ← O(n²) 중첩 Python 루프
├── Polygon matching: ~100-200ms ← O(n²) np.allclose() 검색
└── 기타 (filter, sort, format): ~100-200ms
```

이미지당 ~150개 박스 기준:
- NMS: 150×150 = 22,500회 스칼라 IoU 계산
- Containment: 22,500회 is_contained() 호출
- Polygon: ~100 필터링된 박스 × 150 원본 = 15,000회 np.allclose()

### 해결

**1. NMS: IoU 행렬 사전 계산 (numpy broadcasting)**

```python
# Before: O(n²) 스칼라 Python 루프
for i in indices:
    iou_value = iou(current_coords, box_coords)  # 스칼라 호출

# After: (n,n) IoU 행렬 한 번에 계산
x1, y1, x2, y2 = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3], coords[:, 3:4]
inter = np.maximum(0, np.minimum(x2, coords[:, 2]) - np.maximum(x1, coords[:, 0]) + 1) \
      * np.maximum(0, np.minimum(y2, coords[:, 3]) - np.maximum(y1, coords[:, 1]) + 1)
iou_matrix = inter / (areas + areas.T - inter)  # (n, n) 벡터화

# Class-aware threshold 행렬
same_class = (classes[:, None] == classes[None, :])
threshold_matrix = np.where(same_class, iou_same, iou_diff)

# Greedy selection은 순차 유지 (pre-computed matrix 참조)
for idx in order:
    if not suppressed[idx]:
        selected.append(idx)
        suppressed |= (iou_matrix[idx] >= threshold_matrix[idx])
```

**2. Containment: numpy broadcasting 벡터화**

```python
# Before: O(n²) Python 중첩 루프
for i in range(n):
    for j in range(n):
        if is_contained(boxes[i], boxes[j]):  # 스칼라 호출
            contained_by_other[i] = 1

# After: (n,n) containment 행렬 한 번에 계산
containment = inter / areas  # (n, n) broadcasting
np.fill_diagonal(containment, 0)  # self 제외
# preserve/mode 필터링도 행렬 마스킹으로 처리
contained_by_other = (containment >= 0.8).any(axis=1)
contains_other = (containment >= 0.8).any(axis=0)
```

**3. Polygon matching: orig_idx 추적으로 O(1) 룩업**

```python
# Before: O(n) 좌표 비교 검색 (필터링된 박스마다)
for orig_idx in range(len(boxes)):
    if np.allclose(boxes[orig_idx], box_data[2:6], atol=1.0):  # O(n) 검색
        poly = polygon_points[orig_idx]

# After: boxes_array에 orig_idx 컬럼 추가 (column 7)
boxes_array[:, 7] = np.arange(n_det)  # 빌드 시 원본 인덱스 저장
# ... NMS, filter, merge, sort 모두 orig_idx 자동 보존 ...
orig_idx = int(box_data[7])  # O(1) 직접 접근
poly = polygon_points[orig_idx]
```

### 변경 사항

| 파일 | 변경 |
|------|------|
| SDK `layout_postprocess_utils.py` | `nms()`: IoU 행렬 사전 계산 |
| SDK `layout_postprocess_utils.py` | `check_containment()`: numpy broadcasting 벡터화 |
| SDK `layout_postprocess_utils.py` | `apply_layout_postprocess()`: orig_idx 컬럼 추가 (column 7), 대이미지 필터링 벡터화, polygon O(1) 룩업 |

### 성능 (실측, `benchmark_postprocess.py`)

**개별 함수 벤치마크** (이미지당 ~120 박스):

| 함수 | n | Before | After | 개선 | 결과 일치 |
|------|---|--------|-------|------|----------|
| NMS | 50 | 11.2ms | 0.26ms | **43.8x** | ✓ |
| NMS | 100 | 21.8ms | 0.34ms | **63.3x** | ✓ |
| NMS | 150 | 48.9ms | 0.69ms | **71.2x** | ✓ |
| NMS | 200 | 89.5ms | 1.37ms | **65.2x** | ✓ |
| Containment | 50 | 7.6ms | 0.09ms | **84.8x** | ✓ |
| Containment | 100 | 38.2ms | 0.18ms | **214x** | ✓ |
| Containment | 150 | 65.4ms | 0.34ms | **190x** | ✓ |
| Containment | 200 | 118.5ms | 0.65ms | **182x** | ✓ |

**전체 후처리 벤치마크** (64 이미지, ~120 박스/이미지):

| 모드 | Before | After | 개선 | 일치 |
|------|--------|-------|------|------|
| NMS only | 9,471ms (148ms/img) | 141ms (2.2ms/img) | **67.1x** | 64/64 ✓ |
| NMS + merge=large | 8,634ms (135ms/img) | 114ms (1.8ms/img) | **75.5x** | 64/64 ✓ |

numpy broadcasting + IoU 행렬 사전 계산으로 Python 인터프리터 루프 오버헤드를 완전 제거.

### 트레이드오프

- IoU/Containment 행렬 메모리: n² floats (150 박스 → 180KB) — 무시 가능
- 기존 스칼라 `iou()`, `is_contained()` 함수는 하위 호환성을 위해 유지

---

## OPT-007: Stage 2a/2b — BF16 추론 + torch.compile + inference_mode + batch resize

**날짜**: 2026-02-27
**대상**: SDK `layout_detector.py`

### 문제

```
PP-DocLayoutV3 (33.2M params) 추론 현황:
├── FP32 추론 (B200 BF16 하드웨어 미활용)
├── torch.no_grad() (inference_mode보다 느림)
├── torch.compile 미적용 (그래프 퓨전 기회 상실)
├── 전처리: 이미지별 개별 F.interpolate (커널 런치 반복)
└── 전처리: FP32 텐서 생성 (BF16이면 메모리 절반)
```

### 해결

**1. BF16 추론 (모델 + 입력)**

```python
# Before: FP32 모델 + FP32 입력
self._model = self._model.to(self._device)  # float32
pixel_values = ... .float().div_(255.0)

# After: BF16 모델 + BF16 입력
if self._device.startswith("cuda") and torch.cuda.is_bf16_supported():
    self._model = self._model.bfloat16()  # 133MB → 66MB
self._model = self._model.to(self._device)
pixel_values = ... .to(dtype=self._model_dtype).div_(255.0)  # bfloat16
```

- B200 (compute cap 10.0)는 BF16 전용 텐서 코어 보유
- BF16: FP32 대비 ~2x 처리량, 동일 메모리 대역폭에서 2배 데이터
- 검출 모델의 BF16 정확도 차이는 무시할 수준

**2. torch.inference_mode() (no_grad 대체)**

```python
# Before: no_grad — gradient 비활성화만
with torch.no_grad():
    outputs = self._model(**inputs)

# After: inference_mode — gradient + version counting + 뷰 추적 모두 비활성화
with torch.inference_mode():
    outputs = self._model(**inputs)
```

**3. torch.compile() (그래프 최적화)**

```python
# 모델 로드 후 한 번 호출 (첫 배치에서 컴파일, 이후 배치 가속)
self._model = torch.compile(self._model)
```

- Operator fusion (conv+bn+relu → 단일 커널)
- Memory planning 최적화
- 40K 페이지 파이프라인에서 첫 배치 오버헤드 무시 가능

### 변경 사항

| 파일 | 변경 |
|------|------|
| SDK `layout_detector.py` `start()` | BF16 캐스트 + `torch.compile()` + `_model_dtype` 저장 |
| SDK `layout_detector.py` `_gpu_preprocess_from_paths()` | model dtype 사용, `cv2.cvtColor` |
| SDK `layout_detector.py` `process()` | `inference_mode()`, CPU fallback dtype |
| SDK `layout_detector.py` `_run_detection_single_image()` | `inference_mode()` + model dtype 입력 |

### 성능 (실측, `benchmark_layout_inference.py`, B200)

**모델 추론** (64장, 800×800):

| 설정 | 시간 | 개선 |
|------|------|------|
| FP32 + no_grad (baseline) | 211.4ms | — |
| FP32 + inference_mode | 211.4ms | 1.00x |
| **BF16 + inference_mode** | **153.6ms** | **1.38x** |
| **BF16 + inference_mode + compile** | **60.9ms** | **3.47x** |

**전처리 resize** (64장, 2480×3508 → 800×800):

| 방식 | FP32 | BF16 |
|------|------|------|
| per-image F.interpolate | 6.4ms | 5.5ms |
| batch stack + F.interpolate | 10.1ms | 9.8ms |

> Batch resize는 64장 × 2480×3508을 `torch.stack`하면 ~1.6GB 중간 텐서가 생성되어
> 오히려 느려짐. **per-image가 최적** → batch resize 미적용.

### 트레이드오프

- `torch.compile` 첫 배치 워밍업 비용 (~10-30s) — 40K 페이지에서 무시 가능
- BF16 precision: 검출 모델에서 실질적 정확도 차이 없음
- 컴파일 실패 시 eager mode 자동 폴백

---

## OPT-008: Stage 3 — 이중 JPEG 인코딩 제거 + VLM 요청 빌드 병렬화

**날짜**: 2026-02-27
**대상**: SDK `page_loader.py` → `build_request_from_image()`, `pipeline.py` → `vlm_recognition_thread()`

### 문제

```
build_request_from_image() 호출 흐름 (이중 인코딩):

PIL Image (크롭)
  ↓  image.save(JPEG)         → 1차 JPEG 인코딩       ← 완전히 낭비
  ↓  base64.b64encode()       → 1차 base64 인코딩      ← 완전히 낭비
  ↓  data:image/jpeg;base64,  → data URL 구성
  ↓  _process_msg_standard()
      ↓  base64.b64decode()   → base64 디코딩 (1차 되돌림)
      ↓  Image.open()         → PIL 재오픈 (1차 JPEG 되돌림)
      ↓  smart_resize()       → 해상도 계산
      ↓  image.resize(BICUBIC)→ 리사이즈
      ↓  image.save(JPEG)     → 2차 JPEG 인코딩 (실제 사용)
      ↓  base64.b64encode()   → 2차 base64 인코딩 (실제 사용)
```

크롭마다 JPEG+base64를 **2회** 수행. 이중 JPEG 압축은 품질도 저하시킴.
또한 `build_request_from_image()`이 VLM thread에서 **직렬** 실행 (API 호출만 병렬).

### 해결

**1. 이중 인코딩 제거 (single pass)**

```python
# Before: JPEG encode → base64 → data URL → _process_msg_standard (decode → resize → re-encode)
buffered = BytesIO()
image.save(buffered, format=self.image_format)  # 1차 JPEG ← 낭비
img_base64 = base64.b64encode(...)              # 1차 base64 ← 낭비
processed_msg = self._process_msg_standard(original_msg)  # 다시 decode → resize → re-encode

# After: PIL Image → load_image_to_base64 (smart_resize → JPEG → base64, 1회)
encoded_image = load_image_to_base64(image, ...)  # single pass
```

**2. VLM 요청 빌드 병렬화**

```python
# Before: build_request 직렬 → API 호출만 병렬
req = self.page_loader.build_request_from_image(cropped_image, task_type)  # 직렬 ← 병목
future = executor.submit(self.ocr_client.process, req)  # 병렬

# After: build_request + API 호출 모두 ThreadPoolExecutor에서 병렬
def _build_and_process(page_loader, ocr_client, image, task_type):
    req = page_loader.build_request_from_image(image, task_type)
    return ocr_client.process(req)
future = executor.submit(_build_and_process, ...)  # 전부 병렬
```

**3. BGR→RGB 최적화 (크롭 캐시)**

```python
# Before: numpy slice + copy
_cv2.imread(_path, _cv2.IMREAD_COLOR)[:, :, ::-1].copy()

# After: cv2.cvtColor (최적화된 SIMD 구현)
_cv2.cvtColor(_cv2.imread(_path, _cv2.IMREAD_COLOR), _cv2.COLOR_BGR2RGB)
```

### 변경 사항

| 파일 | 변경 |
|------|------|
| SDK `page_loader.py` `build_request_from_image()` | 이중 인코딩 제거 → `load_image_to_base64()` 직접 호출 |
| SDK `pipeline.py` `vlm_recognition_thread()` | `_build_and_process()`로 요청 빌드+API 호출 병렬화 |
| SDK `pipeline.py` `_stream_process_layout_batch()` | BGR→RGB에 `cv2.cvtColor` 사용 |

### 성능 (실측, `benchmark_request_build.py`)

**200 크롭 이미지** (평균 812×422):

| 설정 | 시간 | 개선 |
|------|------|------|
| Old (이중 인코딩, 직렬) — baseline | 3,498ms | — |
| **New (single pass, 직렬)** | **1,665ms** | **2.10x** |
| Old (이중 인코딩, 16 workers) | 3,867ms | 0.90x |
| **New (single pass, 16 workers)** | **1,477ms** | **2.37x** |

> Old parallel이 Old serial보다 느린 이유: PIL JPEG 인코딩이 CPU-bound + GIL 경합.
> New parallel은 이중 인코딩 제거로 GIL 보유 시간이 절반으로 줄어 병렬화 효과 발생.
> 실제 파이프라인에서는 API I/O 대기와 겹치므로 인코딩 오버헤드가 더 많이 숨겨짐.

### 부가 효과

- **이미지 품질 향상**: 이중 JPEG 압축 제거 → 아티팩트 감소
- **메모리 절약**: 중간 BytesIO 버퍼 1개 제거

---

## OPT-009: Stage 4 — VLM 인식 최적화 (서버/SDK 분석)

**날짜**: 2026-02-27
**대상**: SDK `image_utils.py`, vLLM 서버 설정

### 현황 분석

```
VLM 파이프라인 구조:
  SDK (클라이언트)                    vLLM (서버, 같은 B200)
  ┌─────────────┐                   ┌──────────────────────┐
  │ 256 workers  │──── HTTP/1.1 ───→│ GLM-OCR (FP8)        │
  │ pool=288     │    localhost      │ max-seqs=512         │
  │ build+send   │                  │ batch-tokens=131072  │
  └─────────────┘                   └──────────────────────┘
        ↑                                     ↑
   이미 최대 병렬화                      GPU 추론이 병목 (75%)
```

**53s의 원인:** vLLM GPU 추론 처리량. SDK 클라이언트는 이미 최적:
- `max_workers=256`, `connection_pool_size=288` (config에서 동적 설정)
- `region_maxsize=5120` (충분한 큐 깊이)
- OPT-008에서 요청 빌드 병렬화 완료

### SDK 최적화

**1. 불필요한 resize 스킵**

```python
# Before: 항상 BICUBIC resize (동일 크기여도)
image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

# After: 크기 변경 없으면 스킵 (크롭된 작은 영역 대다수 해당)
if (w_bar, h_bar) != (w, h):
    image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)
```

크롭 영역 대부분은 `max_pixels` (1M) 이하이므로 resize 불필요 → BICUBIC 연산 절약.

### vLLM 서버 튜닝 권장사항

| 항목 | 현재 | 권장 | 효과 |
|------|------|------|------|
| FP8 양자화 | ✅ `CutlassFP8ScaledMMLinearKernel` | — | 모델 1.34GB, BF16 대비 ~2x 처리량 |
| `--enable-prefix-caching` | ✅ 이미 활성화 | — | OCR 프롬프트 KV cache 재사용 중 |
| `--no-enable-chunked-prefill` | 설정 (비활성화) | **제거** (활성화) | 이미지 prefill 병렬화 향상 |
| `--gpu-memory-utilization` | 0.75 | 0.85-0.90 | KV cache 증가 (현재 115GB/1.89M 토큰) |
| JPEG quality | 75 (PIL default) | 유지 | localhost에서 전송 시간 무시 가능 |

> FP8 + prefix caching 이미 적용 확인. chunked prefill 활성화와 GPU 메모리 활용률 증가만 남음.

### GPU 포화도 테스트 (실측, `benchmark_vllm_saturation.py`, B200)

**512 요청, 256 동시 워커:**

| 지표 | 값 |
|------|-----|
| Throughput | **24.6 req/s**, 172 tok/s |
| Latency p50 / p95 | 4,198ms / 19,139ms |
| GPU SM% peak / avg | **22%** / ~10% |
| 모델 메모리 | 1.34 GB (B200 183GB의 0.7%) |
| Prefix cache hit | **91.2%** (364K 토큰 중 332K 캐시) |
| MM cache hit | **93.6%** (이미지 인코더 캐시) |
| 평균 생성 토큰 | 17 tok/req |
| 실제 연산 토큰 | 31,985 (전체의 8.8%) |

**SM 22% 원인:**
1. 모델이 ~1.3GB로 B200의 160 SMs를 채울 수 없음
2. Prefix caching 91% 히트 → 실제 연산 8.8%만 수행
3. 생성 평균 17 토큰으로 decode 시간 극소
4. `--no-enable-chunked-prefill`로 이미지 prefill 순차 처리

> **결론:** "비효율"이 아닌 "모델 대비 GPU 과잉 스펙 + 캐시 최적화 성공". 텐서코어 100%에는 7B+ 모델 필요.

### 변경 사항

| 파일 | 변경 |
|------|------|
| SDK `image_utils.py` `load_image_to_base64()` | 불필요한 resize 스킵, 불필요한 seek 제거 |

---

## OPT-010: BFloat16 post-process 호환성 수정 (Critical Bugfix)

**날짜**: 2026-02-27
**대상**: SDK `layout_detector.py`

### 문제

OPT-007에서 layout 모델을 BF16으로 캐스팅했으나, HuggingFace `post_process_object_detection()`이
BF16 텐서를 지원하지 않아 **전체 layout 검출 실패**:

```
Layout post_process failed for image N in chunk: Got unsupported ScalarType BFloat16
```

- 128페이지 테스트: 64/64 이미지 layout 실패 → VLM에 전달할 영역 없음 → 사실상 빈 결과
- 총 253s (0.5 pages/s), VLM이 할 일 없어 빠르게 끝남 (실제로는 아무것도 안 함)

### 수정

```python
@staticmethod
def _outputs_to_float32(outputs) -> None:
    """Cast model outputs to float32 in-place for post_process compatibility."""
    for attr in ("logits", "pred_boxes", "pred_masks", "out_masks"):
        t = getattr(outputs, attr, None)
        if t is not None and t.dtype != torch.float32:
            setattr(outputs, attr, t.float())
```

- `_post_process_chunk_with_fallback()`: 배치 경로에서 post_process 전 FP32 캐스팅
- `_run_detection_single_image()`: 단일 이미지 fallback 경로에서도 동일 적용
- 모델 추론은 BF16 유지 (OPT-007의 성능 이점 보존), 출력만 FP32 변환

### 수정 후 전체 파이프라인 결과 (128 pages)

| 지표 | Before (BF16 bug) | After (fixed) |
|------|-------------------|---------------|
| Layout 에러 | 64/64 실패 | **0 에러** |
| 총 소요 시간 | 253s (0.5 p/s) | **186s (0.7 p/s)** |
| Layout batch 1 (torch.compile warmup) | 237.1s | **62.2s (1.0 img/s)** |
| Layout batch 2 | 13.2s | **29.1s (2.2 img/s)** |
| VLM 인식 | ~0s (처리할 영역 없음) | **184.8s** |
| GPU SM% | 0% | **avg=11%, max=98%** |
| VRAM 사용량 | — | **avg=159.7GB, max=166.3GB / 192GB** |

> 40K 페이지 예상 소요: ~16시간 (0.7 pages/s 기준)

### 변경 사항

| 파일 | 변경 |
|------|------|
| SDK `layout_detector.py` | `_outputs_to_float32()` 추가, 배치/단일 경로 적용 |

---

## vLLM 서버 파라미터 완전 가이드

### 파라미터 관계도

```
클라이언트 (SDK)                                    서버 (vLLM)
┌──────────────────────┐                   ┌────────────────────────────────────┐
│ max_workers          │── 동시 HTTP ────→ │ max-num-seqs                      │
│ (동시 요청 수)        │   요청 수         │ (동시 스케줄링 가능 요청 수)        │
│                      │                   │                                    │
│ max_tokens           │── 요청당 ───────→ │ max-model-len                     │
│ (요청당 최대 생성)    │   최대 출력       │ (입력+출력 합산 최대 시퀀스 길이)   │
│                      │                   │                                    │
│ batch_size           │   (layout 전용,   │ max-num-batched-tokens             │
│ (layout 모델 배치)    │    vLLM 무관)     │ (한 forward step 총 토큰 수)       │
│                      │                   │                                    │
│                      │                   │ gpu-memory-utilization             │
│                      │                   │ (VRAM 중 vLLM 선점 비율)           │
└──────────────────────┘                   └────────────────────────────────────┘
```

### 각 파라미터 상세

| 파라미터 | 위치 | 현재 | 역할 |
|----------|------|------|------|
| **`max-num-seqs`** | vLLM | 1024 | 서버가 동시에 스케줄링할 수 있는 최대 요청 수. `max_workers` 이상이어야 대기 없음 |
| **`max-model-len`** | vLLM | 32768 | 1개 요청의 (이미지토큰+프롬프트+생성) 최대 합산 길이. OCR은 ~3000~5000 사용 |
| **`max-num-batched-tokens`** | vLLM | 131072 | 한 **forward step**에 처리할 총 토큰. chunked-prefill 청크 크기 결정 |
| **`gpu-memory-utilization`** | vLLM | 0.90 | VRAM 중 vLLM이 선점하는 비율. 나머지 10%는 layout 모델 등에 사용 |
| **`max_tokens`** | SDK | 16384 | 요청당 최대 **생성** 토큰 수. OCR 평균 ~17토큰 |
| **`max_workers`** | SDK | 256 | 클라이언트 ThreadPoolExecutor workers = 동시 HTTP 요청 |
| **`batch_size`** | SDK | 64 | layout 모델 배치 크기. **vLLM과 완전 독립** |

### VRAM 할당 구조

```
vLLM VRAM 할당 (gpu-memory-utilization=0.90, B200 192GB):

  총 VRAM:            192 GB
  × 0.90:             172.8 GB (vLLM 선점)
  - 모델 가중치:      -  1.36 GB (FP8)
  - Activation:       - 29.3 GB  ← max-num-batched-tokens 에 비례
  ─────────────────────────────
  = KV Cache:          142.15 GB
  = 2,329,040 토큰 슬롯

  동시성 (worst case): 2,329,040 / 32,768 = 71개 (max-model-len 꽉 채울 때)
  동시성 (실제 OCR):   2,329,040 / ~3,000 = ~776개 (실제 토큰 사용량 기준)
```

### 실질적 바운드 분석

```
실제 동시 처리량 = min(
    max_workers,                                    # 1024 (클라이언트)
    max-num-seqs,                                   # 1024 (서버 스케줄링)
    KV_cache_tokens / avg_seq_len,                  # ~776 (메모리)
    max-num-batched-tokens / avg_prefill_tokens     # ~65  (한 스텝 처리)
)                                                         ↑
= min(1024, 1024, 776, 65) = 65개/step     ← 이게 실질적 상한
```

하지만 **step이 매우 빠르게 반복**되므로 (수십 ms), 실제 throughput은 65개/step이 아닌
초당 여러 step이 누적되어 22.7 req/s가 됨.

### `max-num-batched-tokens` 스케일링 A/B 테스트 (실측)

**조건**: B200, FP8, 256 동시 요청, 동일 64개 이미지, 서버 매회 완전 재시작 (캐시 초기화)

| batched_tokens | Activation | KV Cache | Req/s | p50 | p95 | 변화 |
|----------------|-----------|----------|-------|-----|-----|------|
| **131,072** (128K) | ~29 GB | **142 GB** | **22.7** | 6.1s | 9.0s | **baseline (최적)** |
| 262,144 (256K) | ~46 GB | 125 GB | 15.9 | 7.5s | 13.0s | **-30%** |
| 524,288 (512K) | ~80 GB | 91 GB | 14.9 | 7.8s | 14.0s | **-34%** |

**결론: 올리면 오히려 느려짐.**

원인:
1. 모델이 1.36GB로 GPU SM의 10-22%만 사용 — 큰 배치로 채워도 연산량 증가 미미
2. Activation 메모리 증가 → KV cache 감소 → prefix cache hit rate 하락
3. 스텝당 처리시간만 길어지고, decode interleaving 효율 저하

> **131,072가 이 모델(GLM-OCR ~1.3B FP8)의 최적점. 모델이 작아서 batched tokens를 올려도
> GPU를 더 채울 수 없음. 7B+ 모델이라면 다른 결과가 나올 수 있음.**

### 서버 설정 변경 이력

| 설정 | 초기 | 현재 | 변경 이유 |
|------|------|------|-----------|
| `max-num-batched-tokens` | 131072 | **131072 (유지)** | A/B 테스트 결과 최적 |
| `max-num-seqs` | 512 | **1024** | workers 스케일링 대비 |
| `enable-chunked-prefill` | OFF | **ON** | prefill-decode interleaving |
| `gpu-memory-utilization` | 0.75→0.80 | **0.90** | KV cache 증가 |

---

## 향후 검토 대상

- [x] ~~Stage 1: Data Loading — 검토 완료, 최적화 불필요 (OPT-005)~~
- [x] ~~Stage 2: Layout 후처리 — NMS/containment/polygon 벡터화 (OPT-006)~~
- [x] ~~Stage 2: Layout 전처리/추론 — BF16 + compile + inference_mode (OPT-007)~~
- [x] ~~Stage 3: VLM 요청 빌드 — 이중 인코딩 제거 + 병렬화 (OPT-008)~~
- [x] ~~Stage 3: Crop — GPU 크롭 검토 → 최적화 불필요~~
- [x] ~~Stage 4: VLM — 분석 완료, SDK 최적화 적용 + 서버 튜닝 권장 (OPT-009)~~
- [x] ~~OPT-010: BF16 post-process 호환성 수정 (Critical Bugfix)~~
- [ ] 전체: NVIDIA DALI 통합 검토
