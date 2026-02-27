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

### 트레이드오프

- Layout 전처리에서 nvJPEG GPU 디코딩 대신 cv2 CPU 디코딩 사용 (PNG이므로)
- GPU 디코딩 경로 대비 ~0.8초/64장 느리지만, Stage 0에서 ~2.1초 절약하므로 net positive
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
| Layout detection (Stage 2) | ~90s | 4.2s | **21x** |
| Region crop (Stage 3) | ~18s | ~13s | 1.4x |
| VLM recognition (Stage 4) | 79s | 53s | 1.5x |
| **전체 OCR (64페이지)** | **~160s** | **~71s** | **~2.3x** |
| **처리 속도** | **0.4 pg/s** | **~0.9 pg/s** | **~2.3x** |

---

## 향후 검토 대상

- [ ] Stage 1: Data Loading — SDK PageLoader의 base64 인코딩 최적화
- [ ] Stage 2: Layout 후처리 — NMS/merge 최적화 (현재 1.3s)
- [ ] Stage 3: Crop — GPU 기반 크롭 (현재 CPU 바운드)
- [ ] Stage 4: VLM — 배치 크기 / 동시성 추가 튜닝
- [ ] 전체: NVIDIA DALI 통합 검토
