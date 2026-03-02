# Epic C: 파이프라인 성능 최적화

## 1. 배경 및 목적

GLM-OCR 파이프라인이 64페이지 처리에 160초 소요. 어디가 병목인지 불명확하여 단계별 프로파일링을 수행하고, 식별된 병목에 대해 SDK 최적화 패치를 적용하여 전체 2.25x 성능 향상 달성.

## 2. 프로파일링 및 병목 분석 (C-1)

### 2.1 방법

- SDK 내부 코드에 monkey-patch를 적용하여 단계별 타이밍 측정
- pynvml을 사용한 GPU 모니터링 (SM utilization, memory BW, VRAM)

### 2.2 최적화 전 프로파일

```
64페이지 처리 전체: 160초 (0.4 pages/s)

  Image Save    Layout Batch           VLM Recognition
  ~2s           ~90s                   ~79s
  ├─1%─┤├───────56%──────────┤├──────43%──────────┤
```

### 2.3 Layout 단계 상세 (핵심 병목)

```
Layout process() 내부 (64장, ~90초):
├── 이미지 전처리 (PIL resize + transformers processor)  90.5%  ← 핵심 병목!
├── 모델 추론 (PP-DocLayoutV3)                            2.6%
├── 후처리 (NMS + bbox merge)                             6.9%
```

**핵심 발견**: `PPDocLayoutV3ImageProcessorFast`(transformers)가 전체 시간의 **90.5%**를 소비. 프로세서 설정이 `mean=[0,0,0], std=[1,1,1], rescale_factor=1/255`로, 실질적으로 `pixel / 255.0`만 수행하는데 불필요한 오버헤드가 막대.

### 2.4 VLM 단계 분석

```
vLLM 서버 포화도 테스트 (512 요청, 256 동시):
├── Throughput: 24.6 req/s, 172 tok/s
├── GPU SM% peak/avg: 22% / ~10%
├── Prefix cache hit: 91.2%
├── MM cache hit: 93.6%
└── 실제 연산 토큰: 전체의 8.8%만 실제 계산
```

SM 22%의 원인: 0.9B 모델이 160 SM을 채울 수 없고, prefix caching 91% hit로 실제 연산이 극소.

### 2.5 식별된 병목

| 병목 | 원인 | 최적화 가능성 |
|------|------|-------------|
| Layout 전처리 (90초) | transformers preprocessor 오버헤드 | **높음** — 우회 가능 |
| Layout 후처리 (NMS) | O(n²) Python 스칼라 루프 | **높음** — numpy 벡터화 |
| VLM 인식 (79초) | GPU 모델 추론 자체 | **낮음** |
| Region 크롭 (18초) | 반복적 PIL→numpy 변환 | **중간** |

## 3. SDK 최적화 패치 — 2.25x (C-2)

### 3.1 주요 최적화

**OPT-001: Arrow raw bytes + /dev/shm**
| Before | After | 개선 |
|--------|-------|------|
| PIL 디코딩→JPEG 재인코딩→/tmp/ (7.9s) | raw bytes→/dev/shm (0.04s) | **184x** |

**OPT-002: Layout 전처리 GPU 디코딩 + 프로세서 우회**
| Before | After | 개선 |
|--------|-------|------|
| transformers preprocessor (~90s) | nvJPEG + F.interpolate (4.2s) | **21x** |

핵심: `torchvision.io.decode_jpeg(device='cuda')` nvJPEG 하드웨어 디코더 + `F.interpolate` GPU 리사이즈로 transformers preprocessor 완전 대체.

**OPT-004: VLM 서버 튜닝**
| Before | After | 개선 |
|--------|-------|------|
| max-num-batched-tokens 8K (79s) | 65K (53s) | **1.5x** |

**OPT-006: Layout 후처리 벡터화**
| 함수 | Before | After | 개선 |
|------|--------|-------|------|
| NMS (n=150) | 48.9ms | 0.69ms | **71x** |
| Containment (n=150) | 65.4ms | 0.34ms | **190x** |

**OPT-007: BF16 추론 + torch.compile**
| Before | After | 개선 |
|--------|-------|------|
| FP32 + no_grad (211ms) | BF16 + inference_mode + compile (61ms) | **3.47x** |

**OPT-008: 이중 JPEG 인코딩 제거**
| Before | After | 개선 |
|--------|-------|------|
| JPEG→base64→decode→resize→re-encode (3,498ms) | single pass (1,477ms) | **2.37x** |

**OPT-010: BFloat16 post-process 호환성 수정 (Bugfix)**
HuggingFace `post_process_object_detection()`이 BF16 텐서 미지원 → 모델 출력만 FP32 캐스팅.

### 3.2 max-num-batched-tokens A/B 테스트

서버 매회 완전 재시작하여 5개 값 비교 (캐시 영향 제거):

| batched_tokens | Req/s | Tok/s | p50 |
|----------------|-------|-------|-----|
| 32K | 22.0 | 3,150 | 5.7s |
| **64K** | **24.6** | **3,611** | **5.4s** |
| 128K | 22.4 | 3,243 | 5.3s |
| 256K | 15.9 | 2,350 | 7.5s |
| 512K | 14.9 | 2,138 | 7.8s |

64K가 activation과 KV cache 간 최적 균형점.

### 3.3 전체 성능 요약

| 단계 | 최적화 전 | 최적화 후 | 개선 |
|------|----------|----------|------|
| Image extract | ~2.2s | ~0.1s | **20x** |
| Layout detect | ~90s | 4.2s | **21x** |
| Layout postprocess | ~9.5s | 0.14s | **67x** |
| Region crop | ~18s | ~13s | 1.4x |
| VLM request build | ~3.5s | ~1.5s | 2.37x |
| VLM recognition | 79s | 53s | 1.5x |
| **전체 (64페이지)** | **~160s** | **~69s** | **~2.3x** |
| **처리 속도** | **0.4 pg/s** | **~0.9 pg/s** | **~2.25x** |

## 4. 검증

7개 벤치마크 스크립트로 각 최적화 효과를 독립 검증:
- `benchmark_preprocess.py`: 전처리 PIL vs GPU decode
- `benchmark_postprocess.py`: NMS/Containment 벡터화
- `benchmark_layout_inference.py`: BF16 + compile 추론
- `benchmark_request_build.py`: 이중 인코딩 제거
- `benchmark_vllm_saturation.py`: vLLM GPU 포화도
- `benchmark_batched_tokens.py`: max-num-batched-tokens A/B
- `benchmark_batch_scaling.py`: 배치 크기 스케일링

## 5. 산출물

- `profile_pipeline.py`, `profile_layout.py`: 프로파일링 스크립트
- `sdk_optimizations.patch` (41KB): SDK 변경 사항 전체 패치
- `PROFILING.md`, `OPTIMIZATIONS.md`: 분석/최적화 보고서
- `benchmark_*.py`: 7개 벤치마크 스크립트
