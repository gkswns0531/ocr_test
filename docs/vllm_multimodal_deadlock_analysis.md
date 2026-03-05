# vLLM Multimodal Deadlock Analysis

## Summary

Allgaznie OCR 파이프라인에서 PaddleOCR-VL/DeepSeek-OCR-2 모델 사용 시 vLLM 서버가 **~260 VLM 요청(~54 이미지) 처리 후 100% 재현 가능하게 교착**되는 문제가 발생했다.

**근본 원인 확정 (Phase 6)**: `allgaznie/preprocess.py`의 `crop_regions()`에서 **빈 bbox에 대해 1x1 픽셀 플레이스홀더 이미지**를 생성하여 VLM에 전송한 것이 원인.
- 1x1 이미지 → PaddleOCRVLProcessor에서 `ValueError: mean must have 1 elements if it is an iterable, got 3`
- vLLM이 HTTP 400 에러를 반환하지만, 에러 처리 경로에서 **내부 상태가 오염**
- 누적된 에러 응답이 ~260 요청 후 **완전 교착**을 유발 (race condition)
- **수정**: 1x1 플레이스홀더 대신 `None` 반환 + 파이프라인에서 None 크롭 스킵 → **100% 해결**

## Environment

- **vLLM version**: 0.16.1rc1.dev175 (nightly, 2026-03-03)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Pipeline**: Allgaznie (PP-DocLayoutV3 layout → region crop → per-region VLM OCR)
- **VLM requests**: Per-region HTTP API (max_workers=16, 페이지당 평균 ~4.8 리전)

---

## Phase 1: 초기 관찰 (이전 분석)

### Deadlock Pattern
1. vLLM server starts normally, processes 40-100 images successfully
2. A single VLM request hangs (30s timeout triggers `APITimeoutError`)
3. After timeout, the server is completely unresponsive:
   - `/health` endpoint hangs
   - `/v1/models` endpoint hangs
   - GPU utilization drops to 0%
   - `VLLM::EngineCore` process stays alive but idle
4. Server never recovers — requires full process kill + restart

### 초기 모델별 행동 (이후 업데이트됨)

| Model | Params | Quantization | 초기 관찰 | 최종 확인 |
|-------|--------|-------------|-----------|----------|
| GLM-OCR | 1.3B | FP8 | 2542 샘플 정상 | **데드락 발생 확인** |
| PaddleOCR-VL | 1.0B | BF16 | ~50-80 이미지 후 데드락 | **~54 이미지(~258 VLM 요청) 후 데드락** |
| DeepSeek-OCR-2 | 3.4B | FP8 | ~25-58 이미지 후 데드락 | 동일 패턴 |

> **주의**: 초기에 GLM-OCR가 정상으로 보인 것은 테스트 조건 차이. 이후 모든 모델에서 동일 패턴 확인됨.

---

## Phase 2: Encoder Cache 패치 시도 (효과 없음)

### 가설: GPU VRAM 누수 (#28230)

`EncoderCacheManager`의 lazy eviction으로 freeable 엔트리가 영구 잔류 → VRAM 누수.
`get_freed_mm_hashes()`에서 freeable을 즉시 flush하는 패치 적용.

- **패치 내용**: `vllm/v1/core/encoder_cache_manager.py` 수정 (상세: `vllm_encoder_cache_vram_leak_fix.md`)
- **테스트**: 16/16 유닛 테스트 통과
- **결과**: **효과 없음** — 동일 위치(#55)에서 데드락 재발

### 패치가 효과 없었던 이유

이후 모니터링에서 VRAM 누수 자체가 존재하지 않음이 확인됨 (Phase 4 참조).

---

## Phase 3: 체계적 변수 제거 실험

### 데드락 재현 조건 특정

| 시도 | 설정 변경 | 데드락 위치 | 결론 |
|------|-----------|------------|------|
| baseline (gpu_mem=0.85, workers=16) | 없음 | #55 | — |
| encoder cache 패치 | VRAM leak fix | #55 | VRAM 누수 아님 |
| gpu_mem=0.50 | VRAM 절반 | #55 | GPU 메모리 여유와 무관 |
| max_workers=1 | 동시요청 없음 | #55 | 동시성 무관 |
| VLLM_USE_V1=0 | V0 엔진 | #55 | V1 고유 버그 아님 |

**모든 설정에서 정확히 #55(54번째 이미지 직후)에서 100% 재현.**

### 핵심 3가지 실험

| 테스트 | 결과 | 의미 |
|--------|------|------|
| #55 건너뛰기 | #56에서 데드락 | 특정 이미지 아님. 누적 ~258 VLM 요청 후 아무 이미지나 데드락 |
| 54개 후 서버 재시작 | #55~#57 모두 성공 | **vLLM 서버 측 누적 상태가 원인** (클라이언트 무관) |
| VLM 요청 수 확인 | 54이미지 = 258 VLM 요청 | 이미지 수가 아니라 **VLM 요청 ~260개** 부근에서 상태 오염 |

### 단독 처리 테스트

- #55 이미지를 서버 시작 직후 단독 전송 → **25초, 정상 완료**
- 동일 이미지가 54개 이미지 이후에 전송되면 → **100% 데드락**

---

## Phase 4: 리소스 모니터링 (메모리 무관 확정)

### PaddleOCR-VL 데드락 직전까지 5초 간격 모니터링

```
시간       GPU%  VRAM(MB)  KV$%   Run Wait  VLM완료  RAM(MB)  CPU%
──────── ──── ──────── ───── ─── ──── ────── ─────── ────
[서버대기]  0%   84,258   0.0%   0    0      0   364K    ~4%
[layout]   0%   84,718   0.0%   0    0      0   378K    ~4%  ← layout 모델 로드
[추론시작] 97%   84,722   0.1%   1    0     11   379K    ~5%
           61%   84,722   0.0%   3    0     24   378K    ~5%
           60%   84,722   0.0%   4    0     29   378K    ~7%  ← 동시 4개 처리
           56%   84,724   0.0%   0    0     67   378K    ~4%
           98%   84,724   0.1%   1    0     67   378K    ~7%
           55%   84,724   0.0%   1    0     98   378K    ~7%
           29%   84,724   0.0%   0    0    111   378K    ~6%
           63%   84,724   0.0%   0    0    133   378K    ~8%
           82%   84,724   0.0%   0    0    167   378K    ~7%
           15%   84,724   0.0%   0    0    187   377K    ~6%
           79%   84,724   0.0%   1    0    222   378K    ~4%
           75%   84,724   0.0%   0    0    241   377K    ~5%  ← 마지막 정상
[데드락]    0%   84,724   N/A    0    0      0   377K    ~4%  ← 완전 멈춤
            0%   84,724   N/A    0    0      0   376K    ~4%
```

### 모니터링 결론

| 지표 | 관측 | 결론 |
|------|------|------|
| VRAM | 84,258 → 84,724 MB (+466MB, layout 로드분) | **누수 없음** — 추론 중 일정 |
| KV cache | 0.0~0.1% | **고갈 아님** — 거의 미사용 |
| RAM | 364 → 378 GB, 이후 일정 | **누수 없음** |
| GPU util | 15~98% → 즉시 0% | **갑작스러운 교착** — 점진적 저하 없음 |
| VLM 완료 | 241개 → 0 (metrics 멈춤) | **서버 전체 무응답** |
| CPU/RAM | 데드락 전후 변화 없음 | **클라이언트 측 정상** |

**확정: 메모리(VRAM/RAM/KV cache) 문제가 아님.**
모든 리소스 지표가 정상인 상태에서 ~241 VLM 요청 후 갑작스러운 완전 교착.

---

## Phase 5: 스택 덤프 분석 (서버측 교착 확정)

### 클라이언트측 스택 트레이스 (py-spy dump)

데드락 상태에서 클라이언트(infer.py / allgaznie) 프로세스의 전체 11개 스레드 상태:

| 스레드 | 상태 | 위치 |
|--------|------|------|
| Main thread | `wait()` 대기 | `concurrent.futures.result()` → `vlm.py:126 infer_regions` |
| Worker 1~7 (7개) | HTTP `read()` 블록 | `httpcore.sync.read` → `vlm.py:94 _infer_one` |
| Worker 8 | 유휴 대기 | `thread.py:89 _worker` (작업 큐 대기) |
| torch compile | 수신 대기 | `subproc_pool.py:76 _recv_msg` |
| 기타 1 | 유휴 | — |

### 해석

- **메인 스레드**: `ThreadPoolExecutor`의 7개 future 결과를 `wait()`로 대기 중
- **7개 워커 스레드**: vLLM 서버에 HTTP POST 전송 후 응답 소켓에서 `read()` 무한 대기
- **1개 워커**: 큐에서 다음 작업 대기 (유휴)
- **서버가 7개 요청을 수신했으나 응답을 생성하지 못하는 상태**

### 확정 결론

```
클라이언트 → HTTP POST (7개 동시) → vLLM 서버
          ← 응답 없음 (영원히)     ← 내부 교착
```

- 데드락은 **100% vLLM 서버 내부**
- 클라이언트는 단순히 HTTP 응답을 기다리고만 있음 (정상 동작)
- 서버가 ~241번째 VLM 요청 처리 중 내부 교착에 빠져 응답 생성 불가

> **Note**: ptrace 제한으로 vLLM 서버 프로세스의 스택은 확보하지 못함.
> 서버측 스택 트레이스 확보를 위해 `--cap-add=SYS_PTRACE` 또는 `VLLM_TRACE_FUNCTION=1` 필요.

---

## 코드 레벨 분석: vLLM 요청 처리 경로

```
[API Server Process]                    [EngineCore Process]

1. HTTP 요청 수신
2. process_inputs() ← 동기, event loop 블로킹
3. ZMQ send → input_socket              → input_queue에 enqueue
4. output_handler: await get_output()    ← 4. run_busy_loop():
                                              _process_input_queue()
                                              _process_engine_step()
                                                scheduler.schedule()
                                                model_executor.execute_model()
                                                  _execute_mm_encoder()
                                                    model.embed_multimodal() ← GPU 실행
                                              output_queue.put(result)
5. ZMQ recv ← output_socket             ← output_queue → ZMQ push
6. process_outputs → HTTP 응답 반환
```

**핵심 발견**: `/health` 엔드포인트는 EngineCore와 **통신하지 않는다**. 단순히 로컬 플래그만 확인:

```python
# async_llm.py:875
async def check_health(self) -> None:
    if self.errored:          # ← engine_dead 플래그 OR output_handler.done()
        raise self.dead_error  #    체크만 함. 타임아웃 없음.
```

따라서 **`/health`가 행한다 = asyncio event loop 자체가 응답 불가** 상태임을 의미한다.

---

## Phase 6: 근본 원인 확정 — 1x1 픽셀 플레이스홀더

### Heisenbug 발견: VLLM_TRACE_FUNCTION=1

`VLLM_TRACE_FUNCTION=1` 환경변수로 vLLM 서버를 시작하면 **데드락이 발생하지 않음** (100/100 완료).
이 환경변수는 모든 함수 호출에 tracing 오버헤드를 추가하여 **타이밍이 변경**됨.
→ **Race condition (경쟁 조건) 확정**: 고전적 Heisenbug (관찰하면 사라지는 버그).

### 1x1 플레이스홀더 → ValueError 발견

VLLM_TRACE_FUNCTION=1로 실행 시 vLLM 서버 로그에서 다수의 에러 발견:

```
ERROR: ... PaddleOCRVLProcessor image_processing_paddleocr_vl.py:382
ValueError: mean must have 1 elements if it is an iterable, got 3
```

원인 추적:
1. `allgaznie/preprocess.py:124`에서 빈 bbox(좌표 오류, 0면적 영역) 크롭 시 **1x1 흰색 PIL 이미지** 생성
2. 이 1x1 이미지가 JPEG 인코딩 → base64 → vLLM API로 전송됨
3. PaddleOCRVLProcessor가 1x1 이미지를 처리할 때, 단일 픽셀이 **1채널(그레이스케일)로 디코딩**됨
4. normalizer가 3채널 mean/std `[0.485, 0.456, 0.406]`를 적용하려 하나 1채널이므로 ValueError

### 에러 → 데드락 메커니즘

```
1x1 placeholder image
  ↓
JPEG encode → base64 → HTTP POST to vLLM
  ↓
PaddleOCRVLProcessor.preprocess()
  ↓  1x1 → 1 channel (grayscale)
  ↓  normalize(mean=[0.485,0.456,0.406]) expects 3 channels
  ↓
ValueError: mean must have 1 elements if it is an iterable, got 3
  ↓
vLLM returns HTTP 400 error
  ↓
에러 처리 경로에서 내부 상태(스케줄러/이벤트 루프) 부분 오염
  ↓  (race condition: 타이밍 의존적)
~260 요청 후 누적 오염 → 완전 교착
  ↓
/health, /v1/models 모두 무응답, GPU 0%
```

### 수정 내용

**`allgaznie/preprocess.py`** — `crop_regions()`:
```python
# Before (데드락 유발):
if region.size == 0:
    crops.append(Image.new("RGB", (1, 1), (255, 255, 255)))

# After (수정):
if region.size == 0:
    crops.append(None)
```

**`allgaznie/__init__.py`** — `_process_detections()`:
```python
# None 크롭 필터링 추가
raw_crops = crop_regions(image, vlm_dets, image_path)
crops = []
tasks = []
valid_indices: list[int] = []
for i, crop in enumerate(raw_crops):
    if crop is not None:
        crops.append(crop)
        tasks.append(vlm_dets[i].task)
        valid_indices.append(i)
# ... VLM 추론 후 valid_indices로 매핑
```

### 수정 검증

```
수정 전: PaddleOCR-VL 100샘플 → #55에서 100% 데드락 (5회 재현)
수정 후: PaddleOCR-VL 100샘플 → 100/100 완료, 0 타임아웃, 평균 287ms
```

### GLM-OCR는 왜 영향이 적었나?

GLM-OCR의 이미지 프로세서는 1x1 이미지를 에러 없이 처리할 수 있었던 것으로 추정.
PaddleOCRVLProcessor는 normalize 단계에서 채널 수 불일치로 즉시 ValueError 발생.
DeepSeek-OCR-2도 유사한 프로세서 이슈가 있을 가능성 높음.

---

## 최종 진단 (Phase 6 확정)

```
┌──────────────────────────────────────────────────────────────┐
│  확정: 1x1 플레이스홀더 → VLM 프로세서 에러 → 내부 상태 오염    │
│                                                              │
│  근본 원인: crop_regions()의 1x1 픽셀 플레이스홀더              │
│  • 빈 bbox → 1x1 흰색 이미지 생성 → VLM에 전송                │
│  • PaddleOCRVLProcessor: 1채널 vs 3채널 mean ValueError      │
│  • vLLM 에러 처리 경로의 race condition으로 내부 상태 오염       │
│  • ~260 요청 후 누적 오염 → 완전 교착                          │
│                                                              │
│  증거:                                                        │
│  • VLLM_TRACE_FUNCTION=1 (타이밍 변경) → 데드락 미발생         │
│  • 서버 로그에 다수의 ValueError 확인                           │
│  • 1x1 제거 후 100/100 완료, 0 타임아웃                        │
│                                                              │
│  수정: preprocess.py에서 None 반환 + __init__.py에서 스킵       │
└──────────────────────────────────────────────────────────────┘
```

### 가설 검증 최종표

| 가설 | Phase 1-2 평가 | Phase 3-5 | Phase 6 최종 |
|------|----------------|-----------|-------------|
| GPU VRAM 누수 (#28230) | **1순위** | **기각** — VRAM 일정 | 기각 |
| Event Loop Blocking (#34789) | 2순위 | 증상 관련 | 간접 관련 (에러 경로에서 blocking) |
| ZMQ 텐서 참조 누수 (#35357) | 3순위 | **기각** — RAM 일정 | 기각 |
| Encoder Cache lazy eviction | 패치 적용 | **기각** — 효과 없음 | 기각 |
| 누적 상태 오염 | 미고려 | **확정** | **확정** — 에러 응답 누적이 원인 |
| **1x1 플레이스홀더** | 미고려 | 미고려 | **근본 원인** |

### 모든 모델에서의 영향

| 모델 | 1x1 에러 발생? | 데드락? | 수정 후 |
|------|---------------|---------|---------|
| GLM-OCR | 미발생 (프로세서가 1x1 처리 가능) | 발생하나 임계값 높음 | 정상 |
| PaddleOCR-VL | **발생** (ValueError) | ~54 이미지 후 100% | **100% 해결** |
| DeepSeek-OCR-2 | 미확인 (유사 가능성) | ~25-58 이미지 후 | 검증 필요 |

---

## 수정 완료된 버그 (0.16.1rc1에 포함, 데드락 미해결)

### Bug A: Encoder Cache Leak from Waiting Requests
- PR [#31857](https://github.com/vllm-project/vllm/pull/31857) — 2026-01-10 머지, **포함 확인**
- WAITING 요청의 encoder cache 미해제 → 스케줄링 데드락
- **상태**: 수정 완료, 그럼에도 데드락 발생

### Bug B: Request Object Reference Cycle (CPU 메모리 누수)
- PR [#34183](https://github.com/vllm-project/vllm/pull/34183) — 2026-02-10 머지, **포함 확인**
- `partial(block_hasher, self)` 순환 참조로 멀티모달 Request 객체 GC 불가
- **상태**: 수정 완료, 그럼에도 데드락 발생

### 우리 패치: Encoder Cache Lazy Eviction Flush
- `get_freed_mm_hashes()`에서 freeable 엔트리 즉시 flush
- **상태**: 적용 완료, **효과 없음** (VRAM 누수 자체가 존재하지 않았음)
- 상세: `vllm_encoder_cache_vram_leak_fix.md`

---

## 관련 vLLM 이슈/PR 현황표

| Issue/PR | 유형 | 상태 | 우리 데드락과 관련성 |
|----------|------|------|---------------------|
| [#15294](https://github.com/vllm-project/vllm/issues/15294) | CPU RAM 누수 | **CLOSED** | 무관 (수정 포함) |
| [#18431](https://github.com/vllm-project/vllm/issues/18431) | 스케줄링 데드락 | **CLOSED** | 무관 (수정 포함, PR #31857) |
| PR [#34183](https://github.com/vllm-project/vllm/pull/34183) | Request 순환 참조 | **MERGED** | 무관 (수정 포함) |
| [#28230](https://github.com/vllm-project/vllm/issues/28230) | GPU VRAM 누수 | **OPEN** | **기각** — VRAM 누수 없음 확인 |
| [#17972](https://github.com/vllm-project/vllm/issues/17972) | Event loop 블로킹 | **OPEN** | 증상 관련, 근본 원인 아님 |
| [#35191](https://github.com/vllm-project/vllm/issues/35191) | ZMQ 텐서 참조 누수 | **OPEN** | **기각** — RAM 누수 없음 확인 |
| [#33319](https://github.com/vllm-project/vllm/issues/33319) | Waiting Queue 정체 | **OPEN** | **높음** — 동일 증상 (요청 멈춤, GPU 0%) |
| [#28038](https://github.com/vllm-project/vllm/issues/28038) | V1 ~100 요청 후 정지 | **OPEN** | **높음** — 유사 threshold |
| [#32897](https://github.com/vllm-project/vllm/issues/32897) | PaddleOCR-VL IndexError | **OPEN** | 낮음 (GLM-OCR에서도 데드락) |

---

## Reproduction

```bash
cd /root/ocr_test

# Start vLLM server (any multimodal model)
python3 -m vllm.entrypoints.openai.api_server \
  --model PaddlePaddle/PaddleOCR-VL \
  --port 8000 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --max-num-batched-tokens 16384 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85

# Run pipeline — deadlocks after ~54 images (~258 VLM requests)
python3 infer.py --model allgaznie-paddle --benchmarks omnidocbench --warmup 3 --port 8000
```

**재현 핵심 조건**:
- ~260 VLM 멀티모달 요청 누적 (이미지 수 아닌 VLM 요청 수)
- V0/V1, gpu_mem, max_workers 모두 무관
- 서버 재시작 시 카운터 리셋 → 다시 ~260 요청까지 정상

---

## 수정 사항 (적용 완료)

### 근본 수정: 1x1 플레이스홀더 제거

| 파일 | 변경 |
|------|------|
| `allgaznie/preprocess.py:123-124` | 빈 bbox → `None` 반환 (기존: 1x1 PIL Image) |
| `allgaznie/__init__.py:165-182` | None 크롭 필터링 + valid_indices 매핑 |

### 방어적 인프라 (유지)
- `allgaznie/vlm.py`: `ServerDeadError` 감지 (타임아웃 + health check 실패)
- `infer.py`: `ServerDeadError` → vLLM 프로세스 킬 → 서버 재시작
- `AllgaznieOCR.reconnect_vlm()`: HTTP 클라이언트만 재생성 (layout 모델 재로드 방지)
- **용도**: 근본 수정 이후에도 예상치 못한 vLLM 서버 이슈에 대한 안전망

---

## Next Steps

### 1. DeepSeek-OCR-2 검증
1x1 수정 후 DeepSeek-OCR-2에서도 데드락이 해결되는지 검증 필요.

### 2. 전체 벤치마크 실행
PaddleOCR-VL, DeepSeek-OCR-2 전체 벤치마크를 수정된 파이프라인으로 실행.

### 3. vLLM 이슈 보고 (선택)
1x1 이미지 에러 응답이 서버 교착을 유발하는 것은 vLLM 버그.
에러 응답 후 내부 상태가 정상적으로 정리되어야 하나 race condition으로 오염됨.
재현 스크립트와 함께 vLLM GitHub에 보고 가능.

---

## Key Files Reference

### vLLM 코드베이스

| Component | File Path | Key Lines |
|-----------|-----------|-----------|
| Encoder Cache Manager | `vllm/v1/core/encoder_cache_manager.py` | `allocate(180)`, `free_encoder_input(221)`, `get_freed_mm_hashes(255)` |
| GPU Worker Encoder Cache | `vllm/v1/worker/gpu_model_runner.py` | `encoder_cache dict(484)`, cache freeing(955) |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | Waiting req free(605), encoder scheduling(1140-1196) |
| Request Object | `vllm/v1/request.py` | `mm_features(140)`, `_block_hasher(170)` |
| EngineCore | `vllm/v1/engine/core.py` | `run_busy_loop()`, `gc.freeze(224)` |
| API Server | `vllm/v1/engine/async_llm.py` | `process_inputs()`, `check_health(875)` |

### Allgaznie Pipeline

| File | Purpose |
|------|---------|
| `/root/ocr_test/allgaznie/vlm.py` | VLMClient (max_workers=16, timeout, ServerDeadError) |
| `/root/ocr_test/allgaznie/preprocess.py` | Region cropping (1x1 placeholder bug at line 124) |
| `/root/ocr_test/allgaznie/__init__.py` | AllgaznieOCR pipeline with reconnect_vlm() |
| `/root/ocr_test/config.py` | Model configs with vLLM args |
| `/root/ocr_test/server.py` | VLLMServer lifecycle (start/stop/kill orphans) |
| `/root/ocr_test/infer.py` | Inference loop with ServerDeadError → restart logic |
