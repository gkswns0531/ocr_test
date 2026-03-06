# MinerU 3-Way Comparison: OmniDocBench Benchmark

## Overview

MinerU-VL (1.2B, `opendatalab/MinerU2.5-2509-1.2B`) 모델을 3가지 파이프라인으로 OmniDocBench (1,355 samples)에서 평가.

| Model | Latency (ms/sample) | Text (94.6) | Formula ED | Formula CDM | Table TEDS | Reading Order | Overall |
|:---|---:|---:|---:|---:|---:|---:|---:|
| MinerU Original | 3,279 | 94.6 | 80.5 | 91.0 | 86.4 | 94.8 | **89.4** |
| MinerU Optimized | 3,408 | 94.3 | 80.2 | 90.4 | 83.8 | 94.5 | 88.7 |
| **Allgaznie-MinerU** | **861** | 94.4 | 81.7 | 90.4 | **88.5** | 92.8 | **89.6** |

### Key Findings

- **Allgaznie-MinerU는 MinerU Original 대비 3.8x 빠르면서 동등 이상의 품질 달성**
- Table TEDS 88.5% — MinerU Original (86.4%)보다 높음
- Reading Order만 소폭 하락 (92.8 vs 94.8) — Allgaznie의 PP-DocLayoutV3 vs MinerU 자체 layout 차이
- 10가지 최적화 + FP8 quantization 모두 적용 유지

---

## Pipeline Descriptions

### 1. MinerU Original
기존 MinerU SDK (`mineru` 패키지) 그대로 사용. `hybrid` 백엔드 (내장 VLM).
- gpu_memory_utilization: 0.5 (기본값)
- 내부 transformers 엔진 사용

### 2. MinerU Optimized
MinerU SDK에 소스 레벨 최적화 적용 + 외부 vLLM 서버 (`hybrid-http-client` 백엔드).
- 적용 최적화: vLLM 서버 튜닝 (#4), Pipeline 모델 BF16 (#8), MFD batch_size 수정
- gpu_memory_utilization: 0.85, max-num-batched-tokens: 16384, max-num-seqs: 256, FP8
- MinerU 파이프라인 (OCR det/rec, MFD, Layout) 그대로 사용

### 3. Allgaznie-MinerU
Allgaznie 2-stage 파이프라인 (PP-DocLayoutV3 Layout + VLM) 에 MinerU-VL을 VLM으로 사용.
- 10가지 최적화 전부 내장: FP8, GPU JPEG decode, cv2 crop caching, vLLM 서버 튜닝, concurrent HTTP, vectorized NMS/containment, BF16 pipeline, torch.compile, single JPEG encode
- `--logits-processors` 로 MinerU의 NoRepeatNGram 로직 서버에 등록

---

## MinerU-VL 통합 시 핵심 패치 (vlm.py)

MinerU-VL 모델을 Allgaznie VLMClient에서 사용할 때 3가지 필수 설정:

### 1. Prompt 형식: `\n` prefix 필수
```python
"MinerU-VL": {
    "text": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "formula": "\nFormula Recognition:",
}
```
MinerU-VL은 `\n` prefix로 학습됨. 없으면 모델 output 품질 저하.

### 2. System Prompt 필수
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [...]},
]
```
System prompt 없으면 table 등에서 OTSL 대신 plain text 출력.

### 3. `skip_special_tokens: False` 필수
```python
extra_body = {
    "skip_special_tokens": False,
    ...
}
```
vLLM은 기본적으로 special tokens를 strip함. MinerU-VL의 OTSL 토큰 (`<fcel>`, `<nl>` 등)은 special tokens로 분류되어, 이 옵션 없이는 모든 구조 정보가 사라져 plain text만 남음.

**이 3가지가 없으면 Table TEDS가 0.05%로 떨어짐 (88.5% → 0.05%).**

### 4. Table Task: Greedy Decoding
```python
if task == "table":
    temperature = 0.0
    top_p = 0.01
    extra_body["vllm_xargs"] = {"no_repeat_ngram_size": 100}
```
Table task에서 OTSL format 출력을 위해 greedy decoding 필수. `no_repeat_ngram_size=100`은 대형 테이블에서 반복 방지.

### 5. OTSL → HTML 후처리
```python
from mineru_vl_utils.post_process import convert_otsl_to_html
```
VLM output (OTSL format) → HTML table 변환. OmniDocBench 평가는 HTML 기반 TEDS metric 사용.

### 6. `<|im_end|>` 토큰 제거
MinerU-VL은 output 끝에 `<|im_end|>` 토큰을 붙일 수 있음. 후처리에서 strip.

---

## A/B Test 결과 (Table OTSL 출력 검증)

동일 table crop 이미지로 4가지 요청 형식 비교:

| Test | Format | System Prompt | `\n` Prefix | temp | skip_special_tokens | OTSL Output |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | PNG (MinerU exact) | Yes | Yes | 0.0 | False | **Yes** |
| 2 | JPEG (Allgaznie original) | No | No | 0.1 | default(True) | No |
| 3 | JPEG + fixes (no penalties) | Yes | Yes | 0.0 | False | **Yes** |
| 4 | JPEG + full MinerU params | Yes | Yes | 0.0 | False | **Yes** |

- PNG vs JPEG: 차이 없음 (Test 1 vs 4)
- `presence_penalty`, `frequency_penalty`: Table OTSL 출력에 영향 없음 (Test 3 vs 4)
- **System prompt + `\n` prefix + `skip_special_tokens=False` + greedy decoding이 OTSL 출력의 필수 조건**

---

## Files Modified

| File | Change |
|------|--------|
| `allgaznie/__init__.py` | MinerU-VL을 VLM_MODEL_IDS, VLM_DISPLAY_NAMES에 추가 |
| `allgaznie/vlm.py` | MinerU-VL 프롬프트/매핑 추가, system prompt, skip_special_tokens, greedy table decoding, OTSL→HTML 후처리, `<\|im_end\|>` strip |
| `config.py` | `mineru-optimized`, `allgaznie-mineru` 모델 설정 추가 |
| `client.py` | `MinerUOptimizedClient` 추가 (hybrid-http-client 백엔드) |
| `infer.py` | `mineru_optimized` 백엔드를 needs_server에 추가 |
| `run_mineru_comparison.sh` | 3-way 비교 벤치마크 실행 스크립트 |

---

## Reproduction

```bash
# Phase 1: MinerU Optimized (외부 vLLM + MinerU SDK)
python3 infer.py --model mineru-optimized --benchmarks omnidocbench --warmup 3

# Phase 2: Allgaznie-MinerU (Allgaznie 파이프라인 + MinerU-VL)
python3 infer.py --model allgaznie-mineru --benchmarks omnidocbench --warmup 3

# Phase 3: Evaluation
python3 eval_bench.py --model mineru-optimized --benchmarks omnidocbench
python3 eval_bench.py --model allgaznie-mineru --benchmarks omnidocbench
```
