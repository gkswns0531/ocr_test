# VLM Parameter Audit Report

> Full codebase audit of all Allgaznie VLM inference parameters.
> Verified against official model repositories, SDKs, generation_config.json, and preprocessor_config.json.
>
> Date: 2026-03-06
> Scope: GLM-OCR, PaddleOCR-VL, MinerU-VL, DeepSeek-OCR2

---

## Executive Summary

A comprehensive audit of the Allgaznie pipeline revealed **5 critical misconfigurations** that were silently degrading output quality for all VLM models. All issues have been fixed.

| # | Issue | Models Affected | Severity | Root Cause |
|---|-------|-----------------|----------|------------|
| 1 | `AllgaznieConfig` still had `temperature=0.1, top_p=0.1, repetition_penalty=1.1` — overriding VLMClient's greedy defaults | All models | **CRITICAL** | Config values passed explicitly to VLMClient, bypassing its defaults |
| 2 | `min_pixels` / `max_pixels` hardcoded to Qwen2VL defaults instead of per-model official values | PaddleOCR-VL (worst), GLM-OCR | **CRITICAL** | Single default for all models |
| 3 | Missing `presence_penalty` and `frequency_penalty` for MinerU-VL | MinerU-VL | **HIGH** | Not in original implementation |
| 4 | GLM-OCR `repetition_penalty` changed from official 1.1 to 1.0 | GLM-OCR | **MODERATE** | Greedy patch over-corrected |
| 5 | Missing `no_repeat_ngram_size=35` for DeepSeek-OCR2 | DeepSeek-OCR2 | **HIGH** | vLLM doesn't support it natively; required custom logits processor |

All previous Allgaznie benchmark results were run with incorrect settings and must be re-run.

---

## Methodology

### Sources Verified

| Model | Official Source | File Path |
|-------|---------------|-----------|
| GLM-OCR | `generation_config.json` | `/workspace/.cache/huggingface/hub/models--zai-org--GLM-OCR/.../generation_config.json` |
| GLM-OCR | `preprocessor_config.json` | `/workspace/.cache/huggingface/hub/models--zai-org--GLM-OCR/.../preprocessor_config.json` |
| GLM-OCR | SDK config.yaml | `/tmp/GLM-OCR/glmocr/config.yaml` |
| GLM-OCR | SDK config.py | `/tmp/GLM-OCR/glmocr/config.py` (PageLoaderConfig) |
| PaddleOCR-VL | `generation_config.json` | `/workspace/.cache/huggingface/hub/models--PaddlePaddle--PaddleOCR-VL/.../generation_config.json` |
| PaddleOCR-VL | `preprocessor_config.json` | `/workspace/.cache/huggingface/hub/models--PaddlePaddle--PaddleOCR-VL/.../preprocessor_config.json` |
| PaddleOCR-VL | PaddleX pipeline | `/usr/local/lib/python3.12/dist-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py` |
| MinerU-VL | `generation_config.json` | `/workspace/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/.../generation_config.json` |
| MinerU-VL | `preprocessor_config.json` | `/workspace/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/.../preprocessor_config.json` |
| MinerU-VL | MinerU SDK | `/usr/local/lib/python3.12/dist-packages/mineru_vl_utils/mineru_client.py` |
| DeepSeek-OCR2 | `config.json` | `/workspace/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/.../config.json` |
| DeepSeek-OCR2 | `processor_config.json` | `/workspace/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/.../processor_config.json` |
| DeepSeek-OCR2 | `modeling_deepseekocr2.py` | `/workspace/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/.../modeling_deepseekocr2.py` |

### Key Insight: Greedy Decoding and Penalties

When `temperature=0.0`, vLLM uses **greedy decoding** (argmax). In this mode:

- `top_p` and `top_k` are **irrelevant** (argmax ignores sampling filters)
- `repetition_penalty` **still applies** (multiplicative logit modification before argmax)
- `presence_penalty` **still applies** (additive logit modification before argmax)
- `frequency_penalty` **still applies** (additive, proportional to token count)

This means penalty parameters affect output quality even with greedy decoding.

---

## Per-Model Configuration

### 1. GLM-OCR (`zai-org/GLM-OCR`)

#### Official Sources

**generation_config.json:**
```json
{
  "do_sample": false,
  "eos_token_id": [59246, 59253],
  "pad_token_id": 59246
}
```

**preprocessor_config.json:**
```json
{
  "size": {"shortest_edge": 12544, "longest_edge": 9633792},
  "patch_size": 14,
  "merge_size": 2,
  "image_processor_type": "Glm46VImageProcessor"
}
```

**SDK config.py (PageLoaderConfig defaults):**
```python
temperature: float = 0.01
top_p: float = 0.00001
top_k: int = 1
repetition_penalty: float = 1.1
min_pixels: int = 12544   # 112 * 112
max_pixels: int = 71372800  # 14 * 14 * 4 * 1280
```

**SDK config.yaml (runtime override):**
```yaml
temperature: 0.8
top_p: 0.9
top_k: 50
repetition_penalty: 1.1
```

**SDK task prompts:**
```yaml
task_prompt_mapping:
  text: "Text Recognition:"
  table: "Table Recognition:"
  formula: "Formula Recognition:"
```

#### Resolved Configuration

| Parameter | Official Value | Our Setting | Rationale |
|-----------|---------------|-------------|-----------|
| temperature | 0.0 (`do_sample: false`) | 0.0 | `generation_config.json` is authoritative |
| top_k | 1 | 1 | Match |
| repetition_penalty | **1.1** | **1.1** (model-specific override in `_infer_one`) | SDK config.py and config.yaml both specify 1.1 |
| presence_penalty | not specified (0.0) | 0.0 | No official source |
| frequency_penalty | not specified (0.0) | 0.0 | No official source |
| min_pixels | 12544 | 12544 | `preprocessor_config.json` shortest_edge |
| max_pixels | 9633792 | 9633792 | `preprocessor_config.json` longest_edge |
| Prompt (text) | `"Text Recognition:"` | `"Text Recognition:"` | Exact match |
| Prompt (table) | `"Table Recognition:"` | `"Table Recognition:"` | Exact match |
| Prompt (formula) | `"Formula Recognition:"` | `"Formula Recognition:"` | Exact match |
| System message | None | None | SDK doesn't add one |
| Image format | JPEG | JPEG | `config.yaml: image_format: JPEG` |

#### Notes

- **FP8 quantization** (`--quantization fp8`): Used for throughput. Not in official SDK, but GLM-OCR is small enough that FP8 quality impact is minimal.
- **MTP speculative decoding**: Official README recommends `--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`. We don't use it. Speed-only optimization, no quality impact.
- **Discrepancy in official configs**: `config.yaml` has `temperature=0.8`, `config.py` has `temperature=0.01`, `generation_config.json` has `do_sample=false`. We follow `generation_config.json` (greedy) as authoritative.

---

### 2. PaddleOCR-VL (`PaddlePaddle/PaddleOCR-VL`)

#### Official Sources

**generation_config.json:**
```json
{
  "eos_token_id": 2,
  "pad_token_id": 0,
  "use_cache": false
}
```
No sampling parameters specified — model uses framework defaults (greedy).

**preprocessor_config.json:**
```json
{
  "min_pixels": 147384,
  "max_pixels": 2822400,
  "patch_size": 14,
  "merge_size": 2,
  "image_processor_type": "PaddleOCRVLImageProcessor"
}
```

**PaddleX SDK prompts** (`pipeline.py:298-342`):
```python
"OCR:"                    # text
"Table Recognition:"      # table
"Formula Recognition:"    # formula
```

**PaddleX SDK predictor** (`predictor.py:474`):
```python
max_new_tokens = kwargs.get("max_new_tokens", 8192)
```

#### Resolved Configuration

| Parameter | Official Value | Our Setting | Rationale |
|-----------|---------------|-------------|-----------|
| temperature | 0.0 (default greedy) | 0.0 | No sampling params in generation_config |
| top_k | 1 | 1 | Default greedy |
| repetition_penalty | 1.0 (default) | 1.0 | No official override |
| presence_penalty | 0.0 (default) | 0.0 | No official override |
| frequency_penalty | 0.0 (default) | 0.0 | No official override |
| min_pixels | **147384** (384x384) | **147384** | `preprocessor_config.json` |
| max_pixels | **2822400** (1680x1680) | **2822400** | `preprocessor_config.json` |
| max_tokens | 8192 | 8192 | SDK predictor default |
| Prompt (text) | `"OCR:"` | `"OCR:"` | Exact match |
| Prompt (table) | `"Table Recognition:"` | `"Table Recognition:"` | Exact match |
| Prompt (formula) | `"Formula Recognition:"` | `"Formula Recognition:"` | Exact match |
| System message | None | None | SDK sends user-only messages |

#### Notes

- **No FP8**: PaddleOCR-VL is 0.9B params — too small for FP8 to be worthwhile.
- **Pixel limits were critically wrong before**: `min_pixels` was 12544 (11.7x too small) and `max_pixels` was 1003520 (2.8x too small). This caused significant resolution loss for small text regions.
- **`<fcel>` table tokens**: These are NOT special tokens (unlike MinerU OTSL), so `skip_special_tokens=True` (default) is correct — it won't strip them.

---

### 3. MinerU-VL (`opendatalab/MinerU2.5-2509-1.2B`)

#### Official Sources

**generation_config.json:**
```json
{
  "do_sample": true,
  "temperature": 0.01,
  "top_p": 0.001,
  "top_k": 1,
  "repetition_penalty": 1.0
}
```

**preprocessor_config.json:**
```json
{
  "min_pixels": 3136,
  "max_pixels": 1605632,
  "patch_size": 14,
  "merge_size": 2,
  "image_processor_type": "Qwen2VLImageProcessor"
}
```

**MinerUSamplingParams** (`mineru_client.py:19-29`):
```python
class MinerUSamplingParams(SamplingParams):
    def __init__(
        self,
        temperature: float | None = 0.0,
        top_p: float | None = 0.01,
        top_k: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        no_repeat_ngram_size: int | None = 100,
        max_new_tokens: int | None = None,
    ):
```

**Per-task overrides** (`mineru_client.py:50-54`):
```python
DEFAULT_SAMPLING_PARAMS = {
    "table":     MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation":  MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]":  MinerUSamplingParams(),  # no penalties
}
```

**Official prompts** (`mineru_client.py:43-48`):
```python
DEFAULT_PROMPTS = {
    "table":     "\nTable Recognition:",
    "equation":  "\nFormula Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]":  "\nLayout Detection:",
}
```

**System prompt** (`base_client.py:7`):
```python
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
```

#### Resolved Configuration

| Parameter | Official Value | Our Setting | Rationale |
|-----------|---------------|-------------|-----------|
| temperature | 0.0 | 0.0 | MinerUSamplingParams default |
| top_k | 1 | 1 | Match |
| repetition_penalty | 1.0 | 1.0 | Match |
| **presence_penalty** (all tasks) | **1.0** | **1.0** | Per-task defaults: table/equation/default all = 1.0 |
| **frequency_penalty** (table) | **0.005** | **0.005** | `DEFAULT_SAMPLING_PARAMS["table"]` |
| **frequency_penalty** (text/formula) | **0.05** | **0.05** | `DEFAULT_SAMPLING_PARAMS["[default]"]` and `["equation"]` |
| **no_repeat_ngram_size** | **100** (all tasks) | **100** (all tasks) | MinerUSamplingParams default, via logits processor |
| skip_special_tokens | False | False | Required for OTSL tokens (`<fcel>`, `<nl>`, etc.) |
| min_pixels | 3136 | 3136 | `preprocessor_config.json` |
| max_pixels | 1605632 | 1605632 | `preprocessor_config.json` |
| Prompt (text) | `"\nText Recognition:"` | `"\nText Recognition:"` | Exact match (leading `\n` is intentional) |
| Prompt (table) | `"\nTable Recognition:"` | `"\nTable Recognition:"` | Exact match |
| Prompt (formula) | `"\nFormula Recognition:"` | `"\nFormula Recognition:"` | Exact match |
| System message | `"You are a helpful assistant."` | `"You are a helpful assistant."` | Exact match |
| OTSL-to-HTML | `convert_otsl_to_html()` (table only) | `_try_otsl_to_html()` (table only) | Same function, with error handling |
| `<\|im_end\|>` strip | Implicit (framework handles) | Explicit strip | Defensive — no harm |

#### Notes

- **Logits processor**: `VllmV1NoRepeatNGramLogitsProcessor` is registered at vLLM server level (`--logits-processors` flag) and activated per-request via `vllm_xargs: {no_repeat_ngram_size: 100}`.
- **`generation_config.json` has `do_sample=true`** but with `top_k=1`, it's effectively greedy. The SDK overrides with `temperature=0.0` anyway.
- **FP8 quantization**: Used for 1.2B model — standard practice, minimal quality impact.

---

### 4. DeepSeek-OCR2 (`deepseek-ai/DeepSeek-OCR-2`)

#### Official Sources

**No `generation_config.json`** — HF confirmed absent (`.no_exist/` directory).

**`config.json`:**
```json
{
  "architectures": ["DeepseekOCR2ForCausalLM"],
  "model_type": "deepseek_vl_v2",
  "candidate_resolutions": [[1024, 1024]],
  "torch_dtype": "bfloat16",
  "max_position_embeddings": 8192
}
```
Architecture is DeepSeek-VL-V2, **NOT Qwen2VL**. Uses 1024×1024 fixed tile resolution.

**`processor_config.json`:**
```json
{
  "candidate_resolutions": [[1024, 1024]],
  "downsample_ratio": 4,
  "patch_size": 16,
  "processor_class": "DeepseekVLV2Processor"
}
```

**`modeling_deepseekocr2.py` — Official generate() calls:**
```python
# eval_mode=True (benchmarking):
output_ids = self.generate(
    temperature=0.0,
    max_new_tokens=8192,
    no_repeat_ngram_size=35,     # ← KEY: prevents 35-gram repetition
    use_cache=True,
    eos_token_id=tokenizer.eos_token_id,
)

# eval_mode=False (streaming):
# Same but no_repeat_ngram_size=20, skip_special_tokens=False via streamer
```

**Official prompts** (README + model code):
```python
"<image>\n<|grounding|>Convert the document to markdown."  # document parsing
"<image>\nFree OCR."                                        # text/formula/table
```

**Conversation template**: `sft_format='plain'` — no system message, no special roles.

#### Two Inference Paths

**Path A: `DeepSeekOCR2Client` (offline vLLM — `deepseek-ocr2` model key)**

| Parameter | Official | Our Setting | Status |
|-----------|----------|-------------|--------|
| temperature | 0.0 | 0.0 | ✅ |
| max_tokens | 8192 | 8192 | ✅ |
| **no_repeat_ngram_size** | **35** | N/A | ⚠️ Not available in vLLM SamplingParams |
| repetition handling | no_repeat_ngram | `RepetitionDetectionParams(max=40, min=5, count=3)` | ⚠️ Different mechanism (stop vs prevent) |
| skip_special_tokens | False | False | ✅ |
| quantization | BF16 | FP8 | Speed optimization |

`RepetitionDetectionParams` is the closest vLLM native equivalent. It detects repeating patterns and **stops generation** rather than **preventing** them. This is a compromise inherent to vLLM offline mode.

**Path B: `allgaznie-deepseek` (Allgaznie pipeline → vlm.py via vLLM server)**

| Parameter | Official | Our Setting | Status |
|-----------|----------|-------------|--------|
| temperature | 0.0 | 0.0 | ✅ |
| max_tokens | 8192 | 8192 | ✅ |
| **no_repeat_ngram_size** | **35** | **35** (via logits processor) | ✅ Fixed |
| Prompt (text/table/formula) | `"Free OCR."` | `"Free OCR."` | ✅ |
| Image preprocessing | 1024×1024 tiles (model handles) | Qwen2VL smart_resize then vLLM re-processes | ⚠️ Client-side resize is approximate; server applies correct processor |

#### Notes

- **`no_repeat_ngram_size` via logits processor**: `allgaznie-deepseek` config now uses `--logits-processors VllmV1NoRepeatNGramLogitsProcessor` (same as MinerU) with `vllm_xargs: {no_repeat_ngram_size: 35}` per-request.
- **Image pipeline mismatch**: Client-side `smart_resize` uses Qwen2VL parameters, but DeepSeek-VL-V2 has its own image processing (1024×1024 tiles + dynamic crops). The vLLM server's model processor handles the correct preprocessing, so client-side resize mainly controls transmission size.
- **`DeepSeekOCR2Client` limitation**: vLLM offline `SamplingParams` has no `no_repeat_ngram_size`. The `RepetitionDetectionParams` provides partial mitigation by stopping generation after detecting repetition, but cannot prevent it proactively like the official HF `generate()` does.

---

## What Was Wrong Before

### Issue 1: AllgaznieConfig Overriding VLMClient Defaults

**File**: `allgaznie/__init__.py`

```python
# BEFORE (incorrect — overrode VLMClient greedy defaults)
vlm_temperature: float = 0.1       # Should be 0.0
vlm_top_p: float = 0.1             # Should be 1.0
vlm_repetition_penalty: float = 1.1 # Should be 1.0 (model-specific in _infer_one)

# AFTER (fixed)
vlm_temperature: float = 0.0
vlm_top_p: float = 1.0
vlm_repetition_penalty: float = 1.0
```

**Why it mattered**: `AllgaznieOCR.__init__` explicitly passes config values to `VLMClient`, so config defaults override VLMClient defaults. The greedy patch to VLMClient was completely ineffective.

### Issue 2: Hardcoded Pixel Limits

**File**: `allgaznie/vlm.py`

```python
# BEFORE (one size fits all)
min_pixels: int = 12544    # Qwen2VL default
max_pixels: int = 1003520  # Qwen2VL default

# AFTER (per-model from preprocessor_config.json)
_MODEL_PIXEL_DEFAULTS = {
    "GLM-OCR":      (12544, 9633792),    # 3103px equivalent
    "PaddleOCR-VL": (147384, 2822400),   # 1680px equivalent
    "MinerU-VL":    (3136, 1605632),     # 1267px equivalent
}
```

**Impact on PaddleOCR-VL** (worst affected):
- `min_pixels`: was 12544, should be 147384 (11.7x too small)
- `max_pixels`: was 1003520, should be 2822400 (2.8x too small)
- Small region crops were sent at much lower resolution than the model expected

### Issue 3: Missing MinerU Penalties

**File**: `allgaznie/vlm.py` (`_infer_one`)

```python
# BEFORE (no penalties)
extra = {"top_k": 1, "repetition_penalty": 1.0}

# AFTER (official MinerU penalties)
presence_penalty = 1.0
frequency_penalty = 0.005 if task == "table" else 0.05
extra["vllm_xargs"] = {"no_repeat_ngram_size": 100}  # all tasks, not just table
```

### Issue 4: GLM-OCR repetition_penalty

**File**: `allgaznie/vlm.py` (`_infer_one`)

```python
# BEFORE (generic default)
extra = {"repetition_penalty": 1.0}

# AFTER (model-specific)
if self.display_name == "GLM-OCR":
    extra["repetition_penalty"] = 1.1  # from SDK config.py and config.yaml
```

### Issue 5: DeepSeek-OCR2 missing no_repeat_ngram_size

**Files**: `allgaznie/vlm.py`, `config.py`

Official `modeling_deepseekocr2.py` uses `no_repeat_ngram_size=35` for eval mode. vLLM's `SamplingParams` doesn't support this natively. Fix: reuse MinerU's `VllmV1NoRepeatNGramLogitsProcessor`.

```python
# config.py — allgaznie-deepseek: added logits-processors
"--logits-processors", "mineru_vl_utils.logits_processor.vllm_v1_no_repeat_ngram:VllmV1NoRepeatNGramLogitsProcessor",

# vlm.py — _infer_one: added DeepSeek-specific ngram
elif self.display_name == "DeepSeek-OCR2":
    extra["vllm_xargs"] = {"no_repeat_ngram_size": 35}
```

---

## vLLM Server Arguments Audit

### allgaznie-glm

```
--trust-remote-code
--no-enable-prefix-caching     # Each region has unique image — no prefix reuse
--mm-processor-cache-gb 0      # Saves GPU memory (no image cache reuse)
--max-model-len 16384
--max-num-batched-tokens 65536 # High throughput for small model
--max-num-seqs 1024            # High concurrency for small model
--gpu-memory-utilization 0.85
--quantization fp8             # Speed optimization for 2B model
```

### allgaznie-paddle

```
--trust-remote-code
--no-enable-prefix-caching
--mm-processor-cache-gb 0
--max-num-batched-tokens 16384 # Conservative (official CUDA default is 131072)
--max-model-len 16384
--gpu-memory-utilization 0.85
```
No FP8 — model is only 0.9B params.

### allgaznie-mineru

```
--trust-remote-code
--gpu-memory-utilization 0.85
--max-num-batched-tokens 16384
--max-num-seqs 256
--max-model-len 16384
--quantization fp8
--logits-processors mineru_vl_utils.logits_processor.vllm_v1_no_repeat_ngram:VllmV1NoRepeatNGramLogitsProcessor
```

The logits processor implements `no_repeat_ngram_size` at the vLLM engine level, activated per-request via `vllm_xargs`.

### allgaznie-deepseek

```
--trust-remote-code
--max-model-len 8192
--max-num-seqs 16
--gpu-memory-utilization 0.85
--quantization fp8
--logits-processors mineru_vl_utils.logits_processor.vllm_v1_no_repeat_ngram:VllmV1NoRepeatNGramLogitsProcessor
```

Same logits processor as MinerU, but activated with `no_repeat_ngram_size=35` (official eval mode) per-request.

---

## Files Modified

| File | Changes |
|------|---------|
| `allgaznie/__init__.py:50-55` | `temperature=0.0, top_p=1.0, repetition_penalty=1.0, min/max_pixels=0` (auto-detect) |
| `allgaznie/vlm.py:50-57` | Added `_MODEL_PIXEL_DEFAULTS` with per-model official values |
| `allgaznie/vlm.py:129-132` | Pixel auto-resolution: `0 → model-specific default` |
| `allgaznie/vlm.py:167-180` | Model-specific penalties: GLM-OCR rep=1.1, MinerU presence/frequency/ngram, DeepSeek ngram=35 |
| `allgaznie/vlm.py:186-187` | `presence_penalty` and `frequency_penalty` as top-level API params |
| `config.py:101-113` | `allgaznie-deepseek`: added `--logits-processors` for no_repeat_ngram |

---

## Verification

All configurations verified programmatically against official sources:

```
GLM-OCR:       prompts OK, system_msg OK, temp=0.0, rep=1.1, pixels=12544/9633792
PaddleOCR-VL:  prompts OK, system_msg OK, temp=0.0, rep=1.0, pixels=147384/2822400
MinerU-VL:     prompts OK, system_msg OK, temp=0.0, rep=1.0, pres=1.0, freq=task-specific,
               ngram=100(all), skip_special=False, pixels=3136/1605632
DeepSeek-OCR2: prompts OK, system_msg=None, temp=0.0, ngram=35(eval),
               pixels=client-side approximate (vLLM server re-processes to 1024×1024 tiles)
```
