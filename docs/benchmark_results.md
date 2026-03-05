# OCR Benchmark Results Report

**Date**: 2026-03-04
**Environment**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM), vLLM 0.16.1rc1
**Pipeline**: Allgaznie OCR (PP-DocLayoutV3 layout → region crop → per-region VLM → markdown)

---

## 1. Models Evaluated

| Key | Model | Params | Mode | Quantization |
|-----|-------|--------|------|-------------|
| `glm-ocr` | GLM-OCR | 1.3B | VLM-only | FP8 |
| `paddleocr-vl` | PaddleOCR-VL | 1.0B | VLM-only | BF16 |
| `deepseek-ocr2` | DeepSeek-OCR-2 | 3.4B | VLM-only | FP8 |
| `allgaznie-glm` | Allgaznie + GLM-OCR | 1.3B + layout | Pipeline | FP8 |
| `allgaznie-paddle` | Allgaznie + PaddleOCR-VL | 1.0B + layout | Pipeline | BF16 |
| `allgaznie-deepseek` | Allgaznie + DeepSeek-OCR-2 | 3.4B + layout | Pipeline | FP8 |

**VLM-only**: Full-page image → VLM → text (단일 패스)
**Pipeline**: Image → Layout Detection → Region Crop → Per-region VLM → Markdown Assembly (2-stage)

---

## 2. Document Parsing Benchmarks (Pipeline Only)

### 2.1 OmniDocBench (1,358 samples)

Edit Distance (↑ higher = better, 1-NED score):

| Metric | Allgaznie-GLM | Allgaznie-Paddle | Allgaznie-DeepSeek |
|--------|:---:|:---:|:---:|
| **Text Block Edit Dist** | **0.083** | **0.080** | 0.112 |
| **Table TEDS** | **0.887** | 0.856 | 0.590 |
| **Table Structure TEDS** | **0.909** | 0.888 | 0.639 |
| **Display Formula Edit Dist** | 0.770 | 0.775 | **0.800** |
| **Reading Order Edit Dist** | 0.288 | **0.284** | 0.296 |

### 2.2 Upstage DP-Bench (201 samples)

| Metric | Allgaznie-GLM | Allgaznie-Paddle | Allgaznie-DeepSeek |
|--------|:---:|:---:|:---:|
| **NID** | **0.870** | 0.817 | 0.865 |
| **TEDS** | **0.936** | 0.839 | 0.733 |
| **TEDS Structure** | **0.965** | 0.854 | 0.781 |

### 2.3 Nanonets-KIE (988 samples)

| Metric | Allgaznie-GLM | Allgaznie-Paddle | Allgaznie-DeepSeek |
|--------|:---:|:---:|:---:|
| **ANLS** | 0.792 | **0.817** | 0.784 |

---

## 3. VLM-Only Benchmarks

### 3.1 OCRBench (1,000 samples)

| Model | Accuracy |
|-------|:---:|
| **GLM-OCR** | **0.836** |
| PaddleOCR-VL | 0.710 |
| DeepSeek-OCR-2 | 0.481 |

### 3.2 UniMERNet Formula Recognition (200 samples)

| Model | Corpus BLEU | Edit Distance |
|-------|:---:|:---:|
| **PaddleOCR-VL** | **0.897** | **0.083** |
| GLM-OCR | 0.744 | 0.221 |
| DeepSeek-OCR-2 | 0.416 | 0.389 |

### 3.3 PubTabNet Table Recognition (200 samples)

| Model | TEDS | TEDS Structure |
|-------|:---:|:---:|
| **PaddleOCR-VL** | **0.714** | **0.927** |
| GLM-OCR | 0.704 | 0.922 |
| DeepSeek-OCR-2 | 0.689 | 0.893 |

### 3.4 TEDS_TEST Table Recognition (200 samples)

| Model | TEDS | TEDS Structure |
|-------|:---:|:---:|
| **PaddleOCR-VL** | **0.711** | GLM-OCR: **0.913** |
| GLM-OCR | 0.707 | 0.913 |
| DeepSeek-OCR-2 | 0.675 | 0.862 |

### 3.5 Handwritten Forms (200 samples)

| Model | CER ↓ | WER ↓ |
|-------|:---:|:---:|
| **GLM-OCR** | **0.034** | PaddleOCR-VL: **0.126** |
| PaddleOCR-VL | 0.048 | 0.126 |
| DeepSeek-OCR-2 | 0.203 | 0.353 |

---

## 4. Latency Summary

### 4.1 Pipeline Mode (Layout + VLM)

| Benchmark | Allgaznie-GLM | Allgaznie-Paddle | Allgaznie-DeepSeek |
|-----------|:---:|:---:|:---:|
| OmniDocBench | 632ms | 694ms | 1,810ms |
| DP-Bench | 357ms | 392ms | 714ms |
| Nanonets-KIE | 599ms | 660ms | 1,320ms |

### 4.2 VLM-Only Mode

| Benchmark | GLM-OCR | PaddleOCR-VL | DeepSeek-OCR-2 |
|-----------|:---:|:---:|:---:|
| OCRBench | ~800ms | 855ms | 88ms* |
| UniMERNet | ~400ms | 319ms | 116ms* |
| PubTabNet | ~700ms | 652ms | 141ms* |
| TEDS_TEST | ~700ms | 765ms | 167ms* |
| Handwritten | ~100ms | 47ms | 106ms* |

*DeepSeek-OCR-2 uses offline batch mode (vllm.LLM) — latency is per-sample in batch, not comparable to HTTP API latency.

---

## 5. Key Findings

### 5.1 Pipeline vs VLM-Only

Allgaznie pipeline (2-stage) enables **document parsing** capabilities that VLM-only models lack:
- OmniDocBench, DP-Bench, Nanonets-KIE are only measurable with pipeline mode
- Layout detection + region cropping preserves document structure and reading order

### 5.2 Model Comparison

**GLM-OCR**: Best overall for document parsing. Strongest table recognition (TEDS 0.887) and text accuracy (OCRBench 0.836). Best for general-purpose OCR.

**PaddleOCR-VL**: Best formula recognition (UniMERNet BLEU 0.897), competitive table recognition, and best KIE extraction (ANLS 0.817). Best for scientific documents.

**DeepSeek-OCR-2**: Best formula edit distance in OmniDocBench (0.800) but significantly weaker in text recognition (OCRBench 0.481) and table recognition. 3.4B model size doesn't translate to better performance in our benchmarks. Note: vLLM server mode was unsupported for this model, requiring offline batch inference.

### 5.3 Notable Observations

1. **Text block recognition**: Allgaznie-Paddle (0.080) slightly edges Allgaznie-GLM (0.083) on OmniDocBench text blocks
2. **Formula recognition**: DeepSeek-OCR-2 leads in OmniDocBench display formula (0.800 > 0.775 > 0.770), but trails badly in standalone formula (UniMERNet BLEU 0.416)
3. **Table recognition gap**: GLM-OCR (TEDS 0.887) >> Paddle (0.856) >> DeepSeek (0.590) on OmniDocBench
4. **Handwriting**: GLM-OCR excels (CER 3.4%), PaddleOCR-VL acceptable (4.8%), DeepSeek poor (20.3%)

---

## 6. Comparison with Official Reported Scores

Official scores are from model cards / papers. Our eval uses the same benchmarks but may differ in:
- Prompt format (we use per-region prompts for pipeline, model-specific prompts for VLM-only)
- Post-processing (we apply model-specific post-processing where documented)
- Sample selection (we use full datasets or fixed subsets)

| Benchmark | Metric | Official GLM-OCR | Our GLM-OCR | Official PaddleOCR-VL | Our PaddleOCR-VL |
|-----------|--------|:---:|:---:|:---:|:---:|
| OCRBench | Accuracy | ~0.94* | 0.836 | — | 0.710 |
| OmniDocBench | Text Edit Dist | — | 0.083 (pipeline) | 0.035* | 0.080 (pipeline) |
| OmniDocBench | Table TEDS | — | 0.887 (pipeline) | 0.928* | 0.856 (pipeline) |
| PubTabNet | TEDS | — | 0.704 | — | 0.714 |

*Official scores may use different evaluation protocols, full-page inference, or internal test sets. Our pipeline uses 2-stage (layout + per-region VLM) which changes the evaluation dynamics.

**Gap analysis**:
- OCRBench gap (0.94 → 0.836): Our VLM-only mode sends the full image with a generic prompt. Official eval likely uses task-specific prompts and may include chain-of-thought.
- OmniDocBench gap: Our pipeline mode uses layout detection to segment regions, then per-region VLM. Official full-page inference avoids layout errors but cannot provide structured output.

---

## 7. Infrastructure Notes

### subprocess.PIPE Deadlock Fix
The vLLM server managed by `server.py` was using `stdout=subprocess.PIPE` which caused the 64KB pipe buffer to fill, blocking vLLM's event loop and causing complete server deadlock after ~50-70 images. Fixed by redirecting to log file.

### 1x1 Placeholder Fix
Empty bbox crops (from layout detection) were creating 1x1 pixel placeholder images. PaddleOCR-VL's image processor raised `ValueError` on these, contributing to vLLM internal state corruption. Fixed by returning `None` and skipping in the pipeline.

### DeepSeek-OCR-2 vLLM Compatibility
`DeepseekOCR2ForCausalLM` architecture required `kaldi-native-fbank` dependency. The model cannot run in vLLM HTTP server mode for the allgaznie pipeline, only in offline batch mode via `vllm.LLM`.
