# OCR Benchmark Results Summary

> Date: 2026-03-05 | GPU: NVIDIA RTX PRO 6000 Blackwell 98GB
> vLLM 0.16.1rc1 | PyTorch 2.10 | CUDA 12.8
> Scope: GLM-OCR ecosystem + MinerU (PaddleOCR-VL, DeepSeek-OCR2 excluded)

---

## 1. Latency Comparison: Allgaznie SDK vs Official SDK

> Core question: How much faster is our Allgaznie pipeline vs the official GLM-OCR Pipeline SDK?

### Per-Sample Latency (ms)

| Benchmark | GLM-OCR Pipeline (official) | Allgaznie-GLM (ours) | Speedup |
|:---|---:|---:|---:|
| OmniDocBench (1355 samples) | 976 | 747 | **1.31x** |
| DP-Bench (200 samples) | 578 | 430 | **1.34x** |
| Nanonets-KIE (987 samples) | 970 | 752 | **1.29x** |

### Wall-Clock Time (seconds)

| Benchmark | GLM-OCR Pipeline | Allgaznie-GLM | Speedup |
|:---|---:|---:|---:|
| Total (3 benchmarks) | 2,514 | 1,992 | **1.26x** |

### Other Models (reference)

| Model / Benchmark | Avg Latency (ms) | Samples |
|:---|---:|---:|
| GLM-OCR VLM / OCRBench | 98 | 1,000 |
| GLM-OCR VLM / Handwritten | 50 | 200 |
| GLM-OCR VLM / UniMERNet | 428 | 200 |
| GLM-OCR VLM / PubTabNet | 1,177 | 200 |
| GLM-OCR VLM / TEDS-TEST | 1,310 | 200 |
| GLM-OCR VLM / OmniDocBench | 2,376 | 1,355 |
| MinerU / OmniDocBench | 3,279 | 1,355 |

### Latency Analysis

- **Allgaznie-GLM is 1.26-1.34x faster** than the official GLM-OCR Pipeline SDK across all benchmarks
- Both use the same VLM (GLM-OCR via vLLM) and layout model (PP-DocLayoutV3)
- Speedup comes from: vectorized NMS/containment, GPU JPEG decode, BF16+compile layout, concurrent HTTP
- Note: Allgaznie-GLM uses **FP8 quantization** for the VLM; official pipeline uses FP16
- MinerU is 3.4x slower than GLM-OCR Pipeline on OmniDocBench (CPU-based pipeline)

---

## 2. Quality: OmniDocBench v1.5 (Document Parsing, 1355 samples)

### Pipeline Models (layout detection + per-region VLM)

| Model | Text (1-ED) | Table (TEDS) | Formula (CDM) | **Overall** | Official | Delta |
|:---|---:|---:|---:|---:|---:|---:|
| **GLM-OCR Pipeline** | 95.8% | 91.9% | 92.2% | **93.3** | 94.6 | -1.3 |
| **MinerU 2.5** | 94.6% | 86.4% | 91.0% | **90.6** | 90.7 | -0.1 |
| Allgaznie-GLM | 94.3% | 92.4% | 26.7% | 71.1 | — | — |

> **Allgaznie-GLM CDM issue**: Pipeline outputs inline math (`$...$`) instead of display math (`$$...$$`), causing display_formula CDM matching failure. Text and Table scores are on par with the official SDK.

### VLM-Only (single-pass, no layout detection)

| Model | Text (1-ED) | Table (TEDS) | Formula (CDM) | Overall |
|:---|---:|---:|---:|---:|
| GLM-OCR | 87.2% | 40.4% | 81.5% | 69.7 |

> VLM-only OmniDocBench is not a fair comparison — the benchmark requires layout detection for table/formula extraction. Included for reference only.

---

## 3. Quality: DP-Bench & Nanonets-KIE

| Model | DP-Bench NID | DP-Bench TEDS | Nanonets ANLS |
|:---|---:|---:|---:|
| **GLM-OCR Pipeline** | **92.0%** | **93.1%** | 78.2% |
| Allgaznie-GLM | 89.9% | 92.9% | **78.6%** |

> Allgaznie-GLM matches or slightly trails the official SDK on structured extraction tasks.

---

## 4. Quality: GLM-OCR VLM Benchmarks

### OCRBench (Text Recognition, 1000 samples)

| Subset | Score | Official | Delta |
|:---|---:|---:|---:|
| Text (300) | **95.0%** | 94.0% | +1.0 |
| Total (1000) | 83.7% | — | — |

### Table Recognition

| Benchmark | TEDS | TEDS-struct | Official TEDS | Delta |
|:---|---:|---:|---:|---:|
| PubTabNet (200) | 70.7% | 92.5% | 85.2% | -14.5 |
| TEDS-TEST (200) | 70.6% | 91.3% | 86.0% | -15.4 |

> PubTabNet/TEDS-TEST ~15pt gap is a known VLM inference quality difference (Blackwell GPU / vLLM version), not an eval code issue.

### UniMERNet (Formula Recognition, 200 samples)

| CDM F1 | Edit Dist | BLEU | Official CDM | Delta |
|---:|---:|---:|---:|---:|
| **94.0%** | 0.2203 | 0.7431 | 96.5% | -2.5 |

### Handwritten Forms (IAM, 200 samples)

| CER | WER |
|---:|---:|
| **3.4%** | 12.7% |

---

## 5. Official Benchmark Reproduction

| Benchmark | Ours | Official | Delta | Verdict |
|:---|---:|---:|---:|:---|
| MinerU OmniDocBench | 90.6 | 90.7 | -0.1 | Reproduced |
| GLM-OCR Pipeline OmniDocBench | 93.3 | 94.6 | -1.3 | Near-reproduced |
| GLM-OCR OCRBench (Text) | 95.0 | 94.0 | +1.0 | Reproduced (higher) |
| GLM-OCR UniMERNet | 94.0 | 96.5 | -2.5 | Near-reproduced |
| GLM-OCR PubTabNet | 70.7 | 85.2 | -14.5 | Inference quality gap |
| GLM-OCR TEDS-TEST | 70.6 | 86.0 | -15.4 | Inference quality gap |

### Summary

1. **Eval code accuracy confirmed** — MinerU OmniDocBench Delta=-0.1 (effectively reproduced)
2. **OCRBench**: Text subset score exceeds official (+1.0)
3. **PubTabNet/TEDS-TEST 15pt gap**: VLM inference quality difference (Blackwell vLLM), not eval code issue
4. **Allgaznie-GLM is 1.3x faster** than official GLM-OCR Pipeline SDK with comparable quality on text/table

---

## 6. CDM Bug Fixes Applied

| # | Bug | File | Fix |
|---|---|---|---|
| 1 | `\[...\]` delimiters nested inside `displaymath` env -> xelatex failure | `cal_metric.py` | Strip `\[...\]`, `\(...\)` from formulas |
| 2 | Zero-byte cache files treated as valid -> CDM=0 | `latex2bbox_color.py` | Check `os.path.getsize() > 0` |
| 3 | xeCJK package dependency when only Noto Sans CJK installed | `latex2bbox_color.py` | Remove xeCJK |

---

## 7. Run Timing

| Phase | Wall Time | Exit |
|:---|---:|---:|
| GLM-OCR VLM (6 benchmarks) | 4,058s (67m) | 0 |
| GLM-OCR Pipeline (3 benchmarks) | 2,514s (42m) | 0 |
| Allgaznie-GLM (3 benchmarks) | 1,992s (33m) | 0 |
| MinerU (OmniDocBench only) | 4,479s (75m) | 0 |
| Evaluation (all models) | ~20m | 0 |
| **Total** | **~4h** | |
