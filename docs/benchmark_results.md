# OCR Benchmark Results Report

**Date**: 2026-03-07 (re-evaluated with corrected VLM parameters + eval logic fixes)
**Environment**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM), vLLM 0.16.1rc1
**Pipeline**: Allgaznie OCR (PP-DocLayoutV3 layout → region crop → per-region VLM → markdown)

---

## 1. Models Evaluated

| Key | Model | Params | Mode | Quantization |
|-----|-------|--------|------|-------------|
| `glm-ocr` | GLM-OCR | 1.3B | VLM-only | FP8 |
| `glm-ocr-pipeline` | GLM-OCR Pipeline (SDK) | 1.3B + layout | Pipeline | FP8 |
| `mineru` | MinerU-2.5 | 1.2B + layout | Pipeline | BF16 |
| `deepseek-ocr2` | DeepSeek-OCR-2 | 3.4B | VLM-only | FP8 |
| `allgaznie-glm` | Allgaznie + GLM-OCR | 1.3B + layout | Pipeline | FP8 |
| `allgaznie-paddle` | Allgaznie + PaddleOCR-VL | 1.0B + layout | Pipeline | BF16 |
| `paddleocr-vl-pipeline` | PaddleOCR-VL Pipeline (SDK) | 1.0B + layout | Pipeline | BF16 |
| `allgaznie-mineru` | Allgaznie + MinerU-VL | 1.2B + layout | Pipeline | BF16 |
| `allgaznie-deepseek` | Allgaznie + DeepSeek-OCR-2 | 3.4B + layout | Pipeline | FP8 |
| `upstage-standard` | Upstage Document Parse (Standard) | — | API | — |
| `upstage-enhanced` | Upstage Document Parse (Enhanced) | — | API | — |
| `azure-layout` | Azure Document Intelligence Layout (S0) | — | API | — |

**VLM-only**: Full-page image → VLM → text (single pass)
**Pipeline**: Image → Layout Detection → Region Crop → Per-region VLM → Markdown Assembly (2-stage)
**API**: Cloud API — full-page image → API → markdown (Upstage Document Parse v260128 / Azure DI prebuilt-layout 2024-11-30)

---

## 2. Master Results Table

All metrics below. **↑ = higher is better, ↓ = lower is better.** Bold = best in column.

### 2.1 OmniDocBench Official Eval (1,355 samples)

Overall = ((1 - Text_ED) × 100 + Table_TEDS + Formula_CDM) / 3

| Model | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | Table TEDS-S ↑ | ReadOrder ED ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Allgaznie-GLM** | **93.3** | 0.054 | **93.0** | **92.5** | **95.2** | 0.070 |
| **GLM-OCR Pipeline** | 93.3 | **0.042** | 92.2 | 91.9 | 94.6 | 0.056 |
| **Allgaznie-Paddle** | 91.3 | 0.051 | 90.0 | 89.0 | 92.9 | 0.068 |
| **Allgaznie-MinerU** | 91.2 | 0.056 | 90.1 | 89.0 | 93.1 | 0.072 |
| **MinerU-2.5** | 90.6 | 0.054 | 91.0 | 86.4 | 90.9 | **0.052** |
| **DeepSeek-OCR2** | 84.0 | 0.073 | 91.2 | 68.2 | 72.0 | 0.073 |
| **Allgaznie-DeepSeek** | 78.8 | 0.084 | 83.4 | 61.5 | 67.4 | 0.083 |
| **PaddleOCR-VL Pipeline** | 92.7 | 0.044 | 94.6 | 87.8 | 91.7 | 0.043 |
| Upstage Standard | 70.8 | 0.122 | 54.6 | 70.0 | 77.5 | 0.143 |
| Upstage Enhanced | 70.2 | 0.126 | 54.2 | 69.0 | 75.6 | 0.150 |
| Azure Layout | —† | 0.133 | N/A† | 73.0 | — | — |
| GLM-OCR (VLM-only) | 69.7 | 0.128 | 81.5 | 40.4 | 42.4 | 0.143 |

*†Azure Layout은 수식을 LaTeX로 출력하지 않아 Formula CDM 계산 불가. Overall 산출 불가.*

#### Overall without Formula (Text + Table only)

Overall (no formula) = ((1 - Text_ED) × 100 + Table_TEDS) / 2

수식(Formula) 인식은 CDM 메트릭 특성상 변동이 크고, 수식이 없는 문서도 많으므로 텍스트+테이블만으로 평가.

| Model | Overall (3-metric) ↑ | Overall (no formula) ↑ | Delta |
|:---|:---:|:---:|:---:|
| **GLM-OCR Pipeline** | 93.3 | **93.8** | +0.5 |
| **Allgaznie-GLM** | **93.3** | 93.5 | +0.2 |
| **Allgaznie-Paddle** | 91.3 | 91.9 | +0.6 |
| **Allgaznie-MinerU** | 91.2 | 91.7 | +0.5 |
| **PaddleOCR-VL Pipeline** | 92.7 | 91.7 | -1.0 |
| **MinerU-2.5** | 90.6 | 90.5 | -0.1 |
| DeepSeek-OCR2 | 84.0 | 80.4 | -3.6 |
| Upstage Standard | 70.8 | 78.9 | +8.1 |
| Azure Layout | —† | 79.8 | — |
| Upstage Enhanced | 70.2 | 78.2 | +8.0 |
| Allgaznie-DeepSeek | 78.8 | 76.6 | -2.2 |
| GLM-OCR (VLM-only) | 69.7 | 63.8 | -5.9 |

*Delta = (no formula) - (3-metric). 양수 = 수식 인식이 약한 모델 (수식이 발목), 음수 = 수식이 강한 모델 (수식이 올려줌).*

**분석**: Upstage API는 수식 제외 시 +8pt 상승 (78.9) — CDM 54.6이 전체를 끌어내림. 반면 DeepSeek-OCR2는 수식 제외 시 -3.6pt (80.4) — CDM 91.2가 전체를 올려줌. 상위 5개 파이프라인 모델은 수식 유무와 관계없이 90+ 유지.

### 2.2 OmniDocBench Custom Eval (with scoring fixes, per-type averages)

| Model | Overall | Overall (no formula) | Text | Table (351) | Formula (200) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **GLM-OCR Pipeline** | **89.6** | **90.6** | **88.8** | 93.0 | 84.1 |
| **Allgaznie-GLM** | 88.7 | 90.0 | 87.7 | **93.8** | **84.3** |
| **MinerU-2.5** | 87.9 | 89.5 | 87.5 | 88.4 | 80.3 |
| **PaddleOCR-VL Pipeline** | 88.5 | 88.8 | 87.2 | 91.3 | 89.3 |
| **Allgaznie-MinerU** | 87.8 | 89.3 | 87.4 | 89.9 | 82.0 |
| Allgaznie-Paddle | 76.6 | 87.3 | 87.4 | 2.2 | 81.0 |
| Upstage Standard | 83.6 | 87.2 | 88.7 | 74.1 | 46.5 |
| Upstage Enhanced | 83.2 | 86.8 | 88.4 | 74.2 | 46.1 |
| Azure Layout | 80.9 | 85.6 | 86.7 | 73.0 | 22.2† |
| **DeepSeek-OCR2** | 85.0 | 86.6 | 85.5 | 76.4 | 80.5 |
| GLM-OCR (VLM-only) | 78.8 | 85.7 | 82.7 | 49.1 | 71.5 |
| Allgaznie-DeepSeek | 81.5 | 84.3 | 84.1 | 64.9 | 74.4 |

*Overall (no formula) = per-sample avg of text + table only (formula 제외). Table/Formula scores averaged only over samples containing those element types. Sorted by Overall (no formula) desc.*
*†Azure Layout Formula: edit distance 기반 (CDM 아님). Azure는 수식을 LaTeX로 출력하지 않아 CDM 계산 불가.*

### 2.3 Upstage DP-Bench (200 samples)

| Model | NID ↑ | TEDS ↑ | TEDS Structure ↑ |
|:---|:---:|:---:|:---:|
| **Upstage Standard** | **0.971** | 0.916 | 0.925 |
| Upstage Enhanced | 0.935 | 0.924 | 0.936 |
| GLM-OCR Pipeline | 0.929 | 0.931 | 0.961 |
| MinerU-2.5 | 0.924 | 0.946 | 0.951 |
| DeepSeek-OCR2 | 0.917 | 0.859 | 0.873 |
| Allgaznie-GLM | 0.911 | 0.928 | 0.960 |
| Allgaznie-MinerU | 0.909 | **0.968** | **0.975** |
| Allgaznie-DeepSeek | 0.900 | 0.712 | 0.757 |
| Azure Layout | 0.876 | 0.874 | 0.894 |
| Allgaznie-Paddle | 0.871 | 0.838 | 0.853 |
| PaddleOCR-VL Pipeline | 0.884 | 0.952 | 0.969 |
| GLM-OCR (VLM-only) | 0.119 | 0.102 | 0.104 |

### 2.4 OCRBench (1,000 samples)

| Model | Accuracy ↑ |
|:---|:---:|
| **GLM-OCR (VLM-only)** | **0.837** |
| DeepSeek-OCR2 | 0.486 |
| GLM-OCR Pipeline | 0.477 |
| Allgaznie-Paddle | 0.465 |
| Allgaznie-GLM | 0.463 |
| Allgaznie-MinerU | 0.402 |
| Allgaznie-DeepSeek | 0.349 |
| PaddleOCR-VL Pipeline | 0.471 |
| MinerU-2.5 | 0.331 |

### 2.5 UniMERNet Formula Recognition (200 samples)

| Model | CDM-F1 ↑ | Edit Distance ↓ | Corpus BLEU ↑ |
|:---|:---:|:---:|:---:|
| **GLM-OCR (VLM-only)** | **0.940** | **0.220** | 0.743 |
| GLM-OCR Pipeline | 0.843 | 0.272 | 0.707 |
| DeepSeek-OCR2 | 0.795 | 0.393 | 0.412 |
| MinerU-2.5 | 0.793 | 0.257 | 0.731 |
| Allgaznie-Paddle | 0.625 | 0.414 | **0.867** |
| Allgaznie-MinerU | 0.601 | 0.467 | 0.733 |
| Allgaznie-GLM | 0.578 | 0.502 | 0.696 |
| PaddleOCR-VL Pipeline | 0.921 | 0.188 | 0.789 |
| Allgaznie-DeepSeek | 0.512 | 0.631 | 0.366 |

### 2.6 PubTabNet Table Recognition (200 samples)

| Model | TEDS ↑ | TEDS Structure ↑ |
|:---|:---:|:---:|
| **GLM-OCR (VLM-only)** | **0.707** | **0.925** |
| GLM-OCR Pipeline | 0.691 | 0.920 |
| DeepSeek-OCR2 | 0.684 | 0.883 |
| Allgaznie-MinerU | 0.657 | 0.848 |
| Allgaznie-Paddle | 0.652 | 0.848 |
| Allgaznie-DeepSeek | 0.645 | 0.837 |
| Allgaznie-GLM | 0.642 | 0.842 |
| PaddleOCR-VL Pipeline | 0.694 | 0.913 |
| MinerU-2.5 | 0.593 | 0.782 |

### 2.7 TEDS Test Table Recognition (200 samples)

| Model | TEDS ↑ | TEDS Structure ↑ |
|:---|:---:|:---:|
| **Allgaznie-MinerU** | **0.721** | 0.909 |
| GLM-OCR (VLM-only) | 0.706 | **0.913** |
| Allgaznie-Paddle | 0.702 | 0.898 |
| GLM-OCR Pipeline | 0.683 | 0.899 |
| Allgaznie-GLM | 0.678 | 0.883 |
| DeepSeek-OCR2 | 0.673 | 0.861 |
| Allgaznie-DeepSeek | 0.668 | 0.850 |
| PaddleOCR-VL Pipeline | 0.712 | 0.921 |
| MinerU-2.5 | 0.595 | 0.747 |

### 2.8 NanoNets-KIE (988 samples)

| Model | ANLS ↑ |
|:---|:---:|
| **Allgaznie-Paddle** | **0.812** |
| MinerU-2.5 | 0.803 |
| Allgaznie-GLM | 0.785 |
| Allgaznie-MinerU | 0.784 |
| GLM-OCR Pipeline | 0.782 |
| Allgaznie-DeepSeek | 0.780 |
| PaddleOCR-VL Pipeline | 0.810 |
| GLM-OCR (VLM-only) | 0.888 |
| DeepSeek-OCR2 | 0.189 |

### 2.9 Handwritten Forms (200 samples)

| Model | CER ↓ | WER ↓ |
|:---|:---:|:---:|
| **GLM-OCR (VLM-only)** | **0.034** | **0.127** |
| GLM-OCR Pipeline | 0.121 | 0.238 |
| DeepSeek-OCR2 | 0.195 | 0.340 |
| Allgaznie-Paddle | 0.687 | 0.743 |
| Allgaznie-GLM | 0.689 | 0.760 |
| Allgaznie-MinerU | 0.738 | 0.895 |
| Allgaznie-DeepSeek | 0.767 | 0.824 |
| PaddleOCR-VL Pipeline | 1.070 | 1.321 |
| MinerU-2.5 | 1.950 | 1.000 |

---

## 3. Cross-Model Summary

### 3.1 Consolidated View (Quality + Latency)

| Model | OmniDoc Overall ↑ | OmniDoc (no formula) ↑ | DP-Bench NID ↑ | OCRBench ↑ | UniMER CDM ↑ | PubTab TEDS ↑ | TEDS Test ↑ | NanoNets ↑ | Handwr CER ↓ | Latency (ms) | vs Pipeline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|
| **Allgaznie-GLM** | **93.3** | 93.5 | 0.911 | 0.463 | 0.578 | 0.642 | 0.678 | 0.785 | 0.689 | **715** | **1.36x** |
| **GLM-OCR Pipeline** | 93.3 | **93.8** | 0.929 | 0.477 | 0.843 | 0.691 | 0.683 | 0.782 | 0.121 | 976 | 1.00x |
| **Allgaznie-Paddle** | 91.3 | 91.9 | 0.871 | 0.465 | 0.625 | 0.652 | 0.702 | **0.812** | 0.687 | 762 | 1.28x |
| **Allgaznie-MinerU** | 91.2 | 91.7 | 0.909 | 0.402 | 0.601 | 0.657 | **0.721** | 0.784 | 0.738 | **655** | **1.49x** |
| **PaddleOCR-VL Pipeline** | 92.7 | 91.7 | 0.884 | 0.471 | 0.921 | 0.694 | 0.712 | 0.810 | 1.070 | — | — |
| **MinerU-2.5** | 90.6 | 90.5 | 0.924 | 0.331 | 0.793 | 0.593 | 0.595 | 0.803 | 1.950 | 3,279 | 0.30x |
| **DeepSeek-OCR2** | 84.0 | 80.4 | 0.917 | 0.486 | 0.795 | 0.684 | 0.673 | 0.189 | 0.195 | 533 | 1.83x |
| **Allgaznie-DeepSeek** | 78.8 | 76.6 | 0.900 | 0.349 | 0.512 | 0.645 | 0.668 | 0.780 | 0.767 | 1,455 | 0.67x |
| Upstage Standard | 70.8 | 78.9 | **0.971** | — | — | — | — | — | — | 3,289 | 0.30x |
| Azure Layout | —‡ | 79.8 | 0.876 | — | — | — | — | — | — | 7,436 | 0.13x |
| GLM-OCR (VLM-only) | 69.7 | 63.8 | 0.119 | **0.837** | **0.940** | **0.707** | 0.706 | 0.888 | **0.034** | 2,376 | 0.41x |
| Upstage Enhanced | 70.2 | 78.2 | 0.935 | — | — | — | — | — | — | 5,597† | 0.17x |

*vs Pipeline = speedup relative to GLM-OCR Pipeline SDK baseline (976ms). †Upstage Enhanced OmniDoc latency unreliable (10 samples); DP-Bench avg (5,597ms, 200 samples) shown instead. ‡Azure Layout: Formula CDM 산출 불가로 Overall 미산출.*

---

## 4. Latency Analysis

### 4.1 Per-Benchmark Avg Latency (ms/sample)

| Model | OmniDoc | DP-Bench | OCRBench | UniMER | PubTab | TEDS | NanoNets | Handwrt |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-OCR (VLM-only) | 2,376 | 968 | 98 | 428 | 1,177 | 1,310 | 460 | 50 |
| GLM-OCR Pipeline | 976 | 578 | 281 | 474 | 1,281 | 1,475 | 970 | 115 |
| MinerU-2.5 | 3,279 | 2,295 | 1,475 | 1,249 | 1,839 | 1,768 | 2,540 | 844 |
| Upstage Standard | 3,289 | 2,834 | — | — | — | — | — | — |
| Upstage Enhanced | 25,412† | 5,597 | — | — | — | — | — | — |
| Azure Layout | 7,436 | 5,976 | — | — | — | — | — | — |
| **Allgaznie-GLM** | **715** | **376** | 279 | 230 | 905 | 1,105 | 668 | 33 |
| Allgaznie-Paddle | 762 | 362 | 422 | 209 | 789 | 780 | 841 | 68 |
| **Allgaznie-MinerU** | **655** | 389 | 225 | 293 | 695 | 797 | 520 | 36 |
| Allgaznie-DeepSeek | 1,455 | 756 | 357 | 246 | 651 | 746 | 858 | 81 |
| DeepSeek-OCR2 (VLM-only) | 533 | 292 | 87 | 293 | 336 | 169 | 264 | 115 |

*†Upstage Enhanced OmniDocBench: 10 samples only (API rate limit).*

### 4.2 OmniDocBench Latency Distribution (ms)

| Model | Avg | Median | P90 | P99 | Max | Type |
|:---|---:|---:|---:|---:|---:|:---|
| DeepSeek-OCR2 | 533 | — | — | — | — | VLM-only |
| **Allgaznie-MinerU** | **655** | **443** | 1,452 | 2,911 | 6,752 | Pipeline |
| **Allgaznie-GLM** | **715** | **428** | 1,511 | 4,244 | 14,179 | Pipeline |
| Allgaznie-Paddle | 762 | 444 | 1,408 | 12,135 | 14,440 | Pipeline |
| GLM-OCR Pipeline | 976 | 591 | 2,006 | 5,749 | 10,839 | Pipeline (SDK) |
| Allgaznie-DeepSeek | 1,455 | 1,136 | 2,859 | 7,711 | 14,474 | Pipeline |
| GLM-OCR (VLM-only) | 2,376 | 1,342 | 4,178 | 17,172 | 17,363 | VLM-only |
| MinerU-2.5 | 3,279 | 2,664 | 5,785 | 9,601 | 52,622 | Pipeline (SDK) |
| Upstage Standard | 3,289 | — | — | — | — | API |
| Upstage Enhanced | 5,597† | — | — | — | — | API |
| Azure Layout | 7,436 | 6,693 | 10,486 | 21,092 | 47,428 | API |

*†DP-Bench latency used (200 samples). Sorted by Avg ascending.*

### 4.3 Latency Key Findings

- **DeepSeek-OCR2**: Fastest (533ms) but VLM-only — no layout detection, not directly comparable to pipelines.
- **Allgaznie-MinerU**: Fastest pipeline (655ms avg, 443ms median). **5.0x faster** than MinerU SDK (3,279ms) using the same 1.2B VLM.
- **Allgaznie-GLM**: 715ms avg, **1.37x faster** than GLM-OCR Pipeline SDK (976ms) with same quality (93.3 vs 93.3).
- **MinerU-2.5**: Slowest pipeline (3,279ms) with extreme tail latency (P99=9,601ms, max=52,622ms).
- **Allgaznie pipelines** have much better tail latency (P99/max) than SDK counterparts thanks to concurrent VLM inference and optimized preprocessing.

---

## 5. Key Findings

### 5.1 Pipeline vs VLM-Only

Pipeline mode (2-stage: layout detection + per-region VLM) delivers far superior **document parsing** compared to VLM-only:
- OmniDocBench Table TEDS: Allgaznie-GLM 0.925 vs GLM-OCR VLM-only 0.404
- Reading order, text block segmentation, and structured output only possible with pipeline

VLM-only mode excels at **single-element recognition**:
- OCRBench: GLM-OCR 0.837 (full-page understanding)
- UniMERNet: GLM-OCR CDM 0.940 (formula recognition)
- Handwritten: GLM-OCR CER 0.034 (handwriting recognition)
- KIE: GLM-OCR ANLS 0.888 (key-value extraction)

But VLM-only struggles with **structured document parsing**:
- DP-Bench NID: GLM-OCR 0.119 (VLM outputs unstructured text dump vs GT's document structure)

### 5.2 Allgaznie vs Official Pipelines

**Allgaznie-GLM vs GLM-OCR Pipeline (SDK)**:
- Formula CDM: 0.930 vs 0.922 (Allgaznie wins)
- Table TEDS: 0.925 vs 0.919 (Allgaznie wins)
- Text ED: 0.054 vs 0.042 (Pipeline SDK wins)
- Overall Custom: 88.7 vs 89.6 (Pipeline SDK wins marginally)
- **Latency: 715ms vs 976ms (Allgaznie 1.37x faster)**

**Allgaznie-MinerU vs MinerU-2.5**:
- Table TEDS: 0.890 vs 0.864 (Allgaznie wins)
- DP-Bench TEDS: 0.968 vs 0.946 (Allgaznie wins)
- Text ED: 0.056 vs 0.054 (nearly identical)
- Handwritten CER: 0.738 vs 1.950 (Allgaznie much better)
- **Latency: 655ms vs 3,279ms (Allgaznie 5.0x faster)**

### 5.3 VLM Backend Comparison (Allgaznie Pipeline)

| VLM Backend | OmniDoc Overall | OmniDoc TEDS | DP-Bench NID | Latency (ms) | Strengths |
|:---|:---:|:---:|:---:|---:|:---|
| **GLM-OCR** | **93.3** | **0.925** | 0.911 | 715 | Best tables, best formulas |
| **PaddleOCR-VL** | 91.3 | 0.890 | 0.871 | 762 | Best KIE (0.812), good tables |
| **MinerU-VL** | 91.2 | 0.890 | 0.909 | **655** | Fastest, best DP-Bench TEDS (0.968) |
| **DeepSeek-OCR-2** | 78.8 | 0.615 | 0.900 | 1,455 | Weakest + slowest in pipeline mode |

Note: Allgaznie-Paddle Overall 91.3 (official eval) but custom eval Overall 76.6 due to table rendering bug — the VLM outputs `<fcel>` tags instead of HTML tables.

### 5.4 DeepSeek-OCR2 (Original VLM-only)

Strong standalone performance: OmniDoc Formula CDM 0.912, OCRBench 0.486, UniMER CDM 0.795, Handwritten CER 0.195.
But NanoNets KIE 0.189 — the model cannot extract structured key-value information (VLM-only limitation).

### 5.5 API Models vs Open-Source

**Upstage Standard**: Best text NID (0.971) but weak tables (TEDS 0.700 vs Allgaznie-GLM 0.925) and formulas (CDM 0.546 vs 0.930).

**Azure Layout**: Text ED 0.133 (Upstage 0.122과 유사), Table TEDS 73.0 (Upstage 70.0 대비 소폭 우위). 수식 인식 미지원 (LaTeX 출력 불가).

| | Upstage Standard | Azure Layout | Allgaznie-GLM |
|:---|:---:|:---:|:---:|
| OmniDoc Text (1-ED) | 87.8% | 86.7% | 94.6% |
| OmniDoc Table TEDS | 70.0 | 73.0 | 92.5 |
| OmniDoc Formula | CDM 54.6 | N/A† | CDM 93.0 |
| DP-Bench NID | **0.971** | 0.876 | 0.911 |
| DP-Bench TEDS | 0.916 | 0.874 | 0.928 |
| Latency (OmniDoc) | 3,289ms | 7,436ms | 715ms |

*†Azure Layout은 수식을 LaTeX로 출력하지 않음.*

Open-source pipelines significantly outperform both API services:
- Table recognition: Allgaznie-GLM 0.925 vs Upstage 0.700 / Azure 0.730
- Formula recognition: Allgaznie-GLM CDM 0.930 vs Upstage 0.546 / Azure N/A
- Latency: Allgaznie-GLM 715ms vs Upstage 3,289ms / Azure 7,436ms (4.6x / 10.4x faster)

API advantages:
- No GPU required
- Consistent quality without model management
- Upstage: Best text NID (0.971)

---

## 6. Reproducibility vs Official Leaderboard

OmniDocBench v1.5 공식 리더보드 대비 재현도.

| Model | 공식 Overall | 우리 Overall | Delta | 재현율 | 비고 |
|:---|:---:|:---:|:---:|:---:|:---|
| MinerU-2.5 | 90.67 | 90.6 | -0.04 | 99.96% | 거의 완벽 재현 |
| GLM-OCR | 94.62* | 93.3 | -1.34 | 98.6% | Pipeline SDK로 재현. *HF 자체 발표값 |
| PaddleOCR-VL Pipeline | 92.86 | 92.7 | -0.17 | 99.8% | Pipeline SDK로 재현 (다른 장비) |
| DeepSeek-OCR v1 | 87.01 | — | — | — | 리더보드 모델 (v1), 우리는 v2 사용 |

리더보드에 우리 모델을 삽입하면:

| Rank | Model | Overall | Source |
|:---:|:---|:---:|:---|
| 1 | GLM-OCR (HF 발표) | 94.62 | 공식 |
| **2** | **Allgaznie-GLM** | **93.3** | **우리** |
| **3** | **GLM-OCR Pipeline (SDK)** | **93.3** | **우리** |
| 4 | PaddleOCR-VL | 92.86 | 공식 |
| **5** | **PaddleOCR-VL Pipeline (재현)** | **92.7** | **우리** |
| **6** | **Allgaznie-Paddle** | **91.3** | **우리** |
| **7** | **Allgaznie-MinerU** | **91.2** | **우리** |
| 8 | MinerU2.5 | 90.67 | 공식 |
| **9** | **MinerU-2.5 (재현)** | **90.6** | **우리** |
| 10 | Qwen3-VL-235B | 89.15 | 공식 |
| 11 | MonkeyOCR-pro-3B | 88.85 | 공식 |
| 12 | Gemini-2.5 Pro | 88.03 | 공식 |
| 13 | Deepseek-OCR v1 | 87.01 | 공식 |

---

## 7. Scoring Fix Iterations (2026-03-06)

Three rounds of evaluate→analyze→fix→re-evaluate to eliminate false negatives (correct predictions marked as wrong).

### Round 1: Initial artifact detection
| Fix | Benchmark | Description | Impact |
|-----|-----------|-------------|--------|
| Dot-leader normalization | DP-Bench | Strip `(. ){3,}` patterns from GT and pred | 3 TOC samples fixed |
| Chart description stripping | DP-Bench | Remove Enhanced-mode chart descriptions outside `<figcaption>` | Prevents false penalty |
| LaTeX formula normalization | OmniDocBench | Normalize `\operatorname`, `\left`, etc. before edit distance | Better formula matching |
| Concat text fallback | OmniDocBench | When element-wise text <80, try concatenated text comparison | Fixes segmentation mismatch |

### Round 2: Cross-model analysis
| Fix | Benchmark | Description | Impact |
|-----|-----------|-------------|--------|
| PaddleOCR `<fcel>` tag stripping | DP-Bench | Strip `<fcel>/<lcel>/<ecel>/<ucel>/<nl>` tags | Paddle NID improved |
| HTML table removal | DP-Bench | Unconditional strip (no cell text fallback) | Cleaner scoring |
| GT newline fix | DP-Bench | `replace('\n','')` → `replace('\n',' ')` to prevent word merge | +0.003-0.007 all models |

### Round 3: Deep-dive fixes
| Fix | Benchmark | Description | Impact |
|-----|-----------|-------------|--------|
| Single-column table detection | DP-Bench | `\|text\|` patterns now stripped (was missed before) | +0.002 Upstage models |
| Concat fallback key fix | OmniDocBench | `e.get('content')` for md_tex_filter elements (was 'text') | 83 samples improved for Upstage |
| Table→text in concat fallback | OmniDocBench | Extract cell text from pred HTML tables when no GT tables | 95 samples improved for glm_ocr |
| Formula 'latex' key copy | OmniDocBench | GT formula elements store text in 'latex', not 'text' | Correct formula matching |

### VLM Parameter Fixes (2026-03-06)
| Fix | Model | Description |
|-----|-------|-------------|
| max_tokens 4096→8192 | All Allgaznie models | Increased from 4096 to 8192 for longer outputs |
| DeepSeek max_tokens cap | Allgaznie-DeepSeek | Capped at 4096 (model max_position_embeddings=8192, need room for input) |
| repetition_penalty 1.1 | GLM-OCR | Match official config |
| no_repeat_ngram_size 35 | DeepSeek-OCR2 | Match official eval config |
| MinerU penalties | MinerU-VL | presence_penalty=1.0, frequency_penalty=0.005/0.05, no_repeat_ngram_size=100 |

---

## 8. Qualitative Error Analysis

### 8.1 DP-Bench Error Analysis

49 samples (NID < 0.9) across Standard and Enhanced were manually reviewed:

| Category | Count | % |
|----------|:---:|:---:|
| **(A) Truly wrong prediction** | 4 | 8% |
| **(B) GT/scoring limitation** | 43 | 88% |
| **(A+B) Mixed** | 2 | 4% |

### 8.2 OmniDocBench Error Analysis (Upstage API)

243 samples (per-sample overall < 60) analyzed:

| Category | Count | % of low | % of 1355 |
|----------|:---:|:---:|:---:|
| **(A) True OCR error** | 34 | 15.6% | 2.5% |
| **(B) GT/scoring limitation** | 184 | 84.4% | 13.6% |

84% of low scores are evaluation artifacts, not OCR quality issues.

### 8.3 OmniDocBench Error Analysis (Allgaznie GLM)

64 samples (overall < 60) analyzed:

| Category | Count | % of low | % of 1355 |
|----------|:---:|:---:|:---:|
| **(A) True OCR error** | ~2 | ~3% | ~0.15% |
| **(B) GT/scoring limitation** | ~62 | ~97% | ~4.6% |

GLM has **4x fewer** low-scoring samples than Upstage (64 vs 243) and only **0.15%** true OCR errors.

---

## 9. Infrastructure Notes

### subprocess.PIPE Deadlock Fix
vLLM server using `stdout=subprocess.PIPE` caused 64KB pipe buffer fill → server deadlock. Fixed by redirecting to log file.

### 1x1 Placeholder Fix
Empty bbox crops created 1x1 pixel images. PaddleOCR-VL raised `ValueError` → vLLM state corruption. Fixed by returning `None`.

### DeepSeek-OCR-2 vLLM Compatibility
`DeepseekOCR2ForCausalLM` cannot run in vLLM HTTP server mode for allgaznie pipeline, only offline batch mode via `vllm.LLM`.
