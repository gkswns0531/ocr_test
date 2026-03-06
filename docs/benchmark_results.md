# OCR Benchmark Results Report

**Date**: 2026-03-06 (re-evaluated with corrected VLM parameters + eval logic fixes)
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
| `allgaznie-mineru` | Allgaznie + MinerU-VL | 1.2B + layout | Pipeline | BF16 |
| `allgaznie-deepseek` | Allgaznie + DeepSeek-OCR-2 | 3.4B + layout | Pipeline | FP8 |
| `upstage-standard` | Upstage Document Parse (Standard) | — | API | — |
| `upstage-enhanced` | Upstage Document Parse (Enhanced) | — | API | — |

**VLM-only**: Full-page image → VLM → text (single pass)
**Pipeline**: Image → Layout Detection → Region Crop → Per-region VLM → Markdown Assembly (2-stage)
**API**: Cloud API — full-page image → API → markdown (Upstage Document Parse v260128)

---

## 2. Master Results Table

All metrics below. **↑ = higher is better, ↓ = lower is better.** Bold = best in column.

### 2.1 OmniDocBench Official Eval (1,355 samples)

| Model | Text ED ↓ | Formula CDM ↑ | Table TEDS ↑ | Table TEDS-S ↑ | ReadOrder ED ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **GLM-OCR Pipeline** | **0.042** | 0.922 | 0.919 | 0.946 | 0.056 |
| **MinerU-2.5** | 0.054 | 0.910 | 0.864 | 0.909 | **0.052** |
| **Allgaznie-GLM** | 0.054 | **0.930** | **0.925** | **0.952** | 0.070 |
| **Allgaznie-Paddle** | 0.051 | 0.900 | 0.890 | 0.929 | 0.068 |
| **Allgaznie-MinerU** | 0.056 | 0.901 | 0.890 | 0.931 | 0.072 |
| **Allgaznie-DeepSeek** | 0.084 | 0.834 | 0.615 | 0.674 | 0.083 |
| **DeepSeek-OCR2** | 0.073 | 0.912 | 0.682 | 0.720 | 0.073 |
| Upstage Standard | 0.122 | 0.546 | 0.700 | 0.775 | 0.143 |
| Upstage Enhanced | 0.126 | 0.542 | 0.690 | 0.756 | 0.150 |
| GLM-OCR (VLM-only) | 0.128 | 0.815 | 0.404 | 0.424 | 0.143 |

### 2.2 OmniDocBench Custom Eval (with scoring fixes, per-type averages)

| Model | Overall | Text | Table (351) | Formula (200) |
|:---|:---:|:---:|:---:|:---:|
| **GLM-OCR Pipeline** | **89.6** | **88.8** | 93.0 | 84.1 |
| **Allgaznie-GLM** | 88.7 | 87.7 | **93.8** | **84.3** |
| **Allgaznie-MinerU** | 87.8 | 87.4 | 89.9 | 82.0 |
| **MinerU-2.5** | 87.9 | 87.5 | 88.4 | 80.3 |
| **DeepSeek-OCR2** | 85.0 | 85.5 | 76.4 | 80.5 |
| Upstage Standard | 83.6 | 88.7 | 74.1 | 46.5 |
| Upstage Enhanced | 83.2 | 88.4 | 74.2 | 46.1 |
| Allgaznie-DeepSeek | 81.5 | 84.1 | 64.9 | 74.4 |
| GLM-OCR (VLM-only) | 78.8 | 82.7 | 49.1 | 71.5 |
| Allgaznie-Paddle | 76.6 | 87.4 | 2.2 | 81.0 |

*Table/Formula scores averaged only over samples containing those element types.*

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
| Allgaznie-Paddle | 0.871 | 0.838 | 0.853 |

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
| MinerU-2.5 | 0.593 | 0.782 |

### 2.7 TEDS Test Table Recognition (200 samples)

| Model | TEDS ↑ | TEDS Structure ↑ |
|:---|:---:|:---:|
| **Allgaznie-MinerU** | **0.721** | **0.909** |
| GLM-OCR (VLM-only) | 0.706 | 0.913 |
| Allgaznie-Paddle | 0.702 | 0.898 |
| GLM-OCR Pipeline | 0.683 | 0.899 |
| Allgaznie-GLM | 0.678 | 0.883 |
| DeepSeek-OCR2 | 0.673 | 0.861 |
| Allgaznie-DeepSeek | 0.668 | 0.850 |
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
| MinerU-2.5 | 1.950 | 1.000 |

---

## 3. Cross-Model Summary

### 3.1 Consolidated View (Model = Row, Benchmark = Column)

| Model | OmniDoc Text ED ↓ | OmniDoc CDM ↑ | OmniDoc TEDS ↑ | DP-Bench NID ↑ | OCRBench ↑ | UniMER CDM ↑ | PubTab TEDS ↑ | TEDS Test ↑ | NanoNets ↑ | Handwr CER ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GLM-OCR Pipeline** | **0.042** | 0.922 | 0.919 | 0.929 | 0.477 | 0.843 | 0.691 | 0.683 | 0.782 | 0.121 |
| **MinerU-2.5** | 0.054 | 0.910 | 0.864 | 0.924 | 0.331 | 0.793 | 0.593 | 0.595 | **0.803** | 1.950 |
| **Allgaznie-GLM** | 0.054 | **0.930** | **0.925** | 0.911 | 0.463 | 0.578 | 0.642 | 0.678 | 0.785 | 0.689 |
| **Allgaznie-Paddle** | 0.051 | 0.900 | 0.890 | 0.871 | 0.465 | 0.625 | 0.652 | 0.702 | **0.812** | 0.687 |
| **Allgaznie-MinerU** | 0.056 | 0.901 | 0.890 | 0.909 | 0.402 | 0.601 | 0.657 | **0.721** | 0.784 | 0.738 |
| **Allgaznie-DeepSeek** | 0.084 | 0.834 | 0.615 | 0.900 | 0.349 | 0.512 | 0.645 | 0.668 | 0.780 | 0.767 |
| **DeepSeek-OCR2** | 0.073 | 0.912 | 0.682 | 0.917 | 0.486 | 0.795 | 0.684 | 0.673 | 0.189 | 0.195 |
| GLM-OCR (VLM-only) | 0.128 | 0.815 | 0.404 | — | **0.837** | **0.940** | **0.707** | 0.706 | — | **0.034** |
| Upstage Standard | 0.122 | 0.546 | 0.700 | **0.971** | — | — | — | — | — | — |
| Upstage Enhanced | 0.126 | 0.542 | 0.690 | 0.935 | — | — | — | — | — | — |

---

## 4. Key Findings

### 4.1 Pipeline vs VLM-Only

Pipeline mode (2-stage: layout detection + per-region VLM) delivers far superior **document parsing** compared to VLM-only:
- OmniDocBench Table TEDS: Allgaznie-GLM 0.925 vs GLM-OCR VLM-only 0.404
- Reading order, text block segmentation, and structured output only possible with pipeline

VLM-only mode excels at **single-element recognition**:
- OCRBench: GLM-OCR 0.837 (full-page understanding)
- UniMERNet: GLM-OCR CDM 0.940 (formula recognition)
- Handwritten: GLM-OCR CER 0.034 (handwriting recognition)

### 4.2 Allgaznie vs Official Pipelines

**Allgaznie-GLM vs GLM-OCR Pipeline (SDK)**:
- Formula CDM: 0.930 vs 0.922 (Allgaznie wins)
- Table TEDS: 0.925 vs 0.919 (Allgaznie wins)
- Text ED: 0.054 vs 0.042 (Pipeline SDK wins)
- Overall Custom: 88.7 vs 89.6 (Pipeline SDK wins marginally)

**Allgaznie-MinerU vs MinerU-2.5**:
- Table TEDS: 0.890 vs 0.864 (Allgaznie wins)
- DP-Bench TEDS: 0.968 vs 0.946 (Allgaznie wins)
- Text ED: 0.056 vs 0.054 (nearly identical)
- Handwritten CER: 0.738 vs 1.950 (Allgaznie much better)

### 4.3 VLM Backend Comparison (Allgaznie Pipeline)

| VLM Backend | OmniDoc Overall | OmniDoc TEDS | DP-Bench NID | Strengths |
|:---|:---:|:---:|:---:|:---|
| **GLM-OCR** | **88.7** | **0.925** | 0.911 | Best tables, best formulas |
| **PaddleOCR-VL** | 76.6 | 0.890 | 0.871 | Best KIE (0.812), good tables |
| **MinerU-VL** | 87.8 | 0.890 | 0.909 | Best DP-Bench TEDS (0.968) |
| **DeepSeek-OCR-2** | 81.5 | 0.615 | 0.900 | Weakest in pipeline mode |

Note: Allgaznie-Paddle Overall 76.6 is low due to table rendering bug (TEDS 2.2 in custom eval) — the VLM outputs `<fcel>` tags instead of HTML tables.

### 4.4 DeepSeek-OCR2 (Original VLM-only)

Strong standalone performance: OmniDoc Formula CDM 0.912, OCRBench 0.486, UniMER CDM 0.795, Handwritten CER 0.195.
But NanoNets KIE 0.189 — the model cannot extract structured key-value information (VLM-only limitation).

### 4.5 Upstage API vs Open-Source

**Upstage Standard**: Best text NID (0.971) but weak tables (TEDS 0.700 vs Allgaznie-GLM 0.925) and formulas (CDM 0.546 vs 0.930).

Open-source pipelines significantly outperform Upstage API on:
- Table recognition: 0.925 vs 0.700 (32% gap)
- Formula recognition: 0.930 vs 0.546 (70% gap)
- Overall OmniDocBench: 88.7 vs 83.6

Upstage API advantages:
- Text-only NID: 0.971 (best)
- No GPU required
- Consistent quality without model management

---

## 5. Scoring Fix Iterations (2026-03-06)

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

## 6. Qualitative Error Analysis

### 6.1 DP-Bench Error Analysis

49 samples (NID < 0.9) across Standard and Enhanced were manually reviewed:

| Category | Count | % |
|----------|:---:|:---:|
| **(A) Truly wrong prediction** | 4 | 8% |
| **(B) GT/scoring limitation** | 43 | 88% |
| **(A+B) Mixed** | 2 | 4% |

### 6.2 OmniDocBench Error Analysis (Upstage API)

243 samples (per-sample overall < 60) analyzed:

| Category | Count | % of low | % of 1355 |
|----------|:---:|:---:|:---:|
| **(A) True OCR error** | 34 | 15.6% | 2.5% |
| **(B) GT/scoring limitation** | 184 | 84.4% | 13.6% |

84% of low scores are evaluation artifacts, not OCR quality issues.

### 6.3 OmniDocBench Error Analysis (Allgaznie GLM)

64 samples (overall < 60) analyzed:

| Category | Count | % of low | % of 1355 |
|----------|:---:|:---:|:---:|
| **(A) True OCR error** | ~2 | ~3% | ~0.15% |
| **(B) GT/scoring limitation** | ~62 | ~97% | ~4.6% |

GLM has **4x fewer** low-scoring samples than Upstage (64 vs 243) and only **0.15%** true OCR errors.

---

## 7. Infrastructure Notes

### subprocess.PIPE Deadlock Fix
vLLM server using `stdout=subprocess.PIPE` caused 64KB pipe buffer fill → server deadlock. Fixed by redirecting to log file.

### 1x1 Placeholder Fix
Empty bbox crops created 1x1 pixel images. PaddleOCR-VL raised `ValueError` → vLLM state corruption. Fixed by returning `None`.

### DeepSeek-OCR-2 vLLM Compatibility
`DeepseekOCR2ForCausalLM` cannot run in vLLM HTTP server mode for allgaznie pipeline, only offline batch mode via `vllm.LLM`.
