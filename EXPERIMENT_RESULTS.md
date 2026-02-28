# SDS-KoPub VDR Benchmark: Embedding Experiment Results

## Overview

Korean document visual retrieval benchmark (SDS-KoPub-VDR-Benchmark) 에서 다양한 임베딩 전략의 Recall@K 성능을 비교한 실험 결과입니다.

## Setup

| Item | Detail |
|------|--------|
| **Dataset** | SamsungSDS-Research/SDS-KoPub-VDR-Benchmark |
| **Corpus** | 40,781 pages (SDS-KoPub-corpus) |
| **Queries** | 600 queries (SDS-KoPub-QA, test split) |
| **OCR Model** | zai-org/GLM-OCR (FP8, vLLM) |
| **Embedding 2B** | Forturne/Qwen3-VL-Embedding-2B-FP8 (dim=2048) |
| **Embedding 8B** | Forturne/Qwen3-VL-Embedding-8B-FP8 (dim=4096) |
| **Caption Model** | GPT-5-mini (vision API, max_completion_tokens=2048) |
| **Hardware** | NVIDIA B200 GPU |

## Embedding Types

| Type | Description | Count | Source |
|------|-------------|-------|--------|
| **text** | OCR text (모든 region 포함, figure_title/vision_footnote 포함) | 40,781 pages | OCR results |
| **pure_text** | OCR text WITHOUT figure_title/vision_footnote regions | 40,781 pages | OCR results (filtered) |
| **region_mm** | Multimodal embeddings (crop image + caption text) | 21,052 regions | Image/chart crops + nearby captions |
| **caption_text** | GPT-5-mini generated descriptions for image/chart regions | 10,284 pages | GPT-5-mini vision API |

## Retrieval Configurations

| Config | Components | Description |
|--------|-----------|-------------|
| text-only | text | OCR 텍스트만 사용 |
| pure-text-only | pure_text | figure_title/vision_footnote 제외한 텍스트 |
| text+region | text + region_mm | 텍스트 + 멀티모달 region 임베딩 |
| pure+region | pure_text + region_mm | pure 텍스트 + 멀티모달 region 임베딩 |
| text+caption | text + caption_text | 텍스트 + GPT-5-mini 캡션 |
| pure+caption | pure_text + caption_text | pure 텍스트 + GPT-5-mini 캡션 |
| text+region+caption | text + region_mm + caption_text | 텍스트 + region + 캡션 (all combined) |
| pure+region+caption | pure_text + region_mm + caption_text | pure 텍스트 + region + 캡션 |

## Results: 2B Model (Qwen3-VL-Embedding-2B-FP8, dim=2048)

### R@K

| K | text-only | pure-text-only | text+region | pure+region | text+caption | pure+caption | text+region+caption | pure+region+caption |
|---|-----------|----------------|-------------|-------------|--------------|--------------|---------------------|---------------------|
| R@1 | 40.2% | 37.0% | 40.5% | 37.2% | 40.2% | 36.7% | **40.5%** | 37.2% |
| R@5 | 67.0% | 63.5% | 67.5% | 65.3% | 67.2% | 63.2% | **67.5%** | 64.8% |
| R@10 | 75.0% | 69.8% | 75.3% | 71.5% | 75.7% | 70.3% | **75.8%** | 71.8% |
| R@20 | 81.2% | 76.3% | 81.3% | 77.8% | **81.7%** | 76.3% | **81.7%** | 77.7% |

### Per-type R@10 (2B)

| Type | text-only | pure-text-only | text+region | pure+region | text+region+caption | pure+region+caption |
|------|-----------|----------------|-------------|-------------|---------------------|---------------------|
| cross | 76.8% | 72.6% | 76.8% | 73.8% | **77.1%** | 73.8% |
| text | 82.5% | 80.6% | 83.5% | 81.6% | **84.5%** | 82.5% |
| visual | 66.5% | 57.1% | 67.1% | 60.2% | **67.7%** | 60.9% |

## Results: 8B Model (Qwen3-VL-Embedding-8B-FP8, dim=4096)

### R@K

| K | text-only | pure-text-only | text+region | pure+region | text+caption | pure+caption | text+region+caption | pure+region+caption |
|---|-----------|----------------|-------------|-------------|--------------|--------------|---------------------|---------------------|
| R@1 | 44.3% | 40.0% | 45.0% | 40.8% | 44.5% | 40.2% | **45.2%** | 40.5% |
| R@5 | 74.2% | 68.7% | **74.8%** | 70.3% | 74.2% | 68.5% | 74.5% | 70.0% |
| R@10 | 79.7% | 74.8% | 80.2% | 76.5% | 80.2% | 75.2% | **80.5%** | 76.8% |
| R@20 | 83.5% | 79.7% | 84.0% | 81.3% | 84.2% | 80.3% | **84.5%** | 81.7% |

### Per-type R@10 (8B)

| Type | text-only | pure-text-only | text+region | pure+region | text+region+caption | pure+region+caption |
|------|-----------|----------------|-------------|-------------|---------------------|---------------------|
| cross | 82.4% | 78.9% | **82.7%** | 79.5% | **82.7%** | 79.5% |
| text | 84.5% | 84.5% | 85.4% | 85.4% | **86.4%** | **86.4%** |
| visual | 70.8% | 60.2% | 71.4% | 64.6% | **72.0%** | 65.2% |

## Key Findings

### 1. 2B vs 8B Model Size

| Metric | 2B (best) | 8B (best) | Improvement |
|--------|-----------|-----------|-------------|
| R@1 | 40.5% | **45.2%** | +4.7%p |
| R@5 | 67.5% | **74.8%** | +7.3%p |
| R@10 | 75.8% | **80.5%** | +4.7%p |
| R@20 | 81.7% | **84.5%** | +2.8%p |

**8B 모델이 2B 대비 전 구간에서 4-7%p 우위.** 특히 R@5에서 차이가 가장 큼.

### 2. text vs pure_text (figure_title/vision_footnote 제외 효과)

| Config | 2B R@10 | 8B R@10 |
|--------|---------|---------|
| text-only | 75.0% | 79.7% |
| pure-text-only | 69.8% | 74.8% |
| **Δ** | **-5.2%p** | **-4.9%p** |

**figure_title/vision_footnote를 포함하는 것이 ~5%p 더 좋음.** 이 영역들이 retrieval에 유용한 정보를 담고 있음.

### 3. Region Multimodal Embeddings 효과

| Config | 2B R@10 | 8B R@10 |
|--------|---------|---------|
| text-only → text+region | 75.0% → 75.3% (+0.3%p) | 79.7% → 80.2% (+0.5%p) |
| pure-text → pure+region | 69.8% → 71.5% (+1.7%p) | 74.8% → 76.5% (+1.7%p) |

**Region 임베딩은 pure_text에서 더 큰 효과** (+1.7%p). text에는 이미 figure_title이 포함되어 있어 추가 효과가 적음.

### 4. GPT-5-mini Caption 효과

| Config | 2B R@10 | 8B R@10 |
|--------|---------|---------|
| text-only → text+caption | 75.0% → 75.7% (+0.7%p) | 79.7% → 80.2% (+0.5%p) |
| text+region → text+region+caption | 75.3% → 75.8% (+0.5%p) | 80.2% → 80.5% (+0.3%p) |

**캡션은 소폭 개선 효과** (+0.3~0.7%p). Region 멀티모달과 겹치는 정보가 많아 추가 이득이 제한적.

### 5. Best Configuration per Model

| Model | Best Config | R@10 |
|-------|-------------|------|
| 2B | text+region+caption | **75.8%** |
| 8B | text+region+caption | **80.5%** |

**두 모델 모두 text+region+caption (all combined)이 최고 성능.**

## Cost Summary

| Item | Cost | Details |
|------|------|---------|
| GPT-5-mini Captioning | $9.72 | 21,052 regions, 13.6M prompt + 12.8M completion tokens |
| OCR (GLM-OCR) | ~$0 | Self-hosted vLLM |
| Embedding (2B + 8B) | ~$0 | Self-hosted vLLM |

## Output Files

### 2B Embeddings (`output/embeddings/`)

| File | Shape | Size | Description |
|------|-------|------|-------------|
| corpus_ocr_text.npy | (40781, 2048) | 637MB | Full OCR text embeddings |
| corpus_pure_text.npy | (40781, 2048) | 637MB | Pure text (no figure_title/vision_footnote) |
| corpus_regions.npy | (21052, 2048) | 329MB | Region multimodal (image + caption) |
| corpus_caption_text.npy | (10284, 2048) | 161MB | GPT-5-mini caption text |
| queries.npy | (600, 2048) | 9.4MB | Query embeddings |
| region_metadata.jsonl | 21052 entries | 7.6MB | Region metadata |
| pure_text_page_ids.json | 40781 entries | 6.9MB | Page ID mapping |
| caption_page_ids.json | 10284 entries | 1.7MB | Caption page ID mapping |

### 8B Embeddings (`output/embeddings_8b/`)

| File | Shape | Size | Description |
|------|-------|------|-------------|
| corpus_ocr_text.npy | (40781, 4096) | 1.27GB | Full OCR text embeddings |
| corpus_pure_text.npy | (40781, 4096) | 1.27GB | Pure text (no figure_title/vision_footnote) |
| corpus_regions.npy | (21052, 4096) | 658MB | Region multimodal (image + caption) |
| corpus_caption_text.npy | (10284, 4096) | 322MB | GPT-5-mini caption text |
| queries.npy | (600, 4096) | 18.8MB | Query embeddings |
| region_metadata.jsonl | 21052 entries | 7.6MB | Region metadata |
| pure_text_page_ids.json | 40781 entries | 6.9MB | Page ID mapping |
| caption_page_ids.json | 10284 entries | 1.7MB | Caption page ID mapping |

### Other Files

| File | Description |
|------|-------------|
| output/region_captions.jsonl | GPT-5-mini captions (21,052 records) |
| output/ocr_results.jsonl | GLM-OCR results (40,781 pages) |
| output/parsed_texts.jsonl | Parsed text for embedding |
