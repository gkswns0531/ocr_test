# SDS-KoPub OCR + Embedding Pipeline

SamsungSDS KoPub VDR Benchmark (40,781 Korean document pages)에 대한 OCR 파싱 + VL 임베딩 파이프라인.

**GLM-OCR** (layout detection + VLM recognition) → **Qwen3-VL-Embedding** (image/text/query embeddings) → **HuggingFace 업로드**까지 단일 스크립트로 실행.

## Quick Start

```bash
# 1. 환경 설치 (B200 / 신규 머신)
bash setup_b200.sh

# 2. 전체 파이프라인 실행 (OCR → 임베딩 → HF 업로드)
python3 run_b200_pipeline.py \
  --hf-repo YOUR_USERNAME/sds-kopub-ocr-embeddings \
  --hf-token hf_xxxxx
```

## 파이프라인 단계

| Step | 설명 | 출력 |
|------|------|------|
| 1. Environment Check | CUDA, vLLM, GLM-OCR SDK 확인 | - |
| 2. Download Dataset | SDS-KoPub 40,781 페이지 다운로드 | `data/` |
| 3. Start vLLM Server | GLM-OCR VLM 서버 시작 (FP8) | - |
| 4. OCR Parsing | 레이아웃 감지 + VLM 인식 | `output/ocr_results.jsonl`, `output/crops/` |
| 5. Stop vLLM Server | OCR 서버 종료 | - |
| 6. Embeddings | 이미지/텍스트/쿼리 임베딩 계산 | `output/embeddings/*.npy` |
| 7. HF Upload | HuggingFace에 결과 업로드 | - |

## 개별 단계 실행

```bash
# OCR만 실행
python3 run_b200_pipeline.py --step ocr

# OCR 중단 후 재개
python3 run_b200_pipeline.py --step ocr --resume

# 임베딩만 실행 (OCR 완료 후)
python3 run_b200_pipeline.py --step embed

# HF 업로드만
python3 run_b200_pipeline.py --step upload --hf-repo USER/REPO --hf-token hf_xxx
```

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--step` | 전체 | `ocr`, `embed`, `upload` 중 하나 |
| `--resume` | off | OCR 중단 후 이어서 처리 |
| `--ocr-dtype` | `fp8` | VLM 추론 dtype (`fp8`/`bf16`/`auto`) |
| `--gpu-mem-util` | `0.80` | vLLM GPU 메모리 사용률 |
| `--batch-size` | `8` | Layout detection 배치 크기 |
| `--workers` | `128` | VLM 병렬 요청 수 |
| `--embedding-model` | `Qwen/Qwen3-VL-Embedding-2B-FP8` | 임베딩 모델 |
| `--no-crops` | off | 이미지/차트 크롭 저장 건너뛰기 |
| `--hf-repo` | - | HuggingFace 업로드 대상 repo |
| `--hf-token` | - | HuggingFace API 토큰 |
| `--output-dir` | `output` | 결과 저장 디렉토리 |

## GPU별 권장 설정

### B200 (192GB VRAM)

```bash
python3 run_b200_pipeline.py \
  --gpu-mem-util 0.80 \
  --batch-size 8 \
  --workers 128 \
  --ocr-dtype fp8
```

### L4 (23GB VRAM)

```bash
python3 run_b200_pipeline.py \
  --step ocr --resume \
  --gpu-mem-util 0.60 \
  --batch-size 8 \
  --workers 64 \
  --ocr-dtype fp8
```

> L4에서는 encoder cache가 ~4,800 토큰으로 제한되어 일부 대형 region(5,000+ 토큰)에서 400 에러 발생 가능. `--gpu-mem-util 0.85`로 올리면 완화됨.

## 출력 구조

```
output/
├── ocr_results.jsonl          # 40,781건 OCR 결과 (regions, markdown, bbox, labels)
├── parsed_texts.jsonl         # 임베딩 입력 텍스트
├── crops/                     # 이미지/차트 영역 크롭 JPEG
└── embeddings/
    ├── corpus_images.npy      # (40781, 2048) 페이지 이미지 임베딩
    ├── corpus_ocr_text.npy    # (40781, 2048) OCR 텍스트 임베딩
    └── queries.npy            # (600, 2048) 쿼리 임베딩
```

### OCR 결과 포맷 (`ocr_results.jsonl`)

```json
{
  "page_id": "doc_123_page_0",
  "page_idx": 0,
  "regions": [
    {"index": 0, "label": "doc_title", "bbox_2d": [x1, y1, x2, y2], "content": "문서 제목"},
    {"index": 1, "label": "table", "bbox_2d": [...], "content": "<table>...</table>"},
    {"index": 2, "label": "image", "bbox_2d": [...], "content": null}
  ],
  "markdown": "# 문서 제목\n\n| col1 | col2 |\n...",
  "image_crops": [{"path": "crops/doc_123_page_0_crop_2.jpg", "bbox": [...], "label": "image"}]
}
```

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    run_b200_pipeline.py                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐ │
│  │ Dataset   │───▶│ GLM-OCR SDK   │───▶│ OCR Results  │ │
│  │ (Arrow)   │    │               │    │ (JSONL)      │ │
│  └──────────┘    │ PP-DocLayout  │    └──────────────┘ │
│                  │ (layout det.) │                      │
│                  │       ↓       │                      │
│                  │ vLLM GLM-OCR  │                      │
│                  │ (FP8 recog.)  │                      │
│                  └───────────────┘                      │
│                                                         │
│  ┌──────────────┐    ┌────────────────┐                │
│  │ OCR Results   │───▶│ Qwen3-VL-Emb   │──▶ .npy files │
│  │ + Images      │    │ (vLLM pooling) │                │
│  └──────────────┘    └────────────────┘                │
│                                                         │
│  ┌──────────────┐                                      │
│  │ HuggingFace  │◀── upload_folder()                   │
│  │ Hub          │                                      │
│  └──────────────┘                                      │
└─────────────────────────────────────────────────────────┘
```

### 내부 파이프라인 (GLM-OCR SDK)

```
Page Image
    │
    ▼
PP-DocLayoutV3 (33M params, FP32)     ← Layout Detection (5.6s/batch of 8)
    │ regions: text, table, formula, image, chart, ...
    ▼
Per-region crop + resize + base64     ← ~20ms/region
    │
    ▼
vLLM GLM-OCR (0.9B, FP8)             ← VLM Recognition (5.2s/region, 64 parallel)
    │ text content, HTML tables, LaTeX formulas
    ▼
Result Formatter → JSONL
```

## 레이턴시 프로파일 (L4, FP8, 32 pages)

| 단계 | 총 시간 | 평균/호출 | Wall % |
|------|---------|-----------|--------|
| VLM HTTP (vLLM) | 1325s (concurrent) | 5.2s/region | **병목** |
| Layout detection | 22.3s | 5.6s/batch | 28.4% |
| Image encode | 2.8s | 11ms/region | 3.6% |
| Region crop | 2.3s | 8.5ms/region | 2.9% |
| Image save | 0.7s | 22ms/page | 0.9% |
| Arrow decode | 0.3s | 9ms/page | 0.4% |

- **Throughput**: 0.41 pages/s (L4), ~2-4 pages/s (B200 예상)
- **Regions/page**: 평균 8.3개
- **VLM concurrency**: 17.3x (64 workers)

프로파일러 실행: `python3 profile_pipeline.py --pages 32`

## 모델 정보

| 모델 | 용도 | 크기 | dtype |
|------|------|------|-------|
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | VLM 텍스트 인식 | 0.9B | FP8 (dynamic quant) |
| [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) | 레이아웃 감지 | 33M | FP32 |
| [Qwen3-VL-Embedding-2B-FP8](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B-FP8) | 이미지/텍스트 임베딩 | 2B | FP8 |

## vLLM 설정 참고

- `--max-model-len 131072`: 모델 최대 시퀀스 길이 (입력+출력). B200에서는 전체 활용 가능.
- `--max_tokens 16384`: SDK의 VLM 생성 최대 토큰 수. 긴 테이블/수식 잘림 방지.
- **Encoder cache**: GPU 메모리에 따라 자동 결정. L4(0.6)=4,800 토큰, B200(0.8)=훨씬 큼. 대형 이미지 region이 이 한도를 초과하면 해당 region만 빈 결과 반환 (non-fatal).

## 파일 구조

```
ocr_test/
├── run_b200_pipeline.py    # 메인 파이프라인 (자체완결 단일 스크립트)
├── setup_b200.sh           # 환경 설치 스크립트
├── profile_pipeline.py     # 레이턴시 프로파일러
├── eval_bench.py           # OCR 벤치마크 평가 (8개 벤치마크)
├── benchmarks.py           # 벤치마크별 평가기
├── metrics.py              # 메트릭 (TEDS, CDM, ANLS, CER 등)
├── config.py               # 모델/벤치마크 설정
└── ...                     # 기타 평가/데모 스크립트
```
