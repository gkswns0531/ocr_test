# Epic B: 프로덕션 OCR 파이프라인

## 1. 배경 및 목적

SDS-KoPub 40,781 페이지를 효율적으로 OCR 처리할 프로덕션 파이프라인이 필요. GPU 192GB HBM3e 환경에서 GLM-OCR 기반 파이프라인을 설계·구현하고, 전체 코퍼스를 OCR 처리하여 HuggingFace에 업로드.

## 2. 파이프라인 아키텍처 (B-1)

```
Arrow Dataset (HuggingFace)
    ↓
[Stage 0] Image Extract — Arrow raw bytes → /dev/shm (PIL 디코드 생략)
    ↓
[Stage 1] Data Loading — PIL lazy open → image_paths_dict
    ↓
[Stage 2] Layout Detection — PP-DocLayoutV3 (33M) GPU 배치 추론
    ├─ GPU JPEG/PNG decode + resize (800×800)
    ├─ RT-DETR 추론 (25개 카테고리)
    └─ NMS + bbox merge + 영역 분류
    ↓
[Stage 3] Region Cropping — 원본 해상도에서 영역 크롭
    ├─ text/table/formula → VLM 인식 큐
    ├─ image/chart → skip (메타데이터만 저장)
    └─ header/footer/page_no → abandon
    ↓
[Stage 4] VLM Recognition — GLM-OCR 0.9B (FP8, vLLM)
    ├─ 256 동시 HTTP 요청
    ├─ Chunked prefill 활성화
    └─ text/table HTML/formula LaTeX 출력
    ↓
[Stage 5] Text Extraction + Embedding + HF Upload
```

### 2.1 핵심 설계 결정

**이미지 추출 (Stage 0)**: HuggingFace Arrow 데이터셋의 raw bytes를 직접 추출하여 `/dev/shm`에 쓰기. 기존 PIL 디코딩→JPEG 재인코딩의 불필요한 왕복 제거. **184x 성능 향상** (7.9s → 0.04s / 64장).

**vLLM 서버 구성**:
```
GLM-OCR 0.9B (FP8 dynamic quantization)
├── max-model-len: 131,072
├── max-num-batched-tokens: 65,536 (A/B 테스트로 최적값 발견)
├── max-num-seqs: 1,024
├── gpu-memory-utilization: 0.90
├── enable-chunked-prefill: ON
├── enable-prefix-caching: ON (91% hit rate)
└── FP8 모델 크기: ~1.36 GB (192GB VRAM의 0.7%)
```

**동시성 설계**: SDK max_workers 256, connection_pool_size 288, Layout batch_size 64. 실질적 throughput ~24.6 req/s.

### 2.2 VRAM 할당 구조

```
총 VRAM:            192 GB
× 0.90:             172.8 GB (vLLM 선점)
- 모델 가중치:      -  1.36 GB (FP8)
- Activation:       - 29.3 GB
= KV Cache:          142.15 GB = 2,329,040 토큰 슬롯
= 실제 동시성:       ~776개 (OCR ~3K tokens/req 기준)
```

### 2.3 GPU 스케일링 가이드

| GPU VRAM | `max-num-seqs` | `workers` | `batch-size` | 동시성 |
|----------|:-:|:-:|:-:|:-:|
| 24 GB (4090) | 21 | 84 | 6 | 22× |
| 48 GB (A6000) | 57 | 228 | 17 | 57× |
| 80 GB (A100) | 104 | 256 | 31 | 104× |
| 192 GB | 269 | 256 | 64 | 269× |

## 3. SDS-KoPub 전체 OCR 실행 (B-2)

### 3.1 데이터셋

| 항목 | 값 |
|------|-----|
| 데이터셋 | SamsungSDS-Research/SDS-KoPub-corpus |
| 총 페이지 | 40,781 |
| 이미지 크기 | 2480×3508 (A4 스캔) |
| 포맷 | Arrow (HuggingFace datasets), PNG |

### 3.2 실행 환경

| 항목 | 값 |
|------|-----|
| OCR 모델 | GLM-OCR 0.9B (FP8) |
| Layout 모델 | PP-DocLayoutV3 (33M, RT-DETR) |
| 서빙 | vLLM (max-num-batched-tokens=65536) |
| 처리 속도 | ~0.9 pages/s |
| 총 소요 시간 | ~12.6시간 |

### 3.3 실행 결과

| 산출물 | 크기 | 설명 |
|--------|------|------|
| `ocr_results.jsonl` | 168 MB | 40,781 페이지의 전체 OCR 결과 (레이아웃 + 텍스트) |
| `parsed_texts.jsonl` | 61 MB | 추출된 순수 텍스트 (임베딩 입력용) |
| `crops_regenerated/` | 21,052 files | image/chart 영역 crop |

### 3.4 OCR 결과 구조

각 페이지의 OCR 결과에는:
- `page_id`: 페이지 식별자
- `regions`: 감지된 영역 목록 (label, bbox, task_type)
- `text_results`: 영역별 인식 텍스트
- `markdown`: 조립된 마크다운 출력
- `crop_paths`: region crop 이미지 경로

### 3.5 Resume 지원

파이프라인은 Arrow shard 단위로 진행 상황을 추적하여, 중단 시 이미 처리된 shard는 건너뛰고 이어서 실행.

## 4. 성과

- 40,781 페이지 전량 처리 가능한 프로덕션 파이프라인 완성
- 최적화 후 0.9 pages/s (73초/64페이지)
- 단일 스크립트로 OCR → 텍스트 → 임베딩 → HF 업로드까지 일괄 처리
- HuggingFace Hub(`Forturne/SDS-KoPub-OCR`)에 전체 결과 업로드 완료

## 5. 산출물

- `run_b200_pipeline.py` (~1,900 LOC): 전체 파이프라인
- `setup_b200.sh`: 환경 설정 스크립트
- `DOCUMENTATION.md`: 파이프라인 설정/사용법 (vLLM 파라미터 가이드 포함)
- `output_dl/ocr_results.jsonl`, `output_dl/parsed_texts.jsonl`, `output_dl/crops_regenerated/`
