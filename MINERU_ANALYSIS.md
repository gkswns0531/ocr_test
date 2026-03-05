# MinerU 아키텍처 정밀 분석 및 Allgaznie 통합 검토

> 분석일: 2026-03-04 | 대상: MinerU v2.7.6 (`mineru-vl-utils` v0.1.22)

---

## 목차

1. [분석 배경](#1-분석-배경)
2. [MinerU 아키텍처 개요](#2-mineru-아키텍처-개요)
3. [Two-Step Extraction 정밀 분석](#3-two-step-extraction-정밀-분석)
4. [Hybrid 엔진 상세](#4-hybrid-엔진-상세)
5. [Allgaznie vs MinerU 아키텍처 비교](#5-allgaznie-vs-mineru-아키텍처-비교)
6. [통합 가능성 평가](#6-통합-가능성-평가)
7. [결론 및 권장안](#7-결론-및-권장안)

---

## 1. 분석 배경

### 현재 상황

Allgaznie OCR SDK는 GLM-OCR Pipeline과 PaddleOCR-VL Pipeline의 공통 아키텍처를 통합한 2-stage 파이프라인:

```
이미지 → PP-DocLayoutV3 (Layout Detection) → Region Crop → Per-Region VLM 추론 → Markdown 조립
```

MinerU-2.7.6은 벤치마크에서 DP-Bench NID 1위(91.4%), TEDS 2위(95.0%), OmniDocBench 공식 수치와 거의 동일(90.6 vs 90.7)을 달성한 하이브리드 파이프라인.

### 분석 목적

MinerU를 Allgaznie에 통합 가능한지 검토:
- MinerU의 아키텍처가 Allgaznie의 `LayoutDetector → VLMClient` 구조로 분해 가능한가?
- 공통 추상화가 가능한가?
- 통합 시 실질적 이점이 있는가?

---

## 2. MinerU 아키텍처 개요

### 2.1 세 가지 백엔드

| 백엔드 | Layout 모델 | Recognition 모델 | 전문 모델 보조 | 성능 |
|:---|:---|:---|:---|:---|
| `pipeline` | DocLayout-YOLO | PaddleOCR + SlanetPlus | MFD + UniMERNet | 범용 |
| `vlm-auto-engine` | MinerU2.5 VLM | MinerU2.5 VLM | 없음 | 고정밀 |
| **`hybrid-auto-engine`** | **MinerU2.5 VLM** | **MinerU2.5 VLM** | **PaddleOCR + MFD + UniMERNet** | **최고 성능** |

벤치마크에서 사용한 것은 `hybrid-auto-engine`.

### 2.2 모델 구성

| 모델 | 역할 | 파라미터 | 사용 백엔드 |
|:---|:---|:---|:---|
| **MinerU2.5-2509-1.2B** | VLM: Layout + Recognition | 1.2B (Qwen2VL 기반) | vlm, hybrid |
| DocLayout-YOLO | Layout Detection | ~30M | pipeline only |
| MFD YOLOv8 | 인라인 수식 영역 검출 | ~30M | pipeline, hybrid |
| UniMERNet / FormulaNet | 수식 인식 (LaTeX) | ~200M | pipeline, hybrid |
| PaddleOCR-torch | 텍스트 검출+인식 | ~15M | pipeline, hybrid |
| SlanetPlus / UnetTable | 테이블 구조 인식 | ONNX | pipeline only |
| PP-LCNet TableCls | 테이블 분류 | ONNX | pipeline |
| PP-LCNet OriCls | 문서 방향 분류 | ONNX | pipeline |
| LayoutReader | 읽기 순서 결정 | — | pipeline |

### 2.3 핵심 의존성

```
mineru[core] → mineru-vl-utils (VLM 추론 클라이언트)
             → pypdfium2 (이미지→PDF 변환)
             → paddleocr-torch (텍스트 OCR)
             → ultralytics (YOLO 모델)
             → onnxruntime (테이블 ONNX 모델)
```

`mineru_vl_utils` 패키지(v0.1.22)가 VLM 추론의 핵심 구현을 담당:
- `MinerUClient`: Two-step extraction 오케스트레이션
- `MinerUClientHelper`: 전처리(리사이즈, 크롭), 후처리(bbox 파싱, post-process)
- `VlmClient` 서브클래스들: transformers, vllm-engine, http-client 등 6개 백엔드

---

## 3. Two-Step Extraction 정밀 분석

### 3.1 전체 흐름

`batch_two_step_extract()`는 이름 그대로 **2단계 추론**:

```
Step 1: batch_layout_detect(images)
  ├─ 이미지 → 1036×1036 리사이즈 (prepare_for_layout)
  ├─ VLM 추론: "\nLayout Detection:" 프롬프트
  ├─ 출력 파싱: <|box_start|>x1 y1 x2 y2<|box_end|><|ref_start|>type<|ref_end|><|rotate_*|>
  └─ → ContentBlock 리스트 (type, bbox[0~1 정규화], angle)

Step 2: batch_content_extract(images, blocks)
  ├─ 원본 이미지에서 bbox로 크롭 (prepare_for_extract)
  │   └─ skip 목록: image, list, equation_block
  │   └─ 회전 보정: 90°/180°/270° rotate
  │   └─ 리사이즈: min_edge=28, max_edge_ratio=50
  ├─ 타입별 프롬프트 선택:
  │   ├─ table → "\nTable Recognition:"
  │   ├─ equation → "\nFormula Recognition:"
  │   └─ text/title/기타 → "\nText Recognition:"
  ├─ VLM 배치 추론 (concurrent 또는 stepping 모드)
  └─ post_process: 수식 정리, 테이블 OTSL→HTML 변환 등
```

**소스 증거** (`mineru_vl_utils/mineru_client.py:646-657`):
```python
def two_step_extract(self, image, priority=None, not_extract_list=None):
    blocks = self.layout_detect(image, priority)                     # Step 1
    block_images, prompts, params, indices = \
        self.helper.prepare_for_extract(image, blocks, not_extract_list)  # Crop
    outputs = self.client.batch_predict(block_images, prompts, params, priority)  # Step 2
    for idx, output in zip(indices, outputs):
        blocks[idx].content = output
    return self.helper.post_process(blocks)
```

### 3.2 Layout Detection 출력 포맷

VLM이 생성하는 텍스트를 regex로 파싱:

```
정규식: ^<|box_start|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<|box_end|><|ref_start|>(\w+?)<|ref_end|>(.*)$

예시 출력:
<|box_start|>45 120 955 180<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>
<|box_start|>45 200 480 850<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>500 200 955 600<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>
```

좌표는 0~1000 범위, 0~1.0으로 정규화하여 사용.

### 3.3 MinerU VLM 블록 타입 (21개)

| 타입 | 설명 | Extract 여부 |
|:---|:---|:---|
| `text` | 텍스트 | O (Text Recognition) |
| `title` | 제목 | O (Text Recognition) |
| `table` | 테이블 | O (Table Recognition) |
| `equation` | 수식 | O (Formula Recognition) |
| `code` | 코드 | O (Text Recognition) |
| `algorithm` | 알고리즘 | O (Text Recognition) |
| `ref_text` | 참고문헌 | O (Text Recognition) |
| `table_caption` | 테이블 캡션 | O (Text Recognition) |
| `image_caption` | 이미지 캡션 | O (Text Recognition) |
| `code_caption` | 코드 캡션 | O (Text Recognition) |
| `table_footnote` | 테이블 각주 | O (Text Recognition) |
| `image_footnote` | 이미지 각주 | O (Text Recognition) |
| `phonetic` | 주음부호 | O (Text Recognition) |
| `unknown` | 미분류 | O (Text Recognition) |
| `image` | 이미지 | X (skip) |
| `list` | 리스트 | X (skip) |
| `equation_block` | 수식 블록 | X (skip) |
| `header` | 머리말 | 설정에 따라 skip 가능 |
| `footer` | 꼬리말 | 설정에 따라 skip 가능 |
| `page_number` | 페이지 번호 | 설정에 따라 skip 가능 |
| `page_footnote` | 각주 | 설정에 따라 skip 가능 |
| `aside_text` | 사이드 텍스트 | 설정에 따라 skip 가능 |

### 3.4 배치 처리 모드

```python
def batch_two_step_extract(self, images, priority=None, not_extract_list=None):
    if self.batching_mode == "concurrent":     # http-client, vllm-async-engine, lmdeploy
        return self.concurrent_two_step_extract(...)   # asyncio 기반 동시 처리
    else:  # "stepping"                                # transformers, vllm-engine
        return self.stepping_two_step_extract(...)     # 동기 배치 처리
```

- **Concurrent 모드**: 페이지별로 layout → extract를 비동기로 병렬 실행
- **Stepping 모드**: 모든 페이지 layout 먼저 → 모든 region extract 배치

### 3.5 VLM 추론 백엔드 (6가지)

| 백엔드 | 방식 | 배치 모드 | 플랫폼 |
|:---|:---|:---|:---|
| `transformers` | HuggingFace 로컬 | stepping | 전체 |
| `vllm-engine` | vLLM 동기 (LLM) | stepping | Linux |
| `vllm-async-engine` | vLLM 비동기 (AsyncLLM) | concurrent | Linux |
| `lmdeploy-engine` | LMDeploy 비동기 | concurrent | Linux/Windows |
| `mlx-engine` | Apple MLX | stepping | macOS |
| `http-client` | OpenAI 호환 HTTP API | concurrent | 전체 |

---

## 4. Hybrid 엔진 상세

### 4.1 동작 모드 결정

```python
# hybrid_analyze.py
_ocr_enable = ocr_classify(pdf_bytes)              # PDF가 스캔본인지 판별
_vlm_ocr_enable = _should_enable_vlm_ocr(          # VLM으로 OCR까지 할지 결정
    _ocr_enable, language, inline_formula_enable
)
# True 조건: _ocr_enable AND language in ["ch","en"] AND inline_formula_enable
```

### 4.2 Mode A: VLM-Only (`_vlm_ocr_enable=True`)

```
이미지 → VLM batch_two_step_extract()
       → layout (21개 타입) + content (텍스트/테이블/수식)
       → 전문 모델 없이 VLM 결과만 사용
       → middle_json 생성
```

- 텍스트 PDF이면서 중/영문이고 수식 활성화 시 선택
- 가장 빠름, VLM만 사용

### 4.3 Mode B: VLM + 전문 모델 (`_vlm_ocr_enable=False`)

```
이미지
  │
  ├─ VLM batch_two_step_extract(not_extract_list=[text, title, header, ...])
  │   └─ Layout 검출 + 테이블/수식/이미지 콘텐츠만 추출
  │   └─ text/title 등은 bbox만, 콘텐츠는 비워둠
  │
  ├─ 이미지 마스킹: image/table/equation 영역 → 흰색 칠하기
  │
  ├─ MFD (YOLOv8): 마스킹된 이미지에서 인라인 수식 검출
  ├─ MFR (UniMERNet): 검출된 수식 영역 → LaTeX 인식
  │
  ├─ PaddleOCR text_detector: 텍스트 영역에서 글자 검출
  │   └─ 해상도별 그룹핑 (64px 단위) → 배치 처리
  ├─ PaddleOCR ocr(): 검출된 글자 영역 인식
  │
  └─ middle_json 생성: VLM 결과 + 전문 모델 결과 병합
```

### 4.4 Hybrid 배치 크기 조정

GPU VRAM에 따라 동적 스케일링:

| VRAM | batch_ratio | OCR 배치 크기 |
|:---|:---|:---|
| 32GB+ | 16 | 16 × base |
| 16-32GB | 8 | 8 × base |
| 12-16GB | 4 | 4 × base |
| 8-12GB | 2 | 2 × base |
| <8GB | 1 | 1 × base |

---

## 5. Allgaznie vs MinerU 아키텍처 비교

### 5.1 파이프라인 구조 비교

```
[Allgaznie]
  이미지 → PP-DocLayoutV3 (DETR, 33M params)    → bbox + 25개 카테고리
       → cv2 crop                                → region 이미지
       → 외부 VLM (GLM/Paddle/DeepSeek, 2B+)    → per-region 텍스트
       → Markdown 조립

[MinerU VLM-Only]
  이미지 → MinerU2.5 VLM (Qwen2VL, 1.2B)        → bbox + 21개 타입 (Layout Detection)
       → PIL crop + rotate                       → region 이미지
       → MinerU2.5 VLM (같은 모델)               → per-region 텍스트 (Content Extract)
       → post_process + middle_json

[MinerU Hybrid]
  이미지 → MinerU2.5 VLM                         → bbox + 일부 콘텐츠
       → PaddleOCR                               → 텍스트 영역 OCR 보강
       → MFD + UniMERNet                         → 인라인 수식 보강
       → middle_json
```

### 5.2 핵심 차이점

| 구분 | Allgaznie | MinerU |
|:---|:---|:---|
| **Layout 모델** | PP-DocLayoutV3 (DETR, object detection) | MinerU2.5 VLM (text generation → bbox 파싱) |
| **Layout 입력 크기** | 800×800 | 1036×1036 |
| **Layout 출력** | 직접 bbox + class (tensor) | structured text → regex 파싱 |
| **Layout-Recognition 관계** | 독립 모델 2개 | 같은 VLM 1개가 두 역할 |
| **VLM 교체** | 자유로움 (HTTP API 기반) | MinerU2.5 전용 (프롬프트/포맷 종속) |
| **카테고리 체계** | 25개 (PP-DocLayout 기준) | 21개 (MinerU 자체 정의) |
| **회전 보정** | 없음 | 0°/90°/180°/270° 자동 보정 |
| **전문 모델 보조** | 없음 | Hybrid: PaddleOCR + MFD + UniMERNet |
| **GPU 메모리** | ~66MB (layout) + VLM 서버 별도 | ~2.6GB (VLM) + ~1GB (전문 모델) |
| **추론 속도** | VLM 서버 의존 | ~5초/페이지 (L4 기준) |

### 5.3 프롬프트 비교

| 태스크 | Allgaznie (GLM-OCR) | MinerU |
|:---|:---|:---|
| Layout | — (object detection) | `"\nLayout Detection:"` |
| Text | `"Text Recognition:"` | `"\nText Recognition:"` |
| Table | `"Table Recognition:"` | `"\nTable Recognition:"` |
| Formula | `"Formula Recognition:"` | `"\nFormula Recognition:"` |

프롬프트가 거의 동일하지만, MinerU의 VLM은 자체 학습된 MinerU2.5이므로 다른 VLM에서는 이 프롬프트가 작동하지 않음.

### 5.4 카테고리 매핑 비교

| Allgaznie (PP-DocLayoutV3) | 태스크 | MinerU 대응 |
|:---|:---|:---|
| text, doc_title, paragraph_title, ... (12개) | text | text, title, ref_text, ... |
| table | table | table |
| display_formula, inline_formula | formula | equation |
| chart, image | skip | image |
| header, footer, number, ... (8개) | abandon | header, footer, page_number, aside_text |
| — | — | code, algorithm, list (Allgaznie에 없음) |
| seal, vertical_text (Allgaznie에만 있음) | text | — |

---

## 6. 통합 가능성 평가

### 6.1 접근 A: Allgaznie LayoutDetector를 VLM 기반으로 추상화

**개요**: `LayoutDetector` 인터페이스를 추상화하여 PP-DocLayoutV3와 VLM 기반 layout 모두 지원.

```python
class LayoutDetector(Protocol):
    def detect(self, images, image_paths=None) -> list[list[Detection]]: ...

class PPDocLayoutDetector(LayoutDetector):      # 현재 구현
    # PP-DocLayoutV3 DETR 모델

class VLMLayoutDetector(LayoutDetector):         # 새로 필요
    # VLM에 "\nLayout Detection:" → bbox 파싱
    # MinerU2.5 전용 출력 포맷 종속
```

**문제점**:
- MinerU2.5의 layout 출력 포맷(`<|box_start|>...`)은 해당 VLM 전용
- 같은 VLM이 layout과 recognition을 모두 하므로, layout과 recognition이 서로 다른 서버를 쓸 수 없음
- GPU 메모리: VLM(~2.6GB)이 layout에 사용되면 PP-DocLayoutV3(~66MB) 대비 40배 더 소비
- 실질적 이점 불분명: MinerU의 강점은 VLM 자체가 아니라 hybrid의 전문 모델 보조에서 옴

**결론**: 대규모 리팩터링 대비 이점이 적음. **비권장**.

### 6.2 접근 B: MinerU를 별도 클라이언트로 래핑 (현재 방식 개선)

**개요**: 기존 `MinerUClient`를 개선하여 Allgaznie 프레임워크와 동일한 인터페이스 제공.

```python
# client.py에 이미 존재
class MinerUClient:
    def infer(self, image, prompt, max_tokens=4096) -> tuple[str, float]: ...
```

**장점**:
- MinerU의 전체 파이프라인(VLM + 전문 모델)을 그대로 활용
- 추가 구현 비용 최소
- MinerU 업데이트에 대한 종속성 최소

**단점**:
- Allgaznie의 모듈별 최적화(GPU decode, vectorized NMS 등) 적용 불가
- MinerU 내부 파이프라인 제어 불가

**결론**: 현실적 최선. **권장**.

### 6.3 접근 C: MinerU VLM을 Allgaznie의 VLM 슬롯에 교체

**개요**: MinerU2.5 VLM을 vLLM 서버에 올리고, Allgaznie의 `VLMClient`로 호출.

```
이미지 → PP-DocLayoutV3 (Allgaznie layout)
     → Region Crop
     → MinerU2.5 VLM (Allgaznie VLMClient로 호출)
     → Markdown 조립
```

**문제점**:
- MinerU2.5의 recognition 프롬프트(`"\nText Recognition:"` 등)는 MinerU 학습 데이터에 최적화
- PP-DocLayoutV3의 25개 카테고리와 MinerU의 21개 타입 간 매핑 불일치
- MinerU2.5는 1.2B로 GLM-OCR(2B+) 대비 소형 → recognition 품질이 더 낮을 수 있음
- MinerU의 강점인 hybrid 전문 모델 보조를 쓸 수 없음

**결론**: 기술적으로 가능하나 이점이 불명확. **추가 검증 필요**.

---

## 7. 결론 및 권장안

### 7.1 핵심 발견

1. **MinerU의 two-step extraction은 구조적으로 Allgaznie와 동일**: Layout Detection → Region Crop → Per-Region VLM 추론. 차이는 layout 모델이 DETR이 아닌 VLM이라는 점.

2. **통합의 본질적 장벽**: MinerU는 같은 VLM 모델이 layout과 recognition을 모두 수행하며, layout 출력 포맷이 해당 VLM에 종속. Allgaznie의 "layout 모델과 VLM 모델이 독립" 설계와 근본적으로 다름.

3. **MinerU의 벤치마크 성능은 hybrid 모드의 전문 모델 보조에서 옴**: VLM-only 모드보다 PaddleOCR + MFD + UniMERNet을 추가한 hybrid가 더 높은 성능. 이 부분은 Allgaznie로 재현 불가.

### 7.2 권장안

| 우선순위 | 방안 | 설명 |
|:---|:---|:---|
| **1 (권장)** | MinerUClient 래핑 유지 | 기존 `client.py` MinerUClient로 벤치마크 호환성 확보. Allgaznie 통합 불필요. |
| 2 (선택) | MinerU2.5 VLM을 Allgaznie VLM 슬롯에 테스트 | PP-DocLayoutV3 + MinerU2.5 VLM 조합으로 벤치마크 실행 후 성능 비교. 이점이 확인되면 `VLM_MODEL_IDS`에 추가. |
| 3 (비권장) | LayoutDetector 추상화 리팩터링 | VLM 기반 layout detector 지원을 위한 대규모 리팩터링. 현 시점에서 ROI 불분명. |

### 7.3 MinerU2.5 VLM 슬롯 테스트 시 필요 작업 (접근 C)

만약 추후 테스트하기로 결정 시:

1. MinerU2.5 VLM을 vLLM 서버에 올리기:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model opendatalab/MinerU2.5-2509-1.2B \
       --trust-remote-code \
       --gpu-memory-utilization 0.85
   ```

2. `allgaznie/vlm.py`에 MinerU 프롬프트 추가:
   ```python
   REGION_PROMPTS["MinerU2.5"] = {
       "text": "\nText Recognition:",
       "table": "\nTable Recognition:",
       "formula": "\nFormula Recognition:",
   }
   ```

3. `config.py`에 모델 엔트리 추가:
   ```python
   "allgaznie-mineru": ModelConfig(
       name="Allgaznie-MinerU",
       model_id="opendatalab/MinerU2.5-2509-1.2B",
       backend="allgaznie",
       vllm_args=["--trust-remote-code", "--gpu-memory-utilization", "0.85"],
   )
   ```

4. 벤치마크 비교 실행 후 판단.

---

## 부록: MinerU 소스 참조

| 파일 | 역할 |
|:---|:---|
| `mineru_vl_utils/mineru_client.py` | Two-step extraction 핵심 구현 (MinerUClient, MinerUClientHelper) |
| `mineru_vl_utils/structs.py` | ContentBlock, BlockType 정의 (21개 타입) |
| `mineru_vl_utils/vlm_client/base_client.py` | VlmClient 인터페이스 + 6개 백엔드 팩토리 |
| `mineru/backend/hybrid/hybrid_analyze.py` | Hybrid 엔진 오케스트레이션 |
| `mineru/backend/vlm/vlm_analyze.py` | VLM-only 엔진 |
| `mineru/backend/pipeline/model_init.py` | 모델 싱글톤 로딩 |
| `mineru/cli/common.py` | `do_parse()` 진입점 |

---

*분석 대상: MinerU v2.7.6, mineru-vl-utils v0.1.22*
*MinerU 레포 위치: `/root/MinerU/`*
