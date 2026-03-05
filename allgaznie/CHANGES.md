# Allgaznie SDK — GLM-OCR SDK 갭 클로저 변경 문서

## 개요

GLM-OCR 공식 SDK 파이프라인과 Allgaznie 파이프라인을 상세 비교 분석한 결과 16개 갭을 식별하고 모두 수정했다.
이 문서는 각 갭의 원인, 수정 내용, 적용 파일을 기록한다.

---

## 변경 요약

| # | 갭 | 파일 | 변경 내용 |
|---|-----|------|----------|
| 1 | NMS iou_diff 임계값 | `postprocess.py` | 0.95 → 0.98 (SDK 일치) |
| 2 | 대형 이미지 필터링 부재 | `postprocess.py`, `layout.py` | `filter_large_images()` 추가 (82%/93% 면적 임계값) |
| 3 | Containment per-class 모드 부재 | `postprocess.py`, `layout.py` | `_CONTAINMENT_MODE` dict + preserve labels (image/seal/chart) |
| 4 | order_seq 미사용 | `layout.py` | `order_seq` 텐서 추출 → `np.argsort` 정렬 (spatial fallback) |
| 5 | polygon_points 미사용 | `layout.py`, `preprocess.py` | `polygon_points` 추출 → `cv2.fillPoly` 마스킹 |
| 6 | smart_resize 부재 | `preprocess.py`, `vlm.py` | `smart_resize()` 구현 (28×28 factor, min/max pixels) |
| 7 | VLM generation 파라미터 고정 | `vlm.py`, `__init__.py` | temperature, top_p, top_k, repetition_penalty 설정 가능 |
| 8 | Content cleaning 부재 | `postprocess.py` | `clean_content()`: \t 제거, 반복 문장부호, 긴 텍스트 중복 제거 |
| 9 | Title formatting 부재 | `postprocess.py` | doc_title→`#`, paragraph_title→`##` |
| 10 | Formula formatting 부재 | `postprocess.py` | `$$\n...\n$$` 래핑 (`$$`, `\[`, `\(` 처리) |
| 11 | Text formatting 부재 | `postprocess.py` | 불릿 정규화, 번호 리스트, 줄바꿈 정규화 |
| 12 | Formula number merging 부재 | `postprocess.py` | `merge_formula_numbers()`: 양방향 formula+formula_number → `\tag{}` |
| 13 | Hyphenated word merging 부재 | `postprocess.py` | `merge_hyphenated_blocks()`: Zipf 빈도 ≥ 2.5 검증 |
| 14 | Bullet point auto-detection 부재 | `postprocess.py` | `format_bullet_points()`: bbox 정렬 기반 (10px 임계값) |
| 15 | Tiny box pre-filter 부재 | `layout.py` | `_prefilter_tiny_boxes()`: 1 mask pixel 미만 박스 마스킹 |
| 16 | Detection 데이터 부족 | `layout.py`, `__init__.py` | Detection에 `order`, `polygon` 필드 추가; regions에 `label` 전달 |

---

## 상세 변경 내역

### 1. NMS iou_diff 임계값 (postprocess.py:46)

**문제**: cross-class NMS 임계값이 0.95로, SDK의 0.98보다 공격적으로 suppression.
**수정**: `iou_diff` 기본값을 `0.98`로 변경.

```python
def vectorized_nms(boxes, scores, classes, iou_same=0.6, iou_diff=0.98):
```

### 2. 대형 이미지 필터링 (postprocess.py:94-129, layout.py:199-210)

**문제**: 페이지 전체를 덮는 "image" 감지가 필터링되지 않아 다른 영역과 간섭.
**수정**: `filter_large_images()` 함수 추가. 가로 이미지는 82%, 세로는 93% 면적 임계값 적용.
`layout.py`의 `detect()` 메서드에서 NMS 후 호출.

```python
def filter_large_images(boxes, labels, label_names, img_size) -> np.ndarray:
    area_thres = 0.82 if img_w > img_h else 0.93
    # "image" 라벨 중 임계값 초과 박스 제거
```

### 3. Containment per-class 모드 (postprocess.py:28-38, 132-196)

**문제**: 단순 containment만 적용. SDK는 클래스별 모드(large/small)와 보존 클래스를 사용.
**수정**:
- `_PRESERVE_LABELS = {"image", "seal", "chart"}` — 절대 제거 불가
- `_CONTAINMENT_MODE` dict — 25개 클래스별 모드 (reference=small, 나머지=large)
- `vectorized_containment()`에 `label_names`, `class_ids` 파라미터 추가
- `layout.py`에서 호출 시 두 파라미터 전달

### 4. order_seq 사용 (layout.py:166-168, 244-247)

**문제**: PP-DocLayoutV3가 출력하는 `order_seq` (모델 예측 읽기 순서)를 무시하고 spatial 정렬만 사용.
**수정**: `post_process_object_detection()` 결과에서 `order_seq` 추출 → `np.argsort(final_order)`로 정렬. order_seq가 없으면 `spatial_reading_order()` fallback.

```python
if "order_seq" in raw:
    order_seq = raw["order_seq"].cpu().numpy()
# ...
if final_order is not None:
    order = np.argsort(final_order).tolist()
else:
    order = spatial_reading_order(final_boxes)
```

### 5. polygon_points + 마스킹 (layout.py:170-171, preprocess.py:130-143)

**문제**: PP-DocLayoutV3의 polygon 출력을 무시. 직사각형 bbox만 사용해 배경 노이즈 포함.
**수정**:
- `layout.py`: `polygon_points` 추출하여 `Detection.polygon`에 저장
- `preprocess.py:crop_regions()`: polygon이 3점 이상이면 `cv2.fillPoly`로 다각형 외부를 흰색으로 마스킹

```python
# preprocess.py
mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
cv2.fillPoly(mask, [poly_points], 1)
output = np.full_like(region, 255, dtype=np.uint8)
cv2.copyTo(region, mask, output)
```

### 6. smart_resize (preprocess.py:148-186, vlm.py:56-59)

**문제**: VLM 전송 전 이미지 리사이즈 없음. SDK는 28×28 factor 정렬 + min/max pixel 제한 적용.
**수정**:
- `preprocess.py:smart_resize()` — 28×28 factor 정렬, 12544~1003520 pixel 범위 유지
- `vlm.py:_image_to_base64_jpeg()` — JPEG 인코딩 전 `smart_resize()` 적용

### 7. VLM generation 파라미터 (vlm.py:76-91, __init__.py:48-53)

**문제**: temperature, top_p 등이 하드코딩. SDK는 config에서 설정 가능.
**수정**:
- `AllgaznieConfig`에 필드 추가: `vlm_temperature=0.1`, `vlm_top_p=0.1`, `vlm_top_k=1`, `vlm_repetition_penalty=1.1`, `vlm_min_pixels=12544`, `vlm_max_pixels=1003520`
- `VLMClient`에 모든 파라미터 전달
- `_infer_one()`에서 `temperature`, `top_p` → OpenAI API, `top_k`, `repetition_penalty` → `extra_body`

### 8. Content Cleaning (postprocess.py:247-309)

**문제**: VLM 출력을 가공 없이 그대로 사용. SDK는 다단계 정리 적용.
**수정**: `clean_content()` 구현:
- 선행/후행 `\t` 제거
- 반복 문장부호 제한 (`.`, `·`, `_`, `\_` → 최대 3개)
- 2048자 이상 텍스트: `_clean_repeated_content()`로 반복 패턴/행 제거

### 9. Title Formatting (postprocess.py:342-349)

**문제**: `doc_title`, `paragraph_title` 라벨이 일반 텍스트로 출력.
**수정**: `format_content()`에서:
- `doc_title` → 기존 `#` 제거 후 `# ` 접두사
- `paragraph_title` → 기존 불릿/`#` 제거 후 `## ` 접두사

### 10. Formula Formatting (postprocess.py:352-363)

**문제**: 수식이 `$$...$$`, `\[...\]`, `\(...\)` 형태로 올 수 있는데 통일되지 않음.
**수정**: `format_content()`에서 모든 형태를 `$$\n...\n$$` 블록 형식으로 통일.

### 11. Text Formatting (postprocess.py:366-385)

**문제**: 불릿(·, •, *), 번호 리스트, 줄바꿈이 정규화되지 않음.
**수정**: `format_content()`에서:
- `·`, `•`, `*` → `- ` (마크다운 불릿)
- `(1)`, `（A）` → `(1) ` 형식
- `1.`, `1)`, `A.` → `1. ` 형식
- 단일 줄바꿈 → 이중 줄바꿈 (`\n` → `\n\n`)

### 12. Formula Number Merging (postprocess.py:394-448)

**문제**: `formula_number` 라벨이 별도 블록으로 출력. SDK는 인접 수식에 `\tag{}`로 병합.
**수정**: `merge_formula_numbers()`:
- Case 1: `formula_number` → `formula` 순서: `\tag{number}` 삽입
- Case 2: `formula` → `formula_number` 순서: 동일 처리
- 괄호 자동 제거: `(1)` → `1`

### 13. Hyphenated Word Merging (postprocess.py:452-500)

**문제**: 줄 끝 하이픈으로 분리된 단어가 병합되지 않음 (예: "continu-" + "ation").
**수정**: `merge_hyphenated_blocks()`:
- 텍스트 블록이 `-`로 끝나고 다음 블록이 소문자로 시작하면 병합 후보
- `wordfreq.zipf_frequency(merged_word, "en") >= 2.5` 검증으로 오병합 방지

### 14. Bullet Point Auto-detection (postprocess.py:504-539)

**문제**: 불릿이 누락된 리스트 항목이 그대로 출력.
**수정**: `format_bullet_points()`:
- 이전/다음 블록이 `- `로 시작하고, 현재 블록이 아닌 경우
- bbox 좌측 정렬 차이 ≤ 10px이면 `- ` 접두사 추가

### 15. Tiny Box Pre-filter (layout.py:274-298)

**문제**: 1 mask pixel 미만의 극소 박스가 후처리를 통과.
**수정**: `_prefilter_tiny_boxes()`:
- `pred_boxes`의 너비/높이가 `1/mask_w`, `1/mask_h` 미만인 박스의 logits를 -100으로 마스킹
- `torch.compile` + `inference_mode` 환경에서 inplace 오류 방지를 위해 `logits.clone()` 사용

### 16. Detection 데이터 확장 (layout.py:52-60, __init__.py:199-218)

**문제**: `Detection`에 reading order와 polygon 정보 없음. `assemble_markdown()`에 label 미전달.
**수정**:
- `Detection` dataclass에 `order: int = -1`, `polygon: list[list[int]] = field(default_factory=list)` 추가
- `__init__.py`의 regions 딕셔너리에 `"label"` 키 포함 → `format_content()`에서 title/formula 포맷 적용

---

## 파일별 변경 규모

| 파일 | 변경 유형 | 핵심 변경 |
|------|----------|----------|
| `allgaznie/postprocess.py` | 대폭 확장 (→595줄) | NMS iou_diff, filter_large_images, containment per-class, content cleaning, title/formula/text formatting, formula number merging, hyphenated merging, bullet detection |
| `allgaznie/layout.py` | 확장 (→317줄) | order_seq 추출, polygon_points 추출, filter_large_images 호출, containment per-class 호출, tiny box pre-filter, Detection dataclass 확장 |
| `allgaznie/preprocess.py` | 확장 (→187줄) | polygon masking (cv2.fillPoly), smart_resize 함수 |
| `allgaznie/vlm.py` | 확장 (→163줄) | generation 파라미터, smart_resize 인코딩 |
| `allgaznie/__init__.py` | 확장 (→231줄) | config 필드 추가, label 전달 |

---

## SDK 참조

공식 GLM-OCR SDK 소스 (`/tmp/GLM-OCR/glmocr/`):
- `utils/layout_postprocess_utils.py` — NMS, containment, large image filter, reading order
- `utils/image_utils.py` — smart_resize, crop_image_region (polygon masking)
- `postprocess/result_formatter.py` — content cleaning, title/formula/text formatting, merging
- `utils/result_postprocess_utils.py` — clean_repeated_content, clean_formula_number
- `layout/layout_detector.py` — tiny box pre-filter, order_seq/polygon extraction
- `dataloader/page_loader.py` — VLM generation params, smart_resize encoding
- `config.yaml` — per-class containment modes, prompts, thresholds

---

## 검증

1. **단위 테스트**: 모든 post-processing 함수 (NMS, containment, smart_resize, polygon masking, formatting 등) 테스트 통과
2. **E2E 테스트**: 실제 이미지로 layout detection 실행 — order_seq, polygon_points 정상 추출 확인
3. **코드 검증**: 16개 갭 항목 × 코드 패턴 매칭으로 전수 검증 (16/16 통과)
4. **재추론**: OmniDocBench 전체 재추론 진행 중 (allgaznie-glm, -paddle, -deepseek)
