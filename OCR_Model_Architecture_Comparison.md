# OCR 모델 아키텍처 비교 분석

> 작성일: 2026-02-19 (크로스체크 완료)
> 비교 대상: MinerU, GLM-OCR, PaddleOCR-VL, DeepSeek-OCR 2, Upstage Document Parse, Azure Document Intelligence Layout

---

## 1. 요약: 구조 유형 분류

| 모델 | 개발사 | 구조 유형 | 오픈소스 | 핵심 특징 |
|------|--------|-----------|----------|-----------|
| **MinerU** | OpenDataLab | 하이브리드 (VLM 기반, 2단계) | O | v1.x: 파이프라인 → v2.5+: VLM 기반 2단계로 진화 |
| **GLM-OCR** | Zhipu AI | 하이브리드 (2단계) | O (MIT+Apache 2.0) | 레이아웃 감지 분리 + 통합 VLM 인식 |
| **PaddleOCR-VL** | Baidu | 하이브리드 (2단계) | O (Apache 2.0) | 레이아웃+읽기순서 분리 + 통합 VLM 인식 |
| **DeepSeek-OCR 2** | DeepSeek | End-to-End 단일 모델 | O | Visual Causal Flow로 읽기순서 자동 학습 |
| **Upstage Document Parse** | Upstage | 멀티스테이지 파이프라인 | X (상용 API) | 전통 파이프라인 + Enhanced 모드에서 VLM 추가 |
| **Azure Document Intelligence** | Microsoft | 멀티스테이지 파이프라인 | X (클라우드/컨테이너) | 엔터프라이즈급 파이프라인, 300+ 언어 |

### 구조 유형 스펙트럼

```
전통 파이프라인                    하이브리드                      End-to-End
(단계별 전문 모델)          (레이아웃 분리 + VLM 인식)          (단일 모델)

 MinerU 1.x                 MinerU 2.5+                   DeepSeek-OCR 2
 Upstage                    GLM-OCR
 Azure Layout               PaddleOCR-VL
```

---

## 2. 개별 모델 상세 분석

### 2.1 MinerU (OpenDataLab)

- **버전**: 최신 2.7.6 (2026/02/06)
- **구조**: v1.x는 멀티스테이지 파이프라인 → **v2.5+는 VLM 기반 2단계 하이브리드**로 진화
- **라이선스**: AGPL-3.0

#### v2.5+ (현재 최신, 2.7.6 포함)

```
PDF/이미지
    ↓
① 글로벌 레이아웃 분석 (MinerU2.5-2509-1.2B VLM 1단계)
    │  썸네일 기반 NaViT + Patch Merger + LLM으로 레이아웃 파악
    ↓
② 영역별 인식 (MinerU2.5-2509-1.2B VLM 2단계)
    │  각 영역을 VLM으로 통합 인식 (텍스트/표/수식)
    ↓
마크다운/JSON 출력
```

**내부 모델 구성 (v2.5+)**:

| 단계 | 모델 | 역할 |
|------|------|------|
| VLM | MinerU2.5-2509-1.2B | 레이아웃 분석 + 영역 인식 통합 (decoupled 2-stage) |
| 추론 서빙 | vLLM | 모델 서빙 (v2.5에서 SGLang → vLLM으로 전환) |

#### v1.x (레거시, Docker 2.1.9에 해당)

```
PDF/이미지
    ↓
① 레이아웃 감지 (DocLayout-YOLO) ─── 문서 영역 바운딩 박스 분류
    ↓
② 레이아웃 분석 (PDF-Extract-Kit-1.0) ── 문서 구조/읽기 순서 파악
    ↓
③ 영역별 인식 (OCR/표/수식 각각 별도 모델)
    ↓
④ VLM 후처리 (MinerU2.0-2505-0.9B, SGLang 서빙)
    ↓
마크다운/JSON 출력
```

> **참고**: 현재 Docker에 설치된 MinerU 2.1.9는 v1.x 아키텍처(DocLayout-YOLO + PDF-Extract-Kit + 0.9B VLM + SGLang)를 사용합니다. 최신 v2.5+와는 아키텍처가 다릅니다.

**장점**: 완전 오픈소스, v2.5+에서 VLM 기반으로 정확도 향상
**단점**: v1.x는 앞단 에러 전파 문제, v2.5+는 단일 VLM 의존도 증가

---

### 2.2 GLM-OCR (Zhipu AI)

- **구조**: 하이브리드 2단계
- **라이선스**: MIT (모델 가중치) + Apache 2.0 (PP-DocLayoutV3 코드)

```
PDF/이미지
    ↓
① 레이아웃 감지 (PP-DocLayout-V3) ── 23개 카테고리, RT-DETR 기반
    ↓
② 영역별 병렬 인식 (GLM-OCR 0.9B VLM) ── 텍스트/표/수식 통합 처리
    ↓
JSON/마크다운 출력
```

**VLM 내부 구조 (0.9B)**:

| 컴포넌트 | 설명 | 파라미터 |
|----------|------|----------|
| CogViT (비전 인코더) | ViT 기반 이미지 인코딩 | ~400M (추정, 공식 미공개) |
| Cross-Modal Connector | 비전→언어 브릿지 | 경량 |
| GLM-0.5B (언어 디코더) | 텍스트/마크다운/LaTeX 생성 | ~500M |

**훈련 기법**:
- Multi-Token Prediction (MTP) Loss: 여러 토큰 동시 예측으로 "의미적 교정" 효과
- Stable Full-Task Reinforcement Learning: 전체 OCR 태스크에 RL 적용

**지원 태스크**: 문서 파싱, 정보 추출 (JSON), 장면 텍스트, 인장/필기/다국어
**성능**: OmniDocBench v1.5 **94.62점 (1위)**, 처리속도 1.86 pages/sec (PDF 기준, 단일 replica/concurrency)

---

### 2.3 PaddleOCR-VL (Baidu)

- **버전**: 최신 PaddleOCR-VL-1.5 (2026/01)
- **구조**: 하이브리드 2단계
- **라이선스**: Apache 2.0

```
PDF/이미지
    ↓
① 레이아웃 분석 (PP-DocLayoutV2/V3)
    ├─ RT-DETR: 20+ 카테고리 영역 감지
    └─ Pointer Network: 읽기 순서 예측 (6-layer Transformer)
    ↓
② 영역별 인식 (PaddleOCR-VL 0.9B)
    ├─ OCR (109+ 언어)
    ├─ 표 인식
    ├─ 수식 인식 (LaTeX)
    └─ 차트 인식
    ↓
마크다운/JSON 출력
```

**VLM 내부 구조 (0.9B)**:

| 컴포넌트 | 설명 |
|----------|------|
| NaViT 비전 인코더 | 동적 해상도 지원, Keye-VL 초기화 |
| 2-layer MLP Projector | GELU 활성화, 비전→언어 브릿지 |
| ERNIE-4.5-0.3B 디코더 | 3D-RoPE 위치 인코딩 |

**기존 PaddleOCR(PP-OCRv5)와 비교**:

| | PP-OCRv5 (전통) | PaddleOCR-VL |
|---|---|---|
| 구조 | 5개 모듈 (핵심 2단계 + 전처리 3개 선택적) | 2단계 하이브리드 |
| 인식 | CTC 기반 텍스트 디코더 | VLM (자기회귀) |
| 기능 | 텍스트 OCR만 | 표/수식/차트 포함 |
| 모델 크기 | ~165MB (서버 기준) | ~900M |

**v1.5 업그레이드**: 사각 바운딩 → 마스크 기반 감지 + 사각형/다각형 출력, 읽기순서를 Transformer 디코더에 통합, 111개 언어
**성능**: OmniDocBench v1.5 94.5%

---

### 2.4 DeepSeek-OCR 2 (DeepSeek)

- **구조**: **End-to-End 단일 모델** (파이프라인 없음)
- **핵심 혁신**: Visual Causal Flow

```
이미지 입력
    ↓
① SAM-base 비전 토크나이저 (80M)
    │ 16x 압축 → 256~1,120 비전 토큰
    ↓
② DeepEncoder V2 (Qwen2-0.5B 기반)
    │ 비전 토큰: 양방향 어텐션 (전체 레이아웃 인식)
    │ 쿼리 토큰: 인과적 어텐션 (읽기 순서 자동 학습)
    ↓
③ DeepSeek-3B MoE 디코더 (DeepSeek-3B-A500M)
    │ 64개 라우팅 + 2개 공유 expert, 6개 활성 (토큰당 ~500M)
    ↓
마크다운/HTML/구조화 텍스트 출력
```

**내부 구조**:

| 컴포넌트 | 아키텍처 | 파라미터 |
|----------|----------|----------|
| Vision Tokenizer | SAM-base + 2 Conv layers | ~80M |
| DeepEncoder V2 | Qwen2-0.5B (커스텀 어텐션 마스크) | ~500M |
| Decoder | DeepSeek-3B-A500M MoE (64 라우팅 + 2 공유 expert) | 3B 총 / ~500M 활성 |

**Visual Causal Flow의 핵심**:
- 별도 레이아웃 감지 모델 없이, 인코더가 읽기 순서를 스스로 학습
- 비전 토큰은 양방향(전체 페이지 인식), 쿼리 토큰은 인과적(순서 결정)
- 다른 모델이 6,000+ 비전 토큰 필요한 반면, 최대 **1,120개**로 처리

**훈련 (3단계)**:

| 단계 | 내용 | 규모 |
|------|------|------|
| 1. Pretraining | 비전 토크나이저 + 인코더 공동 훈련 | 40K iter, 160 A100 |
| 2. Query Enhancement | 인코더 + 디코더 공동 최적화 | 15K iter |
| 3. Specialization | 디코더만 업데이트 | 20K iter |

**성능**: OmniDocBench v1.5 91.09점, 읽기순서 edit distance 0.057
**장점**: 에러 전파 없음, 토큰 효율적
**단점**: 블랙박스, 단계별 디버깅 불가

---

### 2.5 Upstage Document Parse

- **구조**: 멀티스테이지 파이프라인 (MinerU와 가장 유사)
- **라이선스**: 상용 API (비공개)

```
PDF/이미지/DOCX/PPTX/XLSX
    ↓
① 레이아웃 감지 ── 12개 요소 타입 분류 + 바운딩 박스
    ↓
② OCR ── 텍스트 추출 (auto: 디지털 PDF 직접 추출 / force: 항상 OCR)
    ↓
③ 읽기순서 직렬화 ── 요소 정렬, 캡션-테이블/그림 연결
    ↓
④ 포맷 변환 ── HTML / 마크다운 / 텍스트
    ↓
⑤ [Enhanced 모드만] VLM 후처리
    ├─ 복잡한 차트 → 구조화 데이터 + 자연어 설명
    ├─ 이미지/도표 → 요약 설명
    └─ 체크박스 → 체크 상태 감지
    ↓
JSON 응답
```

**감지 요소 (12개)**: `table`, `paragraph`, `figure`, `chart`, `heading1`, `header`, `footer`, `caption`, `equation`, `list`, `index`, `footnote`

**처리 모드**:

| 모드 | 설명 |
|------|------|
| Standard | 텍스트 중심 문서, 단순 표 |
| Enhanced | VLM 추가 (복잡한 차트/도표/체크박스) |
| Auto | 페이지별 복잡도 자동 판단 후 라우팅 |

**내부 모델**: 비공개 (Upstage는 Solar LLM 패밀리 개발사이나, Document Parse에 직접 사용 여부는 미확인)
**성능**: TEDS 93.48점 (표 구조 인식), ~0.6초/페이지
**장점**: 다양한 입력 포맷, Enhanced 모드의 VLM 보강
**단점**: 블랙박스 상용 API, 내부 아키텍처 비공개

---

### 2.6 Azure AI Document Intelligence Layout (Microsoft)

- **버전**: API v4.0 GA (`2024-11-30`), Content Understanding GA (`2025-11-01`)
- **구조**: 멀티스테이지 파이프라인 (공식 문서에서 명시적으로 "파이프라인"이라 칭하지는 않으나, 기능 구성상 추정)
- **라이선스**: 상용 (클라우드 + 온프레미스 컨테이너)

```
PDF/이미지/Office 문서
    ↓
① 전처리 ── 페이지 렌더링, 회전/기울기 보정
    ↓
② OCR (Read Model) ── 고해상도 텍스트 감지, 인쇄/필기 분류, 언어 감지
    ↓
③ 레이아웃 분석 (Deep Learning)
    ├─ 표 감지 + 셀 구조 인식
    ├─ 그림/차트 감지 + 캡션
    └─ 체크박스 감지 (기본 내장)
    ↓
④ 구조 이해
    ├─ 단락 그룹핑 + 역할 분류 (title, heading, footer...)
    ├─ 읽기 순서 결정 (ML 기반)
    └─ 섹션/하위섹션 계층 구조
    ↓
⑤ 출력 조립 ── JSON / 마크다운
```

**내부 모델 (Microsoft Research 기반 추정, 정확한 프로덕션 배포 구성은 비공개)**:

| 기능 | 추정 모델 | 설명 |
|------|-----------|------|
| OCR | TrOCR 계열 (추정) | Transformer 기반, convolution-free |
| 문서 이해 | LayoutLMv3 계열 (추정) | 멀티모달 사전훈련 Transformer |
| 비전 백본 | DiT 계열 (추정) | 자기지도 사전훈련 Document Image Transformer |
| 표 인식 | TSRFormer 계열 (추정) | 표 분리선을 선형 회귀로 예측 |
| 읽기 순서 | ReadingBank 데이터셋 기반 훈련 모델 (추정) | 전용 읽기순서 데이터셋으로 훈련 |

> **주의**: 위 모델들은 Microsoft Research의 Document AI 프로젝트에서 발표된 연구를 기반으로 한 **추정**이며, MS가 프로덕션에서 실제로 어떤 모델을 사용하는지는 공식 확인된 바 없습니다. ReadingBank은 모델이 아니라 **데이터셋**입니다.

**감지 요소**:
- 기하학적 (기본 내장): 단어, 줄, 단락, 표 (행/열/헤더/스팬), 그림, 체크박스(selection marks)
- 기하학적 (유료 add-on): 바코드 (`features=barcodes`), 수식 (`features=formulas`)
- 논리적: `title`, `sectionHeading`, `footnote`, `pageHeader`, `pageFooter`, `pageNumber`

**주요 특징**:
- 300+ 언어 지원 (언어 지정 불필요)
- 최대 2,000페이지 처리
- 인쇄/필기 자동 분류
- 온프레미스 컨테이너 배포 가능 (2025/04~)
- Content Understanding으로 진화 (멀티모달: 텍스트/이미지/오디오/비디오)

**장점**: 엔터프라이즈급 안정성, 방대한 언어 지원, 온프레미스 가능
**단점**: 비공개 아키텍처, VLM 기반 대비 최신 벤치마크 경쟁력 약화 지적

---

## 3. 종합 비교

### 3.1 아키텍처 비교

| | MinerU | GLM-OCR | PaddleOCR-VL | DeepSeek-OCR 2 | Upstage | Azure Layout |
|---|---|---|---|---|---|---|
| **구조** | 하이브리드 (v2.5+) | 하이브리드 | 하이브리드 | End-to-End | 파이프라인 | 파이프라인 (추정) |
| **레이아웃 감지** | VLM 1단계 (v2.5+) | PP-DocLayout-V3 | PP-DocLayoutV2/V3 | 없음 (내재) | 비공개 | DiT/LayoutLM 계열 (추정) |
| **OCR 엔진** | VLM 2단계 (v2.5+) | VLM 통합 | VLM 통합 | VLM 통합 | 비공개 | TrOCR 계열 (추정) |
| **읽기순서** | VLM 내재 | 레이아웃 의존 | Pointer Network | Visual Causal Flow | 비공개 | ML 기반 |
| **VLM 사용** | 핵심 (v2.5+) | 인식 핵심 | 인식 핵심 | 전체 모델 | Enhanced만 | 미사용 |
| **총 파라미터** | 1.2B (v2.5+) | 0.9B | 0.9B | 3B (500M 활성) | 비공개 | 비공개 |

### 3.2 성능 비교 (OmniDocBench v1.5)

| 모델 | 점수 | 비고 |
|------|------|------|
| **GLM-OCR** | **94.62** | 1위, 하이브리드 |
| **PaddleOCR-VL 1.5** | **94.5** | 근소한 차이 |
| **DeepSeek-OCR 2** | **91.09** | End-to-End 최고 |
| MinerU | - | 공식 수치 미확인 |
| Upstage | TEDS 93.48 | 표 구조 특화 벤치마크 |
| Azure Layout | - | 공식 수치 미공개 |

### 3.3 기능 비교

| | MinerU | GLM-OCR | PaddleOCR-VL | DeepSeek-OCR 2 | Upstage | Azure Layout |
|---|---|---|---|---|---|---|
| **오픈소스** | O | O | O | O | X | X |
| **다국어** | O | O | 111개 | O | O | 300+ |
| **표 인식** | O | O | O | O | O | O |
| **수식 인식** | O | O | O | O | O | O (유료 add-on) |
| **필기 인식** | - | O | - | - | - | O |
| **차트 인식** | - | - | O | - | O (Enhanced) | - |
| **온프레미스** | O | O | O | O | X | O (컨테이너) |
| **출력 포맷** | MD | MD/JSON | MD/JSON | MD/HTML | HTML/MD/TXT | JSON/MD |

### 3.4 선택 가이드

| 요구사항 | 추천 모델 | 이유 |
|----------|-----------|------|
| 최고 정확도 | GLM-OCR / PaddleOCR-VL | 벤치마크 1~2위 |
| 토큰 효율성 | DeepSeek-OCR 2 | 최소 비전 토큰 (≤1,120) |
| 모듈 커스터마이징 | MinerU | 각 단계 교체 가능 |
| 엔터프라이즈/규정준수 | Azure Layout | 온프레미스, SLA, 300+ 언어 |
| 빠른 API 통합 | Upstage | 간단한 API, 다양한 입력 포맷 |
| 경량/빠른 처리 | GLM-OCR | 0.9B로 높은 정확도 |

---

## 4. 업계 트렌드

### 진화 방향

```
1세대: 전통 파이프라인 (단계별 전문 모델)
  └─ MinerU 1.x, Upstage, Azure Layout, PP-OCRv5

2세대: 하이브리드 (레이아웃 분리 + VLM 인식 통합)
  └─ MinerU 2.5+, GLM-OCR, PaddleOCR-VL
      → 현재 벤치마크 최고 성능

3세대: End-to-End (단일 모델)
  └─ DeepSeek-OCR 2
      → 아키텍처적으로 가장 진보, 정확도는 아직 하이브리드에 미달
```

### 주요 관찰

1. **하이브리드가 현재 최적**: 레이아웃 감지만 분리하고 나머지를 VLM으로 통합하는 방식이 정확도와 효율성의 균형점
2. **End-to-End의 잠재력**: DeepSeek-OCR 2의 Visual Causal Flow는 혁신적이나, 아직 파이프라인 기반 대비 3~4점 차이
3. **VLM 크기 수렴**: GLM-OCR, PaddleOCR-VL 모두 0.9B로 수렴 — 문서 OCR에 최적화된 모델 크기로 보임
4. **상용 서비스의 딜레마**: Azure, Upstage 등은 VLM 기반 오픈소스에 벤치마크 경쟁력 도전받는 중

---

## 참고 자료

- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [MinerU2.5 Paper (arXiv 2509.22186)](https://arxiv.org/abs/2509.22186)
- [MinerU2.5-2509-1.2B (Hugging Face)](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)
- [GLM-OCR GitHub](https://github.com/zai-org/GLM-OCR)
- [GLM-OCR (Hugging Face)](https://huggingface.co/zai-org/GLM-OCR)
- [PaddleOCR-VL Paper (arXiv 2510.14528)](https://arxiv.org/abs/2510.14528)
- [PaddleOCR-VL-1.5 Paper (arXiv 2601.21957)](https://arxiv.org/abs/2601.21957)
- [DeepSeek-OCR 2 Paper (arXiv 2601.20552)](https://arxiv.org/abs/2601.20552)
- [DeepSeek-OCR 2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [Upstage Document Parse](https://www.upstage.ai/products/document-parse)
- [Upstage Document Parse Console Docs](https://console.upstage.ai/docs/capabilities/document-parse)
- [Upstage DP-Bench (Hugging Face)](https://huggingface.co/datasets/upstage/dp-bench)
- [Azure Document Intelligence Layout](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout)
- [Azure Document Intelligence What's New](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/whats-new)
- [Microsoft Document AI Research](https://www.microsoft.com/en-us/research/project/document-ai/)
- [OmniDocBench Paper (arXiv 2412.07626, CVPR 2025)](https://arxiv.org/abs/2412.07626)

---

## 부록: 크로스체크 결과 요약

> 2026-02-19 크로스체크 수행. 초안 대비 발견된 오류 및 수정 사항 기록.

### 수정된 오류

| 모델 | 항목 | 초안 내용 | 수정 내용 | 심각도 |
|------|------|-----------|-----------|--------|
| **MinerU** | 아키텍처 | DocLayout-YOLO + PDF-Extract-Kit 파이프라인 | v2.5+는 MinerU2.5-2509-1.2B VLM 기반 2단계로 전환 (초안은 v1.x 기준) | 높음 |
| **MinerU** | VLM 모델 | MinerU2.0-2505-0.9B | 최신은 MinerU2.5-2509-1.2B (0.9B→1.2B) | 높음 |
| **MinerU** | 추론 서빙 | SGLang | v2.5+에서 vLLM으로 전환 | 높음 |
| **DeepSeek-OCR 2** | 활성 파라미터 | ~570M | ~500M (공식 명칭: DeepSeek-3B-A500M) | 중간 |
| **Azure Layout** | 출력 포맷 | JSON/마크다운/검색가능 PDF | 검색가능 PDF는 Read 모델 전용, Layout은 JSON/마크다운만 | 중간 |
| **Azure Layout** | 바코드/수식 | 기본 내장 감지 요소로 기재 | 유료 add-on 기능 (features 파라미터 필요) | 중간 |
| **Azure Layout** | 내부 모델 | 확정적으로 기재 | MS 공식 확인 아닌 Research 기반 추정임을 명시 | 중간 |
| **Azure Layout** | ReadingBank | "모델"로 기재 | 모델이 아닌 "데이터셋"임을 수정 | 낮음 |
| **GLM-OCR** | CogViT 파라미터 | ~400M (확정) | ~400M (추정, 공식 미공개) | 낮음 |
| **GLM-OCR** | 라이선스 | MIT | MIT(모델) + Apache 2.0(코드) 이중 라이선스 | 낮음 |
| **PaddleOCR-VL** | v1.5 감지 방식 | instance segmentation | 마스크 기반 감지 + 사각형/다각형 출력 (엄밀히 다름) | 낮음 |
| **PaddleOCR-VL** | PP-OCRv5 구조 | 5단계 파이프라인 | 5개 모듈 중 3개는 선택적 전처리 | 낮음 |

### 확인된 정확한 사실

- GLM-OCR: OmniDocBench v1.5 94.62점 1위, 0.9B 파라미터, MTP Loss + RL 훈련 기법
- PaddleOCR-VL: OmniDocBench v1.5 94.5%, 111개 언어(v1.5), NaViT+ERNIE-4.5-0.3B 구조
- DeepSeek-OCR 2: Visual Causal Flow, SAM-base 토크나이저, Qwen2-0.5B 인코더, 91.09점, Apache 2.0
- Upstage: 12개 요소 타입, 3개 처리 모드, TEDS 93.48, ~0.6초/페이지
- Azure Layout: API v4.0 (2024-11-30), 2000페이지 제한, 6개 논리적 역할, 300+언어, 온프레미스(2025/04~)
