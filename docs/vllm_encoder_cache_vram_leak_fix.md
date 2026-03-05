# vLLM V1 Encoder Cache GPU VRAM Leak 분석 및 수정

> **관련 이슈**: [vllm-project/vllm#28230](https://github.com/vllm-project/vllm/issues/28230)
> **상태**: 수정 구현 완료, 기존 테스트 전체 통과

---

## 1. 문제 현상

vLLM V1 엔진에서 멀티모달(이미지) 추론 시 GPU VRAM이 요청마다 점진적으로 누수되어,
40-100개 이미지 처리 후 서버가 데드락된다.

---

## 2. Root Cause

`EncoderCacheManager`는 "lazy eviction" 모델을 사용한다:

1. 요청 완료 시 `free()` → encoder cache 엔트리를 `freeable` dict로 이동 (refcount 0)
2. `freeable` 엔트리는 **새 할당 시 공간이 부족할 때만** `can_allocate()`에서 퇴출되어 `freed` 리스트에 추가
3. `get_freed_mm_hashes()`는 `freed` 리스트만 반환
4. GPU worker는 `freed` 리스트에 있는 해시만 `encoder_cache.pop()` 호출

**문제**: sequential 워크로드나 저동시성 환경에서는 새 할당 압력이 없어
`can_allocate()`가 퇴출을 트리거하지 않음.
`freeable` 엔트리가 영원히 남아 GPU 텐서가 해제되지 않음.

```
요청 완료 → freeable[mm_hash] (refcount=0, GPU 텐서 유지)
                    ↓
새 할당 압력 없음 → can_allocate() 미호출 → freed[] 비어있음
                    ↓
get_freed_mm_hashes() → [] → GPU worker pop() 미실행 → VRAM 누수
```

---

## 3. 스케줄러 호출 순서

수정의 안전성을 검증하기 위해 확인한 `schedule()` 내부 호출 순서:

```
schedule() {
  1. 스케줄링 루프 (scheduler.py line 322-868)
     ├─ check_and_update_cache()  [line 1156] — freeable에서 복원 가능
     ├─ can_allocate()            [line 1177] — 공간 부족 시 freeable → freed 퇴출
     └─ allocate()               [line 517]  — 공간 예약

  2. SchedulerOutput 생성 (line 874-889)
     └─ get_freed_mm_hashes()    [line 888] ← 수정 지점: freeable 전체 flush

  3. _update_after_schedule()     [line 909]
     └─ _free_encoder_inputs()   [line 961] — free_encoder_input() → freeable 추가
}
```

핵심: `check_and_update_cache()`는 `get_freed_mm_hashes()` **이전**에 실행되므로,
같은 스텝 내에서 freeable 복원 후 flush되는 충돌은 발생하지 않는다.

---

## 4. 수정 내용

### 변경 파일

| 파일 | 변경 |
|------|------|
| `vllm/v1/core/encoder_cache_manager.py` | `get_freed_mm_hashes()` 수정 |
| `tests/v1/core/test_encoder_cache_manager.py` | 테스트 케이스 추가 |

### `get_freed_mm_hashes()` — 수정 전

```python
def get_freed_mm_hashes(self) -> list[str]:
    freed = self.freed
    self.freed = []
    return freed
```

### `get_freed_mm_hashes()` — 수정 후

```python
def get_freed_mm_hashes(self) -> list[str]:
    # Flush freeable entries that no running request references.
    for mm_hash, num_embeds in self.freeable.items():
        self.freed.append(mm_hash)
        del self.cached[mm_hash]
        self.num_free_slots += num_embeds
    self.freeable.clear()

    freed = self.freed
    self.freed = []
    return freed
```

freeable dict의 모든 엔트리를 매 호출마다 freed로 플러시하여,
할당 압력 없이도 GPU 메모리를 회수한다.

### 추가 테스트

```python
def test_freeable_flushed_on_get_freed_mm_hashes():
    """Freeable entries should be flushed to freed on get_freed_mm_hashes()
    even without allocation pressure (fixes GPU VRAM leak #28230)."""
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("r1", ["imgA"], [4])

    manager.allocate(req, 0)
    manager.free_encoder_input(req, 0)

    assert "imgA" in manager.freeable
    assert manager.num_free_slots == 6

    freed = manager.get_freed_mm_hashes()
    assert "imgA" in freed
    assert "imgA" not in manager.freeable
    assert "imgA" not in manager.cached
    assert manager.num_free_slots == 10
    assert manager.num_freeable_slots == 10
```

---

## 5. 안전성 검증

### 5.1 `check_and_update_cache()`와의 충돌 없음

`check_and_update_cache()`는 `freeable.pop(mm_hash)` (line 113)으로 엔트리를
freeable에서 **제거**하고 `cached[mm_hash]`에 request_id를 추가한다.
따라서 `get_freed_mm_hashes()` 실행 시 해당 엔트리는 이미 freeable에 없다.

### 5.2 dict 순회 중 mutation 없음

`self.freeable`을 순회하면서 `self.cached`를 수정한다.
서로 다른 dict이므로 안전하다. `self.freeable`은 루프 후 `clear()`.

### 5.3 `del self.cached[mm_hash]` KeyError 불가

`freeable`에 있는 엔트리는 반드시 `cached`에도 존재한다 (empty set).
`can_allocate()` 퇴출 경로도 양쪽을 함께 제거하므로,
루프에 도달하는 엔트리는 항상 `cached`에 있다.

### 5.4 `num_freeable_slots` 불변식 유지

불변식: `num_freeable_slots = num_free_slots + sum(freeable.values())`

- flush 전: `num_freeable_slots = num_free_slots + sum(freeable.values())`
- flush 후: `num_free_slots` += `sum(freeable.values())`, freeable 비워짐
- → `num_freeable_slots = num_free_slots (new)` ✓

`num_freeable_slots`를 건드리지 않는 것이 정확하다.

### 5.5 GPU worker 측 안전

`encoder_cache.pop(mm_hash, None)` (gpu_model_runner.py line 956)으로
이미 없는 해시도 안전하게 처리된다.

### 5.6 기존 테스트 전체 통과

```
16 passed in 4.05s
```

기존 15개 + 신규 1개 모두 통과.

---

## 6. 트레이드오프 분석

### 6.1 캐시 히트율 저하 (가장 큰 트레이드오프)

원래 코드에서는 freeable 엔트리가 `cached` dict에 남아있어,
다음 스텝에서 동일 이미지를 사용하는 새 요청이 cache hit을 받았다.

수정 후에는 매 스텝마다 flush되어 `cached`에서 삭제된다:

```
Step N: 요청 A 완료 → imgX가 freeable로 이동
        get_freed_mm_hashes() → imgX가 cached에서 삭제, GPU 텐서 해제

Step N+1: 요청 B가 동일한 imgX 사용
           check_and_update_cache() → False (cache miss)
           → encoder 재계산 필요
```

**워크로드별 영향**:

| 워크로드 | 영향 |
|----------|------|
| Sequential (매번 다른 이미지) | 없음 |
| 같은 스텝 내 동일 이미지 | 없음 — `check_and_update_cache()`가 flush 전 실행 |
| 다른 스텝 간 동일 이미지 반복 | **캐시 미스 증가** → encoder 재계산 |

### 6.2 Preemption 후 캐시 손실

```
Step N: 요청 A preempt → free(A) → freeable
        get_freed_mm_hashes() → flush → GPU 텐서 해제

Step N+1: 요청 A 재스케줄 → check_and_update_cache() → False
          → encoder 재계산 필요
```

원래 코드에서는 preempted 요청의 encoder output이 freeable에 남아
재스케줄 시 cache hit이 발생했다.
다만 preemption은 메모리 압력이 높을 때 발생하므로,
즉시 해제가 오히려 바람직할 수 있다.

### 6.3 `_free_encoder_inputs()` 1-step 지연

`_free_encoder_inputs()`는 `get_freed_mm_hashes()` **이후**에 호출된다 (line 961 vs 888).
여기서 freeable에 추가된 엔트리는 **다음 스텝**에서 flush된다.

원래 코드에서는 이 엔트리가 **무한히** 유지될 수 있었으므로 오히려 개선이다.

### 6.4 고동시성 환경

원래 코드는 `can_allocate()` 경로로 자연스럽게 퇴출되었다.
수정 후 매 스텝마다 추가 flush가 일어나 freeable 체류 시간이 줄어들고,
캐시 재활용 기회가 감소할 수 있다.

---

## 7. 요약

| 항목 | 평가 |
|------|------|
| Correctness 버그 | **없음** — 호출 순서, dict 안전성, 불변식 모두 검증 |
| Sequential VRAM 누수 | **해결** — 원래 문제 (#28230) |
| 동일 이미지 캐시 재활용 | **성능 저하** — 다른 스텝에서 같은 이미지 재사용 시 재계산 |
| Preemption 캐시 손실 | **경미한 성능 저하** — 재스케줄 시 encoder 재계산 |
| 고동시성 캐시 히트율 | **경미한 성능 저하** — freeable 체류 시간 감소 |
| `_free_encoder_inputs` 지연 | **수용 가능** — 1-step 지연, 원래보다 개선 |

**핵심 판단**: VRAM 누수로 인한 서버 데드락은 치명적 장애이고,
캐시 히트율 저하는 encoder 재계산 비용에 그친다.
동일 이미지 반복이 잦은 워크로드에서 성능이 중요하다면
freeable에 TTL이나 LRU 보존 로직을 추가하는 확장이 가능하다.

---

## 8. 대안 검토: 트레이드오프를 줄일 수 있는가?

### 8.1 근본 제약 — 트레이드오프를 완전히 없앨 수 없는 이유

아키텍처상 GPU 텐서의 생존과 캐시 히트는 동치이다:

```
get_freed_mm_hashes() → freed 반환 → GPU worker encoder_cache.pop()
                                       ↑ 텐서 삭제됨
                        ↓
                del self.cached[mm_hash]  ← 이걸 안 하면?
                        ↓
        check_and_update_cache() → True (cache hit!)
        하지만 GPU에 텐서가 없음 → correctness bug
```

`freed`로 보고하면 GPU worker가 텐서를 삭제하므로 `cached`에서도 반드시 제거해야 한다.
**"GPU 메모리 해제"와 "캐시 히트 보존"은 같은 엔트리에 대해 양립 불가능하다.**

따라서 질문은 "트레이드오프를 없앨 수 있나?"가 아니라 **"언제 해제할 것인가?"**이다.

### 8.2 대안 비교

#### 대안 A: Two-Generation (1-step grace period)

freeable을 "현재 세대"와 "이전 세대"로 분리. 이전 세대만 flush:

```python
def get_freed_mm_hashes(self) -> list[str]:
    # 이전 세대(1 스텝 생존)만 flush
    for mm_hash, num_embeds in self.stale_freeable.items():
        self.freed.append(mm_hash)
        del self.cached[mm_hash]
        self.num_free_slots += num_embeds

    # 현재 세대 → 이전 세대로 승격
    self.stale_freeable = self.freeable
    self.freeable = OrderedDict()

    freed = self.freed
    self.freed = []
    return freed
```

단, `check_and_update_cache()`와 `can_allocate()`도 수정 필요:

```python
# check_and_update_cache: 두 dict 모두 확인
if not self.cached[mm_hash]:
    if mm_hash in self.freeable:
        num_encoder_embeds = self.freeable.pop(mm_hash)
    else:
        num_encoder_embeds = self.stale_freeable.pop(mm_hash)
    self.num_freeable_slots -= num_encoder_embeds

# can_allocate: stale를 먼저 퇴출 (더 오래된 것부터)
while num_embeds > self.num_free_slots:
    if self.stale_freeable:
        mm_hash, num_free_embeds = self.stale_freeable.popitem(last=False)
    else:
        mm_hash, num_free_embeds = self.freeable.popitem(last=False)
    del self.cached[mm_hash]
    self.freed.append(mm_hash)
    self.num_free_slots += num_free_embeds
```

- 캐시 재활용 1 스텝 유예 (연속 스텝에서 같은 이미지 → 히트)
- VRAM 누수는 최대 1 스텝분으로 바운드
- **수정 범위: 4개 메서드 + `__init__` + `reset`**

#### 대안 B: Cap-based flush (임계치 초과 시만)

```python
def get_freed_mm_hashes(self) -> list[str]:
    freeable_total = sum(self.freeable.values())
    if freeable_total > self.cache_size // 2:
        for mm_hash, num_embeds in self.freeable.items():
            ...
        self.freeable.clear()
    ...
```

- 임계치 이하면 캐시 보존, 초과 시 전체 flush
- 임계치가 arbitrary (cache_size의 절반? 1/4?)
- 임계치 미만이면 여전히 누수 (느릴 뿐)

#### 대안 C: Oldest-first partial flush

```python
def get_freed_mm_hashes(self) -> list[str]:
    max_keep = 4
    while len(self.freeable) > max_keep:
        mm_hash, num_embeds = self.freeable.popitem(last=False)
        self.freed.append(mm_hash)
        del self.cached[mm_hash]
        self.num_free_slots += num_embeds
    ...
```

- 최근 엔트리는 보존, 오래된 것부터 제거
- magic number (max_keep) 필요

### 8.3 대안 평가표

| 방안 | VRAM 안전성 | 캐시 재활용 | 복잡도 | 수정 범위 |
|------|------------|------------|--------|----------|
| 현재 (전체 flush) | 완벽 | 없음 | 최소 | 1 메서드 |
| A: Two-Generation | 1-step 바운드 | 1 스텝 유예 | 중간 | 6곳 |
| B: Cap-based | 임계치 의존 | 임계치 미만 보존 | 낮음 | 1 메서드 |
| C: Oldest-first | 점진적 | 최근 N개 보존 | 낮음 | 1 메서드 |

### 8.4 결론: 현재 구현이 최선인 이유

1. **수정 범위 최소**: 1개 메서드만 변경. upstream PR reviewer가 검증하기 쉬움
2. **실제 영향 미미**: 동일 이미지가 다른 스텝에서 반복되는 워크로드는 드묾
   (sequential inference, batch 모두 해당 없음)
3. **Correctness 위험 0**: 다른 메서드를 건드리지 않음
4. **Preemption 케이스**: 메모리 압력이 높을 때 발생하므로 즉시 해제가 오히려 적합
5. **확장 가능**: 캐시 재활용이 실제로 문제가 되는 워크로드가 보고되면
   Two-Generation으로 확장하는 것이 합리적
