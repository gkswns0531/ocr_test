"""Benchmark: build_request_from_image — old (double encode) vs new (single pass).

Measures JPEG+base64 encoding overhead for cropped image regions.
"""

import time
import base64
import io
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Suppress verbose logging
import logging
logging.getLogger("glmocr").setLevel(logging.ERROR)


# ─── Old implementation (double encode) ────────────────────────────────────────
def _old_build_request_from_image(
    image: Image.Image,
    image_format: str,
    t_patch_size: int,
    max_pixels: int,
    patch_expand_factor: int,
    min_pixels: int,
) -> dict:
    """Original: JPEG encode → base64 → data URL → decode → resize → re-encode."""
    from glmocr.utils.image_utils import load_image_to_base64

    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1st encode (wasted)
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    data_url = f"data:image/{image_format.lower()};base64,{img_base64}"

    # _process_msg_standard path: re-encodes via load_image_to_base64
    encoded_image = load_image_to_base64(
        data_url,
        t_patch_size=t_patch_size,
        max_pixels=max_pixels,
        image_format=image_format,
        patch_expand_factor=patch_expand_factor,
        min_pixels=min_pixels,
    )
    return {
        "url": f"data:image/{image_format.lower()};base64,{encoded_image}",
    }


# ─── New implementation (single pass) ──────────────────────────────────────────
def _new_build_request_from_image(
    image: Image.Image,
    image_format: str,
    t_patch_size: int,
    max_pixels: int,
    patch_expand_factor: int,
    min_pixels: int,
) -> dict:
    """Optimized: PIL image → smart_resize → JPEG → base64 (single pass)."""
    from glmocr.utils.image_utils import load_image_to_base64

    if image.mode != "RGB":
        image = image.convert("RGB")

    encoded_image = load_image_to_base64(
        image,
        t_patch_size=t_patch_size,
        max_pixels=max_pixels,
        image_format=image_format,
        patch_expand_factor=patch_expand_factor,
        min_pixels=min_pixels,
    )
    return {
        "url": f"data:image/{image_format.lower()};base64,{encoded_image}",
    }


def generate_crop_images(n: int = 200, seed: int = 42) -> list[Image.Image]:
    """Generate random crop images simulating layout regions."""
    rng = np.random.RandomState(seed)
    crops = []
    for _ in range(n):
        w = rng.randint(100, 1500)
        h = rng.randint(30, 800)
        arr = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
        crops.append(Image.fromarray(arr))
    return crops


def bench(label: str, func, crops: list[Image.Image], n_runs: int = 5) -> float:
    """Run benchmark, return median ms."""
    params = dict(
        image_format="JPEG",
        t_patch_size=2,
        max_pixels=14 * 14 * 4 * 1280,
        patch_expand_factor=1,
        min_pixels=112 * 112,
    )

    # Warmup
    for crop in crops[:5]:
        func(crop, **params)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for crop in crops:
            func(crop, **params)
        times.append((time.perf_counter() - t0) * 1000)

    med = np.median(times)
    print(f"  {label:45s}: {med:>8.1f}ms  (runs: {[f'{t:.1f}' for t in times]})")
    return med


def bench_parallel(label: str, func, crops: list[Image.Image],
                   max_workers: int = 16, n_runs: int = 5) -> float:
    """Benchmark parallel execution."""
    params = dict(
        image_format="JPEG",
        t_patch_size=2,
        max_pixels=14 * 14 * 4 * 1280,
        patch_expand_factor=1,
        min_pixels=112 * 112,
    )

    # Warmup
    for crop in crops[:5]:
        func(crop, **params)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, crop, **params) for crop in crops]
            for f in futures:
                f.result()
        times.append((time.perf_counter() - t0) * 1000)

    med = np.median(times)
    print(f"  {label:45s}: {med:>8.1f}ms  (runs: {[f'{t:.1f}' for t in times]})")
    return med


def bench_correctness(crops: list[Image.Image]) -> None:
    """Verify old and new produce identical output."""
    params = dict(
        image_format="JPEG",
        t_patch_size=2,
        max_pixels=14 * 14 * 4 * 1280,
        patch_expand_factor=1,
        min_pixels=112 * 112,
    )
    match = 0
    for crop in crops[:50]:
        old = _old_build_request_from_image(crop, **params)
        new = _new_build_request_from_image(crop, **params)
        if old["url"] == new["url"]:
            match += 1
    print(f"\n  Correctness: {match}/50 identical outputs")


def main() -> None:
    n_crops = 200
    print(f"Generating {n_crops} random crop images...")
    crops = generate_crop_images(n_crops)

    sizes = [c.size for c in crops]
    avg_w = np.mean([s[0] for s in sizes])
    avg_h = np.mean([s[1] for s in sizes])
    print(f"Average crop size: {avg_w:.0f}x{avg_h:.0f}")

    # ── Correctness check ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CORRECTNESS CHECK")
    print("=" * 70)
    bench_correctness(crops)

    # ── Serial benchmark ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"BENCHMARK: Serial build_request ({n_crops} crops)")
    print("=" * 70)

    t_old = bench("Old (double encode, serial)", _old_build_request_from_image, crops)
    t_new = bench("New (single pass, serial)", _new_build_request_from_image, crops)
    print(f"\n  Speedup (single pass): {t_old / t_new:.2f}x")

    # ── Parallel benchmark ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"BENCHMARK: Parallel build_request ({n_crops} crops, 16 workers)")
    print("=" * 70)

    t_old_par = bench_parallel("Old (double encode, 16 workers)", _old_build_request_from_image, crops)
    t_new_par = bench_parallel("New (single pass, 16 workers)", _new_build_request_from_image, crops)
    print(f"\n  Speedup (parallel vs serial old): {t_old / t_new_par:.2f}x")
    print(f"  Speedup (parallel new vs serial old): {t_old / t_new_par:.2f}x")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Old serial (baseline):    {t_old:>8.1f}ms")
    print(f"  New serial:               {t_new:>8.1f}ms  ({t_old/t_new:.2f}x)")
    print(f"  Old parallel (16w):       {t_old_par:>8.1f}ms  ({t_old/t_old_par:.2f}x)")
    print(f"  New parallel (16w):       {t_new_par:>8.1f}ms  ({t_old/t_new_par:.2f}x)")


if __name__ == "__main__":
    main()
