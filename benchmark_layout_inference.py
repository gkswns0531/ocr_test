"""Benchmark: Layout model inference FP32 vs BF16 + torch.compile + inference_mode.

Tests PP-DocLayoutV3 with different configurations on real GPU.
"""

import time
import gc
import torch
import numpy as np
from PIL import Image

# Suppress verbose logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("glmocr").setLevel(logging.ERROR)


def generate_test_images(n: int = 64, w: int = 2480, h: int = 3508, seed: int = 42) -> list[Image.Image]:
    """Generate random test images (simulating document pages)."""
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(n):
        # White background with random text-like blocks
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        for _ in range(20):
            x1, y1 = rng.randint(50, w - 200), rng.randint(50, h - 200)
            x2, y2 = x1 + rng.randint(100, 400), y1 + rng.randint(20, 60)
            arr[y1:y2, x1:x2] = rng.randint(0, 100, 3, dtype=np.uint8)
        images.append(Image.fromarray(arr))
    return images


def preprocess_images(images: list[Image.Image], target_h: int, target_w: int,
                      device: str, dtype: torch.dtype) -> torch.Tensor:
    """Preprocess images to model input tensor."""
    import cv2
    arrays = []
    for img in images:
        arr = np.asarray(img)
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        arrays.append(arr)
    batch = np.stack(arrays)
    pixel_values = (
        torch.from_numpy(batch)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device, dtype=dtype, non_blocking=True)
        .div_(255.0)
    )
    return pixel_values


def bench_config(
    label: str,
    model,
    pixel_values: torch.Tensor,
    context_manager,
    n_warmup: int = 2,
    n_runs: int = 5,
) -> float:
    """Benchmark a specific configuration. Returns median time in ms."""
    # Warmup
    for _ in range(n_warmup):
        with context_manager():
            _ = model(pixel_values=pixel_values)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with context_manager():
            _ = model(pixel_values=pixel_values)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    med = np.median(times)
    print(f"  {label:40s}: {med:>8.1f}ms  (runs: {[f'{t:.1f}' for t in times]})")
    return med


def bench_preprocess_resize(images: list[Image.Image], target_h: int, target_w: int,
                            device: str, n_runs: int = 5) -> None:
    """Benchmark per-image vs batch F.interpolate."""
    import cv2
    import torch.nn.functional as F

    print("\n" + "=" * 70)
    print("BENCHMARK: F.interpolate per-image vs batch (GPU resize)")
    print("=" * 70)

    # Decode all to GPU tensors first
    decoded = []
    for img in images:
        arr = np.asarray(img)
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).to(device)
        decoded.append(t)

    for dtype_label, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
        print(f"\n  [{dtype_label}]")

        # Per-image resize
        times_per = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tensors = []
            for d in decoded:
                r = F.interpolate(
                    d.unsqueeze(0).to(dtype=dtype), size=(target_h, target_w),
                    mode='bilinear', align_corners=False,
                )
                tensors.append(r.squeeze(0))
            pv = torch.stack(tensors).div_(255.0)
            torch.cuda.synchronize()
            times_per.append((time.perf_counter() - t0) * 1000)

        # Batch resize (all same size)
        times_batch = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = torch.stack(decoded).to(dtype=dtype)
            pv = F.interpolate(
                batch, size=(target_h, target_w),
                mode='bilinear', align_corners=False,
            ).div_(255.0)
            torch.cuda.synchronize()
            times_batch.append((time.perf_counter() - t0) * 1000)

        med_per = np.median(times_per)
        med_batch = np.median(times_batch)
        speedup = med_per / med_batch if med_batch > 0 else 0
        print(f"    per-image: {med_per:>8.1f}ms")
        print(f"    batch:     {med_batch:>8.1f}ms")
        print(f"    speedup:   {speedup:>5.1f}x")

        del pv
    del decoded
    torch.cuda.empty_cache()


def main() -> None:
    from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessorFast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"PyTorch: {torch.__version__}")

    # Load model
    print("\nLoading PP-DocLayoutV3...")
    model_name = "PaddlePaddle/PP-DocLayoutV3_safetensors"
    processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(model_name)
    model_fp32 = PPDocLayoutV3ForObjectDetection.from_pretrained(model_name)
    model_fp32.eval()

    param_count = sum(p.numel() for p in model_fp32.parameters()) / 1e6
    print(f"Params: {param_count:.1f}M")

    target_h = processor.size.get("height", 800)
    target_w = processor.size.get("width", 800)
    print(f"Target size: {target_h}x{target_w}")

    # Generate test data
    n_images = 64
    print(f"\nGenerating {n_images} test images (2480x3508)...")
    images = generate_test_images(n_images)

    # =========================================================================
    # 1. Model Inference Benchmark
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"BENCHMARK: Model Inference ({n_images} images)")
    print("=" * 70)

    results = {}

    # --- Config A: FP32 + no_grad (baseline) ---
    model_a = model_fp32.to(device)
    pv_fp32 = preprocess_images(images, target_h, target_w, device, torch.float32)
    results["FP32 + no_grad"] = bench_config(
        "FP32 + no_grad (baseline)", model_a, pv_fp32, torch.no_grad
    )

    # --- Config B: FP32 + inference_mode ---
    results["FP32 + inference_mode"] = bench_config(
        "FP32 + inference_mode", model_a, pv_fp32, torch.inference_mode
    )
    del pv_fp32
    model_a = model_a.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # --- Config C: BF16 + inference_mode ---
    model_bf16 = model_fp32.bfloat16().to(device)
    model_bf16.eval()
    pv_bf16 = preprocess_images(images, target_h, target_w, device, torch.bfloat16)
    results["BF16 + inference_mode"] = bench_config(
        "BF16 + inference_mode", model_bf16, pv_bf16, torch.inference_mode
    )

    # --- Config D: BF16 + inference_mode + torch.compile ---
    print("\n  Compiling model (first run includes compilation)...")
    model_compiled = torch.compile(model_bf16)
    # Extra warmup for compilation
    results["BF16 + inference_mode + compile"] = bench_config(
        "BF16 + inference_mode + compile", model_compiled, pv_bf16,
        torch.inference_mode, n_warmup=3, n_runs=5,
    )

    del pv_bf16, model_bf16, model_compiled
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    baseline = results["FP32 + no_grad"]
    for label, ms in results.items():
        speedup = baseline / ms if ms > 0 else 0
        tag = " (baseline)" if label == "FP32 + no_grad" else f" ({speedup:.2f}x)"
        print(f"  {label:40s}: {ms:>8.1f}ms{tag}")

    # =========================================================================
    # 2. Preprocess Resize Benchmark
    # =========================================================================
    bench_preprocess_resize(images, target_h, target_w, device)

    # =========================================================================
    # 3. GPU Memory
    # =========================================================================
    print("\n" + "=" * 70)
    print("GPU MEMORY")
    print("=" * 70)
    torch.cuda.empty_cache()
    gc.collect()

    model_fp32_gpu = model_fp32.to(device)
    torch.cuda.synchronize()
    mem_fp32 = torch.cuda.memory_allocated() / 1e6
    model_fp32_gpu = model_fp32_gpu.cpu()
    torch.cuda.empty_cache()

    model_bf16_gpu = model_fp32.bfloat16().to(device)
    torch.cuda.synchronize()
    mem_bf16 = torch.cuda.memory_allocated() / 1e6
    model_bf16_gpu = model_bf16_gpu.cpu()
    torch.cuda.empty_cache()

    print(f"  FP32 model: {mem_fp32:>8.1f}MB")
    print(f"  BF16 model: {mem_bf16:>8.1f}MB")
    print(f"  Savings:    {mem_fp32 - mem_bf16:>8.1f}MB ({(1 - mem_bf16/mem_fp32)*100:.0f}%)")


if __name__ == "__main__":
    main()
