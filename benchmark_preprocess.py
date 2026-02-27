#!/usr/bin/env python3
"""Benchmark: native optimized image preprocessing vs SDK approach.

Tests:
1. PIL resize (2480x3508 → 800x800) — PIL vs OpenCV
2. Tensor creation — processor vs direct numpy→torch
3. Region cropping — SDK's per-region np.asarray vs cached numpy arrays
"""
import time
import sys
sys.path.insert(0, '/root/glm-ocr-sdk')

import cv2
import numpy as np
import torch
from PIL import Image
from datasets import Dataset
from pathlib import Path

# ===== Load 64 real images =====
print("Loading 64 real images from dataset...")
arrow_dir = Path('data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640')
ds = Dataset.from_file(str(sorted(arrow_dir.glob('*.arrow'))[0]))
t0 = time.time()
pil_images = [ds[i]['image'].convert('RGB') for i in range(64)]
print(f"  Loaded: {time.time()-t0:.1f}s")
print(f"  Sizes: {pil_images[0].size} ... {pil_images[-1].size}")


# ===========================================================================
# Test 1: Image resize (2480x3508 → 800x800)
# ===========================================================================
print("\n" + "=" * 60)
print("TEST 1: Image resize to 800x800")
print("=" * 60)

# SDK approach: PIL.Image.resize with BILINEAR
t0 = time.time()
pil_resized = [img.resize((800, 800), Image.BILINEAR) for img in pil_images]
t_pil = time.time() - t0
print(f"  PIL.resize (BILINEAR):  {t_pil:.3f}s ({t_pil/64*1000:.0f}ms/img)")

# Optimized: cv2.resize
t0 = time.time()
cv2_resized = []
for img in pil_images:
    arr = np.asarray(img)
    resized = cv2.resize(arr, (800, 800), interpolation=cv2.INTER_LINEAR)
    cv2_resized.append(resized)
t_cv2 = time.time() - t0
print(f"  cv2.resize (LINEAR):    {t_cv2:.3f}s ({t_cv2/64*1000:.0f}ms/img) [{t_pil/t_cv2:.1f}x faster]")

# Optimized: cv2.resize + pre-convert to numpy
t0 = time.time()
np_arrays = [np.asarray(img) for img in pil_images]
t_convert = time.time() - t0
print(f"  np.asarray conversion:  {t_convert:.3f}s")

t0 = time.time()
cv2_resized2 = [cv2.resize(arr, (800, 800), interpolation=cv2.INTER_LINEAR) for arr in np_arrays]
t_cv2_only = time.time() - t0
print(f"  cv2.resize only:        {t_cv2_only:.3f}s ({t_cv2_only/64*1000:.0f}ms/img)")
print(f"  Total (convert+resize): {t_convert+t_cv2_only:.3f}s [{t_pil/(t_convert+t_cv2_only):.1f}x faster]")


# ===========================================================================
# Test 2: Tensor creation (800x800 images → GPU float32 tensor)
# ===========================================================================
print("\n" + "=" * 60)
print("TEST 2: Tensor creation (preprocessed 800x800 → GPU tensor)")
print("=" * 60)

device = 'cuda:0'

# SDK approach: transformers processor
from transformers import PPDocLayoutV3ImageProcessorFast
processor = PPDocLayoutV3ImageProcessorFast.from_pretrained('PaddlePaddle/PP-DocLayoutV3_safetensors')

pil_800 = [Image.fromarray(arr) for arr in cv2_resized2]  # 800x800 PIL

t0 = time.time()
proc_out = processor(images=pil_800, return_tensors="pt")
proc_tensor = proc_out["pixel_values"].to(device)
torch.cuda.synchronize()
t_proc = time.time() - t0
print(f"  Processor (800x800 PIL):  {t_proc:.3f}s")

# Optimized: numpy stack → torch (from PIL)
t0 = time.time()
arrays = [np.asarray(img) for img in pil_800]
batch_np = np.stack(arrays)
pixel_values = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(device, non_blocking=True).float().div_(255.0)
torch.cuda.synchronize()
t_direct_pil = time.time() - t0
print(f"  Direct (PIL→numpy→GPU):   {t_direct_pil:.3f}s [{t_proc/t_direct_pil:.1f}x faster]")

# Optimized: from cv2 numpy arrays directly (skip PIL)
t0 = time.time()
batch_np2 = np.stack(cv2_resized2)  # already numpy uint8 (800,800,3)
pixel_values2 = torch.from_numpy(batch_np2).permute(0, 3, 1, 2).contiguous().to(device, non_blocking=True).float().div_(255.0)
torch.cuda.synchronize()
t_direct_np = time.time() - t0
print(f"  Direct (numpy→GPU):       {t_direct_np:.3f}s [{t_proc/t_direct_np:.1f}x faster]")

# Verify correctness
diff = (proc_tensor - pixel_values).abs().max().item()
print(f"  Max diff (proc vs direct): {diff:.6f}")


# ===========================================================================
# Test 3: Region cropping
# ===========================================================================
print("\n" + "=" * 60)
print("TEST 3: Region cropping (simulate 495 crops from 64 images)")
print("=" * 60)

# Simulate detection results: ~8 regions per image with bbox + polygon
# Using realistic bbox sizes (10-40% of image)
rng = np.random.default_rng(42)
detections_per_image = []
for img_idx in range(64):
    n_regions = rng.integers(5, 12)
    regions = []
    for _ in range(n_regions):
        x1 = rng.integers(0, 600)
        y1 = rng.integers(0, 700)
        w = rng.integers(100, 400)
        h = rng.integers(50, 300)
        x2 = min(x1 + w, 999)
        y2 = min(y1 + h, 999)
        # Polygon: 4-point rectangle (normalized 0-1000)
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        regions.append({"bbox_2d": [x1, y1, x2, y2], "polygon": polygon})
    detections_per_image.append(regions)
total_crops = sum(len(r) for r in detections_per_image)
print(f"  Simulated: {total_crops} regions across 64 images")

# SDK approach: crop_image_region (calls np.asarray per crop)
from glmocr.utils.image_utils import crop_image_region

t0 = time.time()
sdk_crops = []
for img_idx, regions in enumerate(detections_per_image):
    for region in regions:
        crop = crop_image_region(pil_images[img_idx], region["bbox_2d"], region["polygon"])
        sdk_crops.append(crop)
t_sdk_crop = time.time() - t0
print(f"  SDK crop_image_region:    {t_sdk_crop:.3f}s ({t_sdk_crop/total_crops*1000:.1f}ms/crop)")

# Optimized: pre-convert images to numpy, then crop
t0 = time.time()
# Pre-convert all images to numpy once
np_images = [np.asarray(img) for img in pil_images]
t_preconv = time.time() - t0

t0 = time.time()
opt_crops = []
for img_idx, regions in enumerate(detections_per_image):
    img_array = np_images[img_idx]
    img_h, img_w = img_array.shape[:2]
    for region in regions:
        x1_n, y1_n, x2_n, y2_n = region["bbox_2d"]
        x1 = int(x1_n * img_w / 1000)
        y1 = int(y1_n * img_h / 1000)
        x2 = int(x2_n * img_w / 1000)
        y2 = int(y2_n * img_h / 1000)

        polygon = region["polygon"]
        img_crop = img_array[y1:y2, x1:x2]
        crop_h, crop_w = img_crop.shape[:2]
        if crop_h == 0 or crop_w == 0:
            opt_crops.append(Image.new('RGB', (1, 1)))
            continue

        if polygon and len(polygon) >= 3:
            scale_x = img_w / 1000
            scale_y = img_h / 1000
            polygon_pixels = np.array(
                [[int(p[0] * scale_x) - x1, int(p[1] * scale_y) - y1] for p in polygon],
                dtype=np.int32,
            )
            mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_pixels], 1)
            output = np.full_like(img_crop, 255, dtype=np.uint8)
            cv2.copyTo(img_crop, mask, output)
            opt_crops.append(Image.fromarray(output))
        else:
            opt_crops.append(Image.fromarray(img_crop.copy()))
t_opt_crop = time.time() - t0
print(f"  Optimized (cached numpy): {t_opt_crop:.3f}s ({t_opt_crop/total_crops*1000:.1f}ms/crop) [{t_sdk_crop/t_opt_crop:.1f}x faster]")
print(f"    Pre-convert overhead:   {t_preconv:.3f}s")
print(f"    Total (preconv+crop):   {t_preconv+t_opt_crop:.3f}s [{t_sdk_crop/(t_preconv+t_opt_crop):.1f}x faster]")


# ===========================================================================
# Test 4: Full pipeline comparison (resize + tensor + inference + postproc + crop)
# ===========================================================================
print("\n" + "=" * 60)
print("TEST 4: Full layout pipeline comparison")
print("=" * 60)

from transformers import PPDocLayoutV3ForObjectDetection
model = PPDocLayoutV3ForObjectDetection.from_pretrained('PaddlePaddle/PP-DocLayoutV3_safetensors')
model.eval().to(device)

# Warmup
with torch.no_grad():
    _ = model(pixel_values=pixel_values[:4])
torch.cuda.synchronize()

# --- SDK-like path (PIL resize + processor) ---
torch.cuda.empty_cache()
t0 = time.time()

# 1. PIL resize
_sdk_resized = [img.resize((800, 800), Image.BILINEAR) for img in pil_images]
t_sdk_resize = time.time() - t0

# 2. Processor
t1 = time.time()
_sdk_proc = processor(images=_sdk_resized, return_tensors="pt")
_sdk_tensor = _sdk_proc["pixel_values"].to(device)
torch.cuda.synchronize()
t_sdk_tensor = time.time() - t1

# 3. Inference
t2 = time.time()
with torch.no_grad():
    _sdk_outputs = model(pixel_values=_sdk_tensor)
torch.cuda.synchronize()
t_sdk_infer = time.time() - t2

# 4. Post-process
t3 = time.time()
target_sizes = torch.tensor([(h, w) for w, h in [img.size for img in pil_images]], device=device)
_sdk_results = processor.post_process_object_detection(_sdk_outputs, threshold=0.3, target_sizes=target_sizes)
t_sdk_postproc = time.time() - t3

t_sdk_total = time.time() - t0
print(f"  SDK-like path:")
print(f"    PIL resize:      {t_sdk_resize:.3f}s")
print(f"    Processor:       {t_sdk_tensor:.3f}s")
print(f"    Inference:       {t_sdk_infer:.3f}s")
print(f"    Post-process:    {t_sdk_postproc:.3f}s")
print(f"    TOTAL:           {t_sdk_total:.3f}s")

# --- Optimized path (cv2 resize + direct tensor) ---
torch.cuda.empty_cache()
t0 = time.time()

# 1. numpy conversion + cv2 resize
np_imgs = [np.asarray(img) for img in pil_images]
cv2_resized_final = [cv2.resize(arr, (800, 800), interpolation=cv2.INTER_LINEAR) for arr in np_imgs]
t_opt_resize = time.time() - t0

# 2. Direct tensor
t1 = time.time()
batch = np.stack(cv2_resized_final)
pv = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device, non_blocking=True).float().div_(255.0)
torch.cuda.synchronize()
t_opt_tensor = time.time() - t1

# 3. Inference
t2 = time.time()
with torch.no_grad():
    _opt_outputs = model(pixel_values=pv)
torch.cuda.synchronize()
t_opt_infer = time.time() - t2

# 4. Post-process (still use processor for this since it handles the model output format)
t3 = time.time()
_opt_results = processor.post_process_object_detection(_opt_outputs, threshold=0.3, target_sizes=target_sizes)
t_opt_postproc = time.time() - t3

t_opt_total = time.time() - t0
print(f"\n  Optimized path:")
print(f"    cv2 resize:      {t_opt_resize:.3f}s")
print(f"    Direct tensor:   {t_opt_tensor:.3f}s")
print(f"    Inference:       {t_opt_infer:.3f}s")
print(f"    Post-process:    {t_opt_postproc:.3f}s")
print(f"    TOTAL:           {t_opt_total:.3f}s")

# Verify results match
n_sdk = sum(len(r['scores']) for r in _sdk_results)
n_opt = sum(len(r['scores']) for r in _opt_results)
print(f"\n  Detections: SDK={n_sdk}, Optimized={n_opt}")
print(f"  Speedup: {t_sdk_total/t_opt_total:.1f}x")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("SUMMARY: Optimization potential for 64 images")
print("=" * 60)
print(f"  {'Component':<30} {'SDK':>8} {'Optimized':>10} {'Speedup':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
print(f"  {'PIL→cv2 resize':<30} {t_sdk_resize:>7.2f}s {t_opt_resize:>9.2f}s {t_sdk_resize/t_opt_resize:>7.1f}x")
print(f"  {'Processor→direct tensor':<30} {t_sdk_tensor:>7.2f}s {t_opt_tensor:>9.2f}s {t_sdk_tensor/t_opt_tensor:>7.1f}x")
print(f"  {'Model inference':<30} {t_sdk_infer:>7.2f}s {t_opt_infer:>9.2f}s {t_sdk_infer/t_opt_infer:>7.1f}x")
print(f"  {'Post-processing':<30} {t_sdk_postproc:>7.2f}s {t_opt_postproc:>9.2f}s {t_sdk_postproc/max(t_opt_postproc,0.001):>7.1f}x")
print(f"  {'Region cropping':<30} {t_sdk_crop:>7.2f}s {t_preconv+t_opt_crop:>9.2f}s {t_sdk_crop/(t_preconv+t_opt_crop):>7.1f}x")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
est_sdk = t_sdk_total + t_sdk_crop
est_opt = t_opt_total + t_preconv + t_opt_crop
print(f"  {'TOTAL (preproc+crop)':<30} {est_sdk:>7.2f}s {est_opt:>9.2f}s {est_sdk/est_opt:>7.1f}x")
