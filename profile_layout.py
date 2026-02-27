#!/usr/bin/env python3
"""Profile layout detection step by step to find the bottleneck."""
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Load one test image from dataset
print("=== Loading test image from dataset ===")
t0 = time.time()
from datasets import Dataset
arrow_dir = Path("data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640")
arrow_files = sorted(f for f in arrow_dir.iterdir() if f.suffix == ".arrow")
ds = Dataset.from_file(str(arrow_files[0]))
print(f"  Dataset loaded: {time.time()-t0:.1f}s, {len(ds)} rows")

# Get 16 images
t0 = time.time()
pil_images = [ds[i]["image"].convert("RGB") for i in range(16)]
print(f"  16 images loaded: {time.time()-t0:.1f}s")
for i, img in enumerate(pil_images[:3]):
    print(f"    Image {i}: size={img.size}, mode={img.mode}")

# Load model
print("\n=== Loading PP-DocLayoutV3 model ===")
from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessorFast

t0 = time.time()
processor = PPDocLayoutV3ImageProcessorFast.from_pretrained("PaddlePaddle/PP-DocLayoutV3_safetensors")
print(f"  Processor loaded: {time.time()-t0:.1f}s")

t0 = time.time()
model = PPDocLayoutV3ForObjectDetection.from_pretrained("PaddlePaddle/PP-DocLayoutV3_safetensors")
model.eval()
print(f"  Model loaded: {time.time()-t0:.1f}s")

# Model size
n_params = sum(p.numel() for p in model.parameters())
model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
print(f"  Parameters: {n_params/1e6:.1f}M, Size: {model_mb:.0f}MB")

t0 = time.time()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"  Model to {device}: {time.time()-t0:.1f}s")

# Check GPU memory after model load
if device.startswith("cuda"):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  GPU memory: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")

# === Profile each step ===
print("\n=== Profiling: Image Preprocessing (processor) ===")
for batch_size in [1, 4, 8, 16]:
    batch = pil_images[:batch_size]
    # Warmup
    _ = processor(images=batch[:1], return_tensors="pt")

    t0 = time.time()
    inputs = processor(images=batch, return_tensors="pt")
    t_preprocess = time.time() - t0

    shapes = {k: v.shape for k, v in inputs.items()}
    print(f"  batch={batch_size}: {t_preprocess:.3f}s ({t_preprocess/batch_size*1000:.0f}ms/img)")
    for k, v in inputs.items():
        mem_mb = v.numel() * v.element_size() / 1e6
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}, {mem_mb:.1f}MB")

print("\n=== Profiling: Data Transfer (to GPU) ===")
for batch_size in [1, 4, 8, 16]:
    batch = pil_images[:batch_size]
    inputs = processor(images=batch, return_tensors="pt")

    torch.cuda.synchronize()
    t0 = time.time()
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    torch.cuda.synchronize()
    t_transfer = time.time() - t0
    print(f"  batch={batch_size}: {t_transfer:.3f}s")
    del inputs_gpu

print("\n=== Profiling: Model Inference ===")
for batch_size in [1, 4, 8, 16]:
    batch = pil_images[:batch_size]
    inputs = processor(images=batch, return_tensors="pt")
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    if batch_size == 1:
        with torch.no_grad():
            _ = model(**inputs_gpu)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs_gpu)
    torch.cuda.synchronize()
    t_inference = time.time() - t0

    if device.startswith("cuda"):
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  batch={batch_size}: {t_inference:.3f}s ({t_inference/batch_size*1000:.0f}ms/img), GPU alloc={alloc:.2f}GB res={reserved:.2f}GB")
    del inputs_gpu, outputs
    torch.cuda.empty_cache()

print("\n=== Profiling: Post-processing ===")
for batch_size in [1, 4, 8, 16]:
    batch = pil_images[:batch_size]
    inputs = processor(images=batch, return_tensors="pt")
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs_gpu)

    target_sizes = torch.tensor([img.size[::-1] for img in batch], device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)
    torch.cuda.synchronize()
    t_postprocess = time.time() - t0

    n_detections = sum(len(r["scores"]) for r in results)
    print(f"  batch={batch_size}: {t_postprocess:.3f}s, {n_detections} detections total")
    del inputs_gpu, outputs, results
    torch.cuda.empty_cache()

print("\n=== Profiling: Full Pipeline (preprocess + transfer + inference + postprocess) ===")
for batch_size in [1, 4, 8, 16]:
    batch = pil_images[:batch_size]

    t0 = time.time()
    t_pre = time.time()
    inputs = processor(images=batch, return_tensors="pt")
    t_pre = time.time() - t_pre

    t_xfer = time.time()
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    torch.cuda.synchronize()
    t_xfer = time.time() - t_xfer

    t_inf = time.time()
    with torch.no_grad():
        outputs = model(**inputs_gpu)
    torch.cuda.synchronize()
    t_inf = time.time() - t_inf

    target_sizes = torch.tensor([img.size[::-1] for img in batch], device=device)
    t_post = time.time()
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)
    torch.cuda.synchronize()
    t_post = time.time() - t_post

    t_total = time.time() - t0
    print(f"  batch={batch_size}: total={t_total:.3f}s  preprocess={t_pre:.3f}s  transfer={t_xfer:.3f}s  inference={t_inf:.3f}s  postprocess={t_post:.3f}s")
    del inputs_gpu, outputs, results
    torch.cuda.empty_cache()

# Also check: does the SDK add overhead?
print("\n=== Profiling: SDK's process() method overhead ===")
print("  (Includes image array conversion, apply_layout_postprocess, etc.)")
from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess

batch = pil_images[:16]
# Reproduce what the SDK does
t0 = time.time()
image_batch = []
for image in batch:
    image_width, image_height = image.size
    image_array = np.array(image.convert("RGB"))
    image_batch.append((image_array, image_width, image_height))
pil_converted = [Image.fromarray(img[0]) for img in image_batch]
t_sdk_preconvert = time.time() - t0
print(f"  SDK image conversion (numpy roundtrip): {t_sdk_preconvert:.3f}s for 16 images")

print("\n=== Done ===")
