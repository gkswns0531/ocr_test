"""Benchmark 16 images through GLM-OCR pipeline.
Measures: latency per image, GPU/RAM usage, disk output size.
Warmup 3 images (different from benchmark set) before measuring.
"""
import os, gc, time, json, shutil, tempfile, psutil
from pathlib import Path
from datasets import Dataset
from glmocr import GlmOcr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_WARMUP = 3
NUM_SAMPLES = 16
MAX_WORKERS = 64
OUT_DIR = Path('/home/ubuntu/ocr_test/bench16_output')
CORPUS_PATH = '/home/ubuntu/vl_embedding/data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gpu_mem():
    """Return (used_MB, total_MB) from nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            text=True)
        used, total = out.strip().split(', ')
        return int(used), int(total)
    except Exception:
        return 0, 0

def get_ram_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024

def dir_size_mb(path):
    total = 0
    for f in Path(path).rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / 1024 / 1024

# ---------------------------------------------------------------------------
# Prepare images
# ---------------------------------------------------------------------------
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

arrow_files = sorted([f for f in os.listdir(CORPUS_PATH) if f.endswith('.arrow')])

# Warmup: shard 1의 처음 3개 (벤치마크와 다른 이미지)
ds_warmup = Dataset.from_file(os.path.join(CORPUS_PATH, arrow_files[1]))
warmup_imgs = []
for idx in range(NUM_WARMUP):
    row = ds_warmup[idx]
    tmp_path = OUT_DIR / f'warmup_{idx}.jpg'
    row['image'].convert('RGB').save(str(tmp_path), format='JPEG', quality=95)
    warmup_imgs.append(str(tmp_path))
del ds_warmup; gc.collect()

# Benchmark: shard 0에서 16개 균등 샘플
ds = Dataset.from_file(os.path.join(CORPUS_PATH, arrow_files[0]))
shard_len = len(ds)
step = shard_len // NUM_SAMPLES
indices = [i * step for i in range(NUM_SAMPLES)]

print(f"Benchmark: shard 0, {shard_len} rows, {NUM_SAMPLES} samples at step={step}")
print(f"Warmup: shard 1, {NUM_WARMUP} images (different from benchmark)")
print(f"max_workers: {MAX_WORKERS}")

tmp_imgs = []
for idx in indices:
    row = ds[idx]
    tmp_path = OUT_DIR / f'input_{idx:05d}.jpg'
    row['image'].convert('RGB').save(str(tmp_path), format='JPEG', quality=95)
    tmp_imgs.append((idx, str(tmp_path), row['image'].size))
del ds; gc.collect()

# ---------------------------------------------------------------------------
# Init pipeline
# ---------------------------------------------------------------------------
cfg_dir = Path(tempfile.mkdtemp())
cfg_path = cfg_dir / 'config.yaml'
shutil.copy('/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml', cfg_path)
text = cfg_path.read_text()
text = text.replace('api_port: 8080', 'api_port: 8000')
text = text.replace('level: INFO', 'level: WARNING')
text = text.replace('max_workers: 32', f'max_workers: {MAX_WORKERS}')
text = text.replace('batch_size: 1', 'batch_size: 64')
cfg_path.write_text(text)

gpu_before_init = get_gpu_mem()
ram_before_init = get_ram_mb()
print(f"\n=== Before pipeline init ===")
print(f"GPU: {gpu_before_init[0]} / {gpu_before_init[1]} MB")
print(f"RAM: {ram_before_init:.0f} MB")

t_init_start = time.time()
ocr = GlmOcr(config_path=str(cfg_path))
t_init = time.time() - t_init_start

gpu_after_init = get_gpu_mem()
ram_after_init = get_ram_mb()
print(f"\n=== After pipeline init ({t_init:.1f}s) ===")
print(f"GPU: {gpu_after_init[0]} / {gpu_after_init[1]} MB (+{gpu_after_init[0] - gpu_before_init[0]} MB)")
print(f"RAM: {ram_after_init:.0f} MB (+{ram_after_init - ram_before_init:.0f} MB)")

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
print(f"\n=== Warmup ({NUM_WARMUP} images from shard 1) ===")
for i, wp in enumerate(warmup_imgs):
    t0 = time.time()
    result = ocr.parse(wp, save_layout_visualization=False)
    regions = result.json_result
    flat = regions[0] if (isinstance(regions, list) and regions and isinstance(regions[0], list)) else (regions or [])
    print(f"  warmup {i+1}: {len(flat)} regions, {time.time()-t0:.1f}s")
    gc.collect()

# Clean warmup files
for wp in warmup_imgs:
    os.remove(wp)

gpu_after_warmup = get_gpu_mem()
ram_after_warmup = get_ram_mb()
print(f"  GPU after warmup: {gpu_after_warmup[0]} MB | RAM: {ram_after_warmup:.0f} MB")

# ---------------------------------------------------------------------------
# Benchmark 16 images
# ---------------------------------------------------------------------------
print(f"\n=== Benchmark ({NUM_SAMPLES} images, max_workers={MAX_WORKERS}) ===")
timings = []
region_counts = []
md_lengths = []

t_total_start = time.time()

for i, (idx, img_path, img_size) in enumerate(tmp_imgs):
    t_start = time.time()

    result = ocr.parse(img_path, save_layout_visualization=True)

    # save() with image crop
    save_dir = OUT_DIR / f'result_{idx:05d}'
    save_dir.mkdir(exist_ok=True)
    result.save(str(save_dir))

    t_elapsed = time.time() - t_start
    timings.append(t_elapsed)

    # Stats
    md = result.markdown_result or ''
    md_lengths.append(len(md))
    regions = result.json_result
    flat = regions[0] if (isinstance(regions, list) and regions and isinstance(regions[0], list)) else (regions or [])
    region_counts.append(len(flat))

    gpu_now = get_gpu_mem()
    ram_now = get_ram_mb()
    print(f"  [{i+1:2d}/16] idx={idx:5d} | {img_size[0]}x{img_size[1]} | "
          f"{len(flat):2d} regions | {len(md):5d} chars | "
          f"{t_elapsed:.1f}s | GPU {gpu_now[0]}MB | RAM {ram_now:.0f}MB")

    gc.collect()

t_total = time.time() - t_total_start

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ocr.close()
shutil.rmtree(cfg_dir, ignore_errors=True)

gpu_final = get_gpu_mem()
ram_final = get_ram_mb()
disk_mb = dir_size_mb(OUT_DIR)
input_disk_mb = sum(Path(p).stat().st_size for _, p, _ in tmp_imgs) / 1024 / 1024
output_disk_mb = disk_mb - input_disk_mb

avg_time = sum(timings) / len(timings)
min_time = min(timings)
max_time = max(timings)
avg_regions = sum(region_counts) / len(region_counts)

print(f"\n{'='*60}")
print(f"BENCHMARK RESULTS ({NUM_SAMPLES} images, max_workers={MAX_WORKERS})")
print(f"{'='*60}")
print(f"\n--- Latency ---")
print(f"  Total:     {t_total:.1f}s")
print(f"  Average:   {avg_time:.1f}s / image")
print(f"  Min:       {min_time:.1f}s")
print(f"  Max:       {max_time:.1f}s")
print(f"  Throughput: {NUM_SAMPLES / t_total:.2f} images/s")
print(f"\n--- Regions ---")
print(f"  Average:   {avg_regions:.1f} regions / image")
print(f"  Total:     {sum(region_counts)} regions")
print(f"\n--- Memory ---")
print(f"  GPU (after warmup): {gpu_after_warmup[0]} MB / {gpu_after_warmup[1]} MB")
print(f"  RAM (peak): {ram_final:.0f} MB")
print(f"\n--- Disk ---")
print(f"  Input images:  {input_disk_mb:.1f} MB ({input_disk_mb/NUM_SAMPLES*1000:.0f} KB/image)")
print(f"  Output total:  {output_disk_mb:.1f} MB ({output_disk_mb/NUM_SAMPLES*1000:.0f} KB/image)")
print(f"  Combined:      {disk_mb:.1f} MB")

print(f"\n--- 40K Extrapolation ---")
N = 40781
est_hours = (avg_time * N) / 3600
est_disk_gb = (output_disk_mb / NUM_SAMPLES * N) / 1024
print(f"  Estimated time:  {est_hours:.1f} hours ({est_hours/24:.1f} days)")
print(f"  Estimated disk:  {est_disk_gb:.1f} GB")
print(f"  (sequential, single L4 GPU, max_workers={MAX_WORKERS})")

print(f"\n{'='*60}")

# Save summary JSON
summary = {
    "num_samples": NUM_SAMPLES,
    "max_workers": MAX_WORKERS,
    "warmup_count": NUM_WARMUP,
    "total_time_s": round(t_total, 1),
    "avg_time_s": round(avg_time, 1),
    "min_time_s": round(min_time, 1),
    "max_time_s": round(max_time, 1),
    "throughput_ips": round(NUM_SAMPLES / t_total, 3),
    "avg_regions": round(avg_regions, 1),
    "gpu_peak_mb": gpu_after_warmup[0],
    "gpu_total_mb": gpu_after_warmup[1],
    "ram_peak_mb": round(ram_final),
    "input_disk_mb": round(input_disk_mb, 1),
    "output_disk_mb": round(output_disk_mb, 1),
    "extrapolation_40k": {
        "est_hours": round(est_hours, 1),
        "est_days": round(est_hours / 24, 1),
        "est_disk_gb": round(est_disk_gb, 1),
    },
    "per_image": [
        {"idx": idx, "time_s": round(t, 1), "regions": rc, "md_chars": ml}
        for (idx, _, _), t, rc, ml in zip(tmp_imgs, timings, region_counts, md_lengths)
    ]
}
(OUT_DIR / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f"Summary saved to {OUT_DIR / 'benchmark_summary.json'}")
