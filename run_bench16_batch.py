"""Benchmark 16 images - batch parse (layout batch_size=64, max_workers=64).
Warmup 3 images first, then measure 16 in a single batch call.
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
BATCH_SIZE = 8
OUT_DIR = Path('/home/ubuntu/ocr_test/bench16_batch_output')
CORPUS_PATH = '/home/ubuntu/vl_embedding/data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_gpu_mem():
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

# Warmup: shard 1
ds_warmup = Dataset.from_file(os.path.join(CORPUS_PATH, arrow_files[1]))
warmup_imgs = []
for idx in range(NUM_WARMUP):
    row = ds_warmup[idx]
    tmp_path = OUT_DIR / f'warmup_{idx}.jpg'
    row['image'].convert('RGB').save(str(tmp_path), format='JPEG', quality=95)
    warmup_imgs.append(str(tmp_path))
del ds_warmup; gc.collect()

# Benchmark: shard 0
ds = Dataset.from_file(os.path.join(CORPUS_PATH, arrow_files[0]))
shard_len = len(ds)
step = shard_len // NUM_SAMPLES
indices = [i * step for i in range(NUM_SAMPLES)]

print(f"Benchmark: {NUM_SAMPLES} images from shard 0 (step={step})")
print(f"Warmup: {NUM_WARMUP} images from shard 1")
print(f"max_workers={MAX_WORKERS}, layout batch_size={BATCH_SIZE}")

bench_imgs = []
for idx in indices:
    row = ds[idx]
    tmp_path = OUT_DIR / f'input_{idx:05d}.jpg'
    row['image'].convert('RGB').save(str(tmp_path), format='JPEG', quality=95)
    bench_imgs.append((idx, str(tmp_path), row['image'].size))
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
text = text.replace('batch_size: 1', f'batch_size: {BATCH_SIZE}')
cfg_path.write_text(text)

gpu_before = get_gpu_mem()
ram_before = get_ram_mb()
print(f"\n=== Before init === GPU: {gpu_before[0]}/{gpu_before[1]} MB | RAM: {ram_before:.0f} MB")

ocr = GlmOcr(config_path=str(cfg_path))

gpu_after = get_gpu_mem()
ram_after = get_ram_mb()
print(f"=== After init === GPU: {gpu_after[0]}/{gpu_after[1]} MB | RAM: {ram_after:.0f} MB")

# ---------------------------------------------------------------------------
# Warmup (1장씩, 측정 안 함)
# ---------------------------------------------------------------------------
print(f"\n=== Warmup ({NUM_WARMUP} images) ===")
for i, wp in enumerate(warmup_imgs):
    t0 = time.time()
    r = ocr.parse(wp, save_layout_visualization=False)
    flat = r.json_result[0] if (isinstance(r.json_result, list) and r.json_result and isinstance(r.json_result[0], list)) else (r.json_result or [])
    print(f"  warmup {i+1}: {len(flat)} regions, {time.time()-t0:.1f}s")
    gc.collect()
for wp in warmup_imgs:
    os.remove(wp)

gpu_warm = get_gpu_mem()
print(f"  GPU after warmup: {gpu_warm[0]} MB | RAM: {get_ram_mb():.0f} MB")

# ---------------------------------------------------------------------------
# Method 1: Sequential (1장씩) - 기준선
# ---------------------------------------------------------------------------
print(f"\n=== Method 1: Sequential (1 image at a time) ===")
seq_timings = []
t_seq_start = time.time()
for i, (idx, img_path, img_size) in enumerate(bench_imgs):
    t0 = time.time()
    result = ocr.parse(img_path, save_layout_visualization=False)
    t_el = time.time() - t0
    seq_timings.append(t_el)
    flat = result.json_result[0] if (isinstance(result.json_result, list) and result.json_result and isinstance(result.json_result[0], list)) else (result.json_result or [])
    print(f"  [{i+1:2d}/16] {len(flat):2d} regions | {t_el:.1f}s")
    gc.collect()
t_seq_total = time.time() - t_seq_start
seq_avg = sum(seq_timings) / len(seq_timings)
print(f"  Total: {t_seq_total:.1f}s | Avg: {seq_avg:.1f}s/img | Throughput: {NUM_SAMPLES/t_seq_total:.2f} img/s")

# ---------------------------------------------------------------------------
# Method 2: Batch (16장 한번에)
# ---------------------------------------------------------------------------
print(f"\n=== Method 2: Batch ({NUM_SAMPLES} images at once) ===")
img_paths = [p for _, p, _ in bench_imgs]

t_batch_start = time.time()
results = ocr.parse(img_paths, save_layout_visualization=True)
t_batch_total = time.time() - t_batch_start

batch_avg = t_batch_total / NUM_SAMPLES
total_regions = 0
for i, (result, (idx, _, img_size)) in enumerate(zip(results, bench_imgs)):
    flat = result.json_result[0] if (isinstance(result.json_result, list) and result.json_result and isinstance(result.json_result[0], list)) else (result.json_result or [])
    md = result.markdown_result or ''
    total_regions += len(flat)
    # save
    save_dir = OUT_DIR / f'result_{idx:05d}'
    save_dir.mkdir(exist_ok=True)
    result.save(str(save_dir))
    print(f"  [{i+1:2d}/16] {len(flat):2d} regions | {len(md):5d} chars")

gpu_final = get_gpu_mem()
ram_final = get_ram_mb()
print(f"  Total: {t_batch_total:.1f}s | Avg: {batch_avg:.1f}s/img | Throughput: {NUM_SAMPLES/t_batch_total:.2f} img/s")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ocr.close()
shutil.rmtree(cfg_dir, ignore_errors=True)

disk_mb = dir_size_mb(OUT_DIR)
input_disk_mb = sum(Path(p).stat().st_size for _, p, _ in bench_imgs) / 1024 / 1024
output_disk_mb = disk_mb - input_disk_mb

print(f"\n{'='*60}")
print(f"BENCHMARK COMPARISON (max_workers={MAX_WORKERS}, layout_batch={BATCH_SIZE})")
print(f"{'='*60}")
print(f"\n--- Sequential (1장씩 parse) ---")
print(f"  Total:      {t_seq_total:.1f}s")
print(f"  Average:    {seq_avg:.1f}s / image")
print(f"  Throughput: {NUM_SAMPLES/t_seq_total:.2f} img/s")
print(f"\n--- Batch ({NUM_SAMPLES}장 한번에 parse) ---")
print(f"  Total:      {t_batch_total:.1f}s")
print(f"  Average:    {batch_avg:.1f}s / image")
print(f"  Throughput: {NUM_SAMPLES/t_batch_total:.2f} img/s")
speedup = t_seq_total / t_batch_total if t_batch_total > 0 else 0
print(f"\n--- Speedup: {speedup:.2f}x ---")
print(f"\n--- Memory ---")
print(f"  GPU: {gpu_final[0]} / {gpu_final[1]} MB")
print(f"  RAM: {ram_final:.0f} MB")
print(f"\n--- Disk ---")
print(f"  Output: {output_disk_mb:.1f} MB ({output_disk_mb/NUM_SAMPLES*1000:.0f} KB/image)")

N = 40781
est_hours = (batch_avg * N) / 3600
est_disk_gb = (output_disk_mb / NUM_SAMPLES * N) / 1024
print(f"\n--- 40K Extrapolation (batch mode) ---")
print(f"  Estimated time: {est_hours:.1f} hours ({est_hours/24:.1f} days)")
print(f"  Estimated disk: {est_disk_gb:.1f} GB")
print(f"{'='*60}")

summary = {
    "max_workers": MAX_WORKERS,
    "layout_batch_size": BATCH_SIZE,
    "sequential": {"total_s": round(t_seq_total,1), "avg_s": round(seq_avg,1), "throughput": round(NUM_SAMPLES/t_seq_total,3)},
    "batch": {"total_s": round(t_batch_total,1), "avg_s": round(batch_avg,1), "throughput": round(NUM_SAMPLES/t_batch_total,3)},
    "speedup": round(speedup, 2),
    "gpu_mb": gpu_final[0], "ram_mb": round(ram_final),
    "output_disk_mb": round(output_disk_mb, 1),
    "extrapolation_40k": {"est_hours": round(est_hours,1), "est_days": round(est_hours/24,1), "est_disk_gb": round(est_disk_gb,1)},
}
(OUT_DIR / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f"Saved to {OUT_DIR / 'benchmark_summary.json'}")
