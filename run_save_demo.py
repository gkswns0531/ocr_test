"""Test result.save() to see image crop behavior."""
import os, gc, shutil, tempfile
from pathlib import Path
from datasets import Dataset
from glmocr import GlmOcr

# Pick samples with image regions: 03_image_text, 04_image_table_multi, 09_chart_focus
SAMPLES = [
    (0, 204, "03_image_text", "장서개발정책 - 이미지+텍스트"),
    (7, 0, "04_image_table_multi", "습지보호 - 이미지+테이블 다수"),
    (0, 153, "09_chart_focus", "장서개발정책 - 차트 중심"),
]

out_dir = Path('/home/ubuntu/ocr_test/pipeline_save_demo')
out_dir.mkdir(exist_ok=True)

corpus_path = '/home/ubuntu/vl_embedding/data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640'
arrow_files = sorted([f for f in os.listdir(corpus_path) if f.endswith('.arrow')])

# Save images first
print("=== Saving original images ===")
shard_cache = {}
for shard, idx, name, desc in SAMPLES:
    if shard not in shard_cache:
        shard_cache[shard] = Dataset.from_file(os.path.join(corpus_path, arrow_files[shard]))
    ds = shard_cache[shard]
    row = ds[idx]
    img_path = out_dir / f'{name}.jpg'
    row['image'].convert('RGB').save(str(img_path), format='JPEG', quality=95)
    print(f'  {name}: {row["image"].size}')
del shard_cache; gc.collect()

# Init pipeline
cfg_dir = Path(tempfile.mkdtemp())
cfg_path = cfg_dir / 'config.yaml'
shutil.copy('/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml', cfg_path)
text = cfg_path.read_text()
text = text.replace('api_port: 8080', 'api_port: 8000')
text = text.replace('level: INFO', 'level: WARNING')
text = text.replace('max_workers: 32', 'max_workers: 1')
cfg_path.write_text(text)

print("\n=== Init pipeline ===")
ocr = GlmOcr(config_path=str(cfg_path))

# Process each and use result.save()
print("\n=== Processing with result.save() ===")
for shard, idx, name, desc in SAMPLES:
    img_path = str(out_dir / f'{name}.jpg')
    print(f'\n--- {name} ({desc}) ---')

    result = ocr.parse(img_path, save_layout_visualization=True)

    # Use result.save() - this should trigger crop_and_replace_images
    save_dir = out_dir / name
    save_dir.mkdir(exist_ok=True)
    result.save(save_dir)

    print(f'  Saved to: {save_dir}')
    # List what was saved
    for f in sorted(save_dir.rglob('*')):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f'  {f.relative_to(save_dir)} ({size_kb:.1f} KB)')

    gc.collect()

ocr.close()
shutil.rmtree(cfg_dir, ignore_errors=True)
print('\n=== Done! ===')
