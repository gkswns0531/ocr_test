"""Generate 10 diverse pipeline demo samples."""
import os, json, gc, shutil, tempfile, traceback
from pathlib import Path
from PIL import Image
from datasets import Dataset

# 10 diverse samples: (shard, idx, name, description)
SAMPLES = [
    # 1. 차트 + 테이블 혼합
    (0, 357, "01_chart_table", "재정보고서 - 차트+테이블"),
    # 2. 차트 + 텍스트 + 테이블 복합
    (0, 459, "02_chart_text_table", "재정보고서 - 차트+텍스트+테이블 복합"),
    # 3. 이미지 포함 페이지
    (0, 204, "03_image_text", "장서개발정책 - 이미지+텍스트"),
    # 4. 이미지+테이블 복합
    (7, 0, "04_image_table_multi", "습지보호 - 이미지+테이블 다수"),
    # 5. 테이블 2개
    (0, 867, "05_multi_table", "지식재산 보호정책 - 테이블 2개"),
    # 6. 테이블+텍스트 혼합 (많은 영역)
    (0, 408, "06_table_text_rich", "재정보고서 - 테이블+텍스트 풍부"),
    # 7. 텍스트 위주 (많은 영역)
    (0, 0, "07_text_heavy", "CSIS 정책간담회 - 텍스트 25영역"),
    # 8. 이미지+테이블+텍스트
    (0, 663, "08_mixed_all", "지하수이용 - 이미지+테이블+텍스트"),
    # 9. 차트 중심
    (0, 153, "09_chart_focus", "장서개발정책 - 차트 중심"),
    # 10. 표지/제목 페이지
    (22, 510, "10_title_page", "에너지 안보 - 표지"),
]

out_dir = Path('/home/ubuntu/ocr_test/pipeline_demo')
out_dir.mkdir(exist_ok=True)

corpus_path = '/home/ubuntu/vl_embedding/data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640'
arrow_files = sorted([f for f in os.listdir(corpus_path) if f.endswith('.arrow')])

# Save all images first (lazy, one shard at a time)
print("=== Saving images ===")
shard_cache = {}
for shard, idx, name, desc in SAMPLES:
    if shard not in shard_cache:
        shard_cache[shard] = Dataset.from_file(os.path.join(corpus_path, arrow_files[shard]))
    ds = shard_cache[shard]
    row = ds[idx]
    img_path = out_dir / f'{name}_original.jpg'
    row['image'].convert('RGB').save(str(img_path), format='JPEG', quality=95)
    print(f'  {name}: {row["file_name"][:50]} ({row["image"].size})')
del shard_cache; gc.collect()

# Init pipeline
from glmocr import GlmOcr
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

# Process each
print("\n=== Processing ===")
for i, (shard, idx, name, desc) in enumerate(SAMPLES, 1):
    img_path = str(out_dir / f'{name}_original.jpg')
    print(f'\n[{i}/10] {name} ({desc})')

    try:
        result = ocr.parse(img_path, save_layout_visualization=True)
        md = result.markdown_result or ''

        # Save markdown
        (out_dir / f'{name}_result.md').write_text(md, encoding='utf-8')

        # Save regions JSON
        regions = result.json_result
        if regions:
            (out_dir / f'{name}_regions.json').write_text(
                json.dumps(regions, ensure_ascii=False, indent=2), encoding='utf-8')

        # Flatten nested list
        flat_regions = regions[0] if (isinstance(regions, list) and regions and isinstance(regions[0], list)) else (regions or [])

        # Print summary
        from collections import Counter
        labels = Counter(r.get('native_label', r.get('label', '?')) for r in flat_regions)
        print(f'  Regions: {len(flat_regions)} | Labels: {dict(labels)}')
        print(f'  Markdown: {len(md)} chars')

    except Exception as e:
        traceback.print_exc()
        print(f'  FAILED: {e}')

    gc.collect()

ocr.close()
shutil.rmtree(cfg_dir, ignore_errors=True)
print('\n=== Done! ===')
