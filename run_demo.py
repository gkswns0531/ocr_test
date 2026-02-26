"""Demo: GLM-OCR pipeline on vl_embedding samples."""
import os, json, gc, shutil, tempfile, traceback
from pathlib import Path
from PIL import Image
from datasets import Dataset

out_dir = Path('/home/ubuntu/ocr_test/pipeline_demo')
out_dir.mkdir(exist_ok=True)

# 1. Save images lazy
corpus_path = '/home/ubuntu/vl_embedding/data/SamsungSDS-Research___sds-ko_pub-vdr-benchmark/SDS-KoPub-corpus/0.0.0/759fcae092aef58436d125e72f74a2b53cdd5640'
arrow_files = sorted([f for f in os.listdir(corpus_path) if f.endswith('.arrow')])

for shard, idx, name in [(0, 300, 'sample1_table'), (25, 100, 'sample2_mixed')]:
    ds = Dataset.from_file(os.path.join(corpus_path, arrow_files[shard]))
    row = ds[idx]
    row['image'].convert('RGB').save(str(out_dir / f'{name}.jpg'), format='JPEG', quality=95)
    print(f'{name}: {row["file_name"][:60]}')
    del ds, row; gc.collect()

# 2. Init pipeline
from glmocr import GlmOcr
cfg_dir = Path(tempfile.mkdtemp())
cfg_path = cfg_dir / 'config.yaml'
shutil.copy('/home/ubuntu/glm-ocr-sdk/glmocr/config.yaml', cfg_path)

text = cfg_path.read_text()
text = text.replace('api_port: 8080', 'api_port: 8000')
text = text.replace('level: INFO', 'level: WARNING')
text = text.replace('max_workers: 32', 'max_workers: 1')
cfg_path.write_text(text)

print('\nInit pipeline...')
try:
    ocr = GlmOcr(config_path=str(cfg_path))
except Exception as e:
    traceback.print_exc()
    raise

# 3. Process
for name in ['sample1_table', 'sample2_mixed']:
    print(f'\n{"="*70}')
    print(f'Processing: {name}')
    print('='*70)

    try:
        result = ocr.parse(str(out_dir / f'{name}.jpg'), save_layout_visualization=True)
    except Exception as e:
        traceback.print_exc()
        print(f'FAILED: {e}')
        continue

    md = result.markdown_result or ''
    (out_dir / f'{name}_result.md').write_text(md, encoding='utf-8')

    if result.json_result:
        (out_dir / f'{name}_regions.json').write_text(
            json.dumps(result.json_result, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Markdown: {len(md)} chars')
    print('\n--- REGIONS ---')
    if isinstance(result.json_result, list):
        for r in result.json_result:
            lbl = r.get('label', '?'); tsk = r.get('task_type', '?')
            txt = (r.get('text', '') or '')[:100].replace('\n',' ')
            print(f'  [{lbl}/{tsk}] {txt}')
    print('\n--- MARKDOWN ---')
    print(md[:4000])
    if len(md) > 4000: print(f'... +{len(md)-4000} chars')
    print('--- END ---')
    gc.collect()

ocr.close()
shutil.rmtree(cfg_dir, ignore_errors=True)
print('\nDone!')
