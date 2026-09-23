"""Verify all original ZIP bytes, official loader parity and saved crop pixels."""
import ast
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
out = root / 'reports/tn5000_seed20260923_final'
data = root / 'data/official/TN5000_forReview'
summary = json.loads((out / 'summary.json').read_text())
records = [json.loads(line) for line in (out / 'manifest.jsonl').read_text().splitlines()]
selected = json.loads((out / 'preview_manifest.json').read_text())
source = next((root / 'vendor/github').glob('*/mmdet/datasets/xml_style.py'))
tree = ast.parse(source.read_text())
function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_parse_instance_info')
namespace = {'ET': ET, 'List': list, 'dict': dict, 'ElementTree': ET.ElementTree}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), 'exec'), namespace)
stub = SimpleNamespace(_metainfo={'classes': ('0', '1')}, cat2label={'0': 0, '1': 1}, bbox_min_size=None, test_mode=True)
for r in records:
    parsed = namespace['_parse_instance_info'](stub, ET.parse(data / r['annotation']).getroot())
    assert [p['bbox'] for p in parsed] == [o['loader_xyxy'] for o in r['objects']], r['sample_id']
    assert [p['bbox_label'] for p in parsed] == [int(o['raw_label']) for o in r['objects']], r['sample_id']
eligible = sorted(r['sample_id'] for r in records if r['structurally_valid'] and len(r['objects']) == 1)
assert random.Random(summary['seed']).sample(eligible, 20) == summary['selected_ids']
for r in selected:
    stem = f'{r["order"]:02d}_{r["sample_id"]}'
    with Image.open(data / r['image']) as im: original = np.asarray(im.convert('RGB'))
    with Image.open(out / 'previews' / (stem + '_full.png')) as im: assert np.array_equal(original, np.asarray(im))
    x1,y1,x2,y2 = r['objects'][0]['crop_xyxy_half_open']
    with Image.open(out / r['roi_file']) as im: assert np.array_equal(original[y1:y2, x1:x2], np.asarray(im)), stem
    assert hashlib.sha256(original[y1:y2, x1:x2].tobytes()).hexdigest() == r['roi_pixel_sha256']
checked = 0
with zipfile.ZipFile(root / 'downloads/TN5000_forReview.zip') as z:
    for entry in z.infolist():
        if entry.is_dir(): continue
        target = root / 'data/official' / entry.filename
        assert hashlib.sha256(z.read(entry)).digest() == hashlib.sha256(target.read_bytes()).digest(), entry.filename
        checked += 1
result = {'status': 'passed', 'official_archive_files_byte_verified': checked,
          'official_loader_boxes_and_labels_verified': len(records),
          'seed_selection_verified': True, 'full_and_roi_pixel_equality_verified': len(selected),
          'source_loader': str(source), 'source_loader_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'note': 'Executes only the official XML parsing method via AST, not the MMDetection runtime or training.'}
(out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
