"""Audit TN5000 VOC files without changing source data; no model imports.

Labels: official paper Data Records, doi:10.1038/s41597-025-05757-4.
Boxes: pinned official mmdet/datasets/xml_style.py _parse_instance_info:
int(float(value)) - 1 for ALL four coordinates. Nonintegers are reported,
never silently rounded. Preview crop uses these coordinates as Pillow's
half-open rectangle; this explicit preview policy preserves loader widths.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps

LABELS = {'0': 'benign', '1': 'malignant'}
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--preview-count', type=int, default=20)
    args = parser.parse_args()
    data, out = args.data_root.resolve(), args.output.resolve()
    if out.is_relative_to(data) or data.is_relative_to(out):
        parser.error('Output and source data directories must be disjoint')
    if out.exists():
        parser.error('Output already exists; use a new output directory')
    out.mkdir(parents=True)
    issues, records, tags = [], [], Counter()
    def issue(kind, sample, detail):
        issues.append({'kind': kind, 'sample_id': sample, 'detail': detail})
    for sub in ['JPEGImages', 'Annotations', 'ImageSets/Main']:
        if not (data / sub).is_dir():
            issue('missing_directory', '', str(data / sub))
    if issues:
        write_json(out / 'issues.json', issues)
        print('MISSING DATA; see', out / 'issues.json')
        return 2
    source_paths = sorted(p for p in data.rglob('*') if p.is_file())
    source_hashes = {str(p.relative_to(data)): sha(p) for p in source_paths}
    write_json(out / 'source_sha256.json', source_hashes)
    images, annotations = {}, {}
    for folder, index, suffixes in [('JPEGImages', images, {'.jpg', '.jpeg', '.png'}), ('Annotations', annotations, {'.xml'})]:
        for path in sorted((data / folder).glob('*')):
            if path.is_file() and path.suffix.lower() in suffixes:
                if path.stem in index:
                    issue('duplicate_stem', path.stem, [str(index[path.stem]), str(path)])
                index[path.stem] = path
    # Official XMLDataset addresses only direct children by split ID.
    extra_files = [str(p.relative_to(data)) for p in source_paths if '.ipynb_checkpoints' in p.parts]
    for name in extra_files: issue('non_dataset_checkpoint_file', '', name)
    splits = {}
    for name in ['train', 'val', 'test', 'trainval']:
        path = data / 'ImageSets/Main' / (name + '.txt')
        if not path.exists():
            issue('missing_split', '', name); splits[name] = []; continue
        ids = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            parts = line.split()
            if len(parts) != 1:
                issue('invalid_split_line', '', {'split': name, 'line': number, 'text': line}); continue
            ids.append(parts[0])
        splits[name] = ids
        for sid, count in Counter(ids).items():
            if count > 1: issue('duplicate_id_within_split', sid, {'split': name, 'count': count})
            if sid not in images or sid not in annotations: issue('split_missing_file', sid, name)
    for a, b in itertools.combinations(['train', 'val', 'test'], 2):
        for sid in sorted(set(splits[a]) & set(splits[b])): issue('split_id_overlap', sid, [a, b])
    for sid in sorted(set(splits['trainval']) ^ (set(splits['train']) | set(splits['val']))):
        issue('trainval_union_mismatch', sid, 'trainval != train union val')
    partition = set().union(*(set(splits[n]) for n in ['train', 'val', 'test']))
    for sid in sorted(set(images) - partition): issue('unassigned_image', sid, 'not in train/val/test')
    file_groups, pixel_groups = defaultdict(list), defaultdict(list)
    for sid in sorted(set(images) | set(annotations)):
        start = len(issues)
        rec = {'sample_id': sid, 'splits': [n for n in ['train', 'val', 'test'] if sid in set(splits[n])], 'objects': []}
        path, xml = images.get(sid), annotations.get(sid)
        if path:
            rec['image'] = str(path.relative_to(data))
            rec['file_sha256'] = source_hashes[rec['image']]
            file_groups[rec['file_sha256']].append(sid)
            try:
                with Image.open(path) as im: im.verify()
                with Image.open(path) as im:
                    im.load(); rec['width'], rec['height'] = im.size
                    rec['mode'] = im.mode; rec['format'] = im.format
                    if im.getexif().get(274, 1) != 1: issue('exif_orientation', sid, im.getexif().get(274))
                    rgb = im.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode() + b'\0' + rgb.tobytes()).hexdigest()
                    rec['pixel_sha256'] = digest; pixel_groups[digest].append(sid)
            except Exception as exc: issue('corrupt_image', sid, str(exc))
        else: issue('missing_image', sid, str(xml))
        if xml:
            rec['annotation'] = str(xml.relative_to(data))
            try:
                tree = ET.parse(xml).getroot(); tags.update(e.tag for e in tree.iter())
                rec['xml_filename'] = tree.findtext('filename')
                if path and tree.findtext('filename') != path.name: issue('filename_mismatch', sid, tree.findtext('filename'))
                dims = [int(tree.findtext('size/' + k)) for k in ['width', 'height', 'depth']]
                rec['xml_size'] = dims
                if 'width' in rec and dims[:2] != [rec['width'], rec['height']]: issue('size_mismatch', sid, dims)
                if 'mode' in rec and dims[2] != Image.getmodebands(rec['mode']): issue('depth_mismatch', sid, dims[2])
                objects = tree.findall('object')
                if not objects: issue('missing_objects', sid, 'No object elements')
                if len(objects) != 1: issue('non_single_object_image', sid, len(objects))
                for idx, obj in enumerate(objects):
                    label = obj.findtext('name')
                    entry = {'index': idx, 'raw_label': label, 'class_name': LABELS.get(label)}
                    rec['objects'].append(entry)
                    if label not in LABELS: issue('unknown_or_missing_label', sid, {'object': idx, 'label': label})
                    entry['difficult'] = obj.findtext('difficult')
                    entry['truncated'] = obj.findtext('truncated')
                    try:
                        values = [float(obj.findtext('bndbox/' + k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                        entry['raw_xyxy'] = values
                        if not all(v.is_integer() for v in values):
                            issue('noninteger_or_nonfinite_box', sid, entry); continue
                        raw = list(map(int, values)); converted = [v - 1 for v in raw]
                        entry['loader_xyxy'] = converted
                        entry['crop_xyxy_half_open'] = converted
                        x1, y1, x2, y2 = raw
                        if x2 <= x1 or y2 <= y1: issue('nonpositive_box', sid, entry)
                        if 'width' in rec and not (1 <= x1 < x2 <= rec['width'] and 1 <= y1 < y2 <= rec['height']): issue('raw_coordinate_exceeds_image_bounds', sid, entry)
                        a, b, c, d = converted
                        if 'width' in rec and not (0 <= a < c <= rec['width'] and 0 <= b < d <= rec['height']): issue('out_of_bounds_box', sid, entry)
                        entry['roi_width'], entry['roi_height'] = x2-x1, y2-y1
                    except (ValueError, TypeError, OverflowError) as exc: issue('malformed_box', sid, {'object': idx, 'error': str(exc)})
            except Exception as exc: issue('malformed_xml', sid, str(exc))
        else: issue('missing_annotation', sid, str(path))
        rec['structurally_valid'] = len(issues) == start
        records.append(rec)
    by_id = {r['sample_id']: r for r in records}
    duplicates = {}
    for mode, groups in [('file', file_groups), ('decoded_rgb', pixel_groups)]:
        duplicates[mode] = []
        for digest, ids in groups.items():
            if len(ids) > 1:
                split_names = sorted({n for sid in ids for n in by_id[sid]['splits']})
                labels = sorted({o['raw_label'] for sid in ids for o in by_id[sid]['objects'] if o['raw_label'] is not None})
                group = {'sha256': digest, 'sample_ids': ids, 'splits': split_names, 'cross_split': len(split_names)>1, 'labels': labels, 'label_conflict': len(labels)>1}
                duplicates[mode].append(group)
                for sid in ids: issue('duplicate_' + mode, sid, group)
    classes = Counter(o['raw_label'] for r in records for o in r['objects'])
    # Paper reports 1428 / 3572. Do not infer which sample accounts for a discrepancy.
    paper_counts = {'0': 1428, '1': 3572}
    if dict(classes) != paper_counts: issue('paper_class_count_mismatch', '', {'actual': dict(classes), 'paper': paper_counts, 'affected_samples': 'Cannot identify from aggregate counts; all labels are in manifest.jsonl'})
    eligible = sorted(r['sample_id'] for r in records if r['structurally_valid'] and len(r['objects']) == 1)
    chosen = random.Random(args.seed).sample(eligible, min(args.preview_count, len(eligible)))
    if len(chosen) < args.preview_count: issue('insufficient_preview_samples', '', len(chosen))
    previews = out / 'previews'; previews.mkdir()
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 16) if Path(font_path).exists() else ImageFont.load_default(size=16)
    selected, panels = [], []
    for order, sid in enumerate(chosen, 1):
        r = by_id[sid]; obj = r['objects'][0]; box = obj['crop_xyxy_half_open']
        with Image.open(data / r['image']) as src: original = src.convert('RGB')
        roi = original.crop(tuple(box))
        boxed = original.copy(); draw = ImageDraw.Draw(boxed)
        # Draw the exact included crop boundary, accounting for Pillow's inclusive outline.
        draw.rectangle([box[0], box[1], box[2]-1, box[3]-1], outline='#00ff75', width=3)
        stem = f'{order:02d}_{sid}'
        original.save(previews / (stem + '_full.png'))
        boxed.save(previews / (stem + '_boxed.png'))
        roi.save(previews / (stem + '_roi.png'))
        panel = Image.new('RGB', (1200, 390), '#111827'); d = ImageDraw.Draw(panel)
        for col, (im, title) in enumerate([(original, 'Full image'), (boxed, 'Annotation box'), (roi, 'ROI (no padding)')]):
            d.text((col*400+10, 8), title, font=font, fill='white')
            thumb = ImageOps.contain(im, (390, 280))
            panel.paste(thumb, (col*400+(400-thumb.width)//2, 35+(280-thumb.height)//2))
        d.text((10, 321), f'{order:02d}  ID={sid}  split={"/".join(r["splits"])}  original={original.width}x{original.height}  ROI={roi.width}x{roi.height}', font=font, fill='white')
        d.text((10, 343), f'XML xyxy={obj["raw_xyxy"]}  crop half-open={box}', font=font, fill='white')
        d.text((10, 365), f'AUDIT ONLY: XML label={obj["raw_label"]} ({obj["class_name"]}); never use this panel as model input', font=font, fill='#fbbf24')
        panel.save(previews / (stem + '_panel.jpg'), quality=95)
        panels.append(panel)
        selected.append({'order': order, **r, 'preview_panel': f'previews/{stem}_panel.jpg', 'roi_file': f'previews/{stem}_roi.png', 'roi_pixel_sha256': hashlib.sha256(roi.tobytes()).hexdigest()})
    for page, offset in enumerate(range(0, len(panels), 5), 1):
        chunk = panels[offset:offset+5]; sheet = Image.new('RGB', (1200, 390*len(chunk)))
        for row, panel in enumerate(chunk): sheet.paste(panel, (0, row*390))
        sheet.save(out / f'contact_sheet_{page:02d}.jpg', quality=94)
    index = ['# TN5000 audit previews', '', f'Seed: {args.seed}. Sorted eligible IDs, Python random.Random(seed).sample, without replacement.', '', 'Labels occur only in audit captions. Full images and ROI PNG files have no added class text.', '']
    for s in selected:
        index.extend([f'## {s["order"]:02d} — {s["sample_id"]}', '', f'![Full / annotation / ROI]({s["preview_panel"]})', ''])
    (out / 'PREVIEWS.md').write_text('\n'.join(index))
    after_paths = sorted(p for p in data.rglob('*') if p.is_file())
    changed = [str(p.relative_to(data)) for p in after_paths if source_hashes.get(str(p.relative_to(data))) != sha(p)]
    changed += sorted(set(source_hashes) - {str(p.relative_to(data)) for p in after_paths})
    for p in changed: issue('source_changed_during_audit', '', p)
    summary = {
        'run_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
        'data_root': str(data), 'source_files': len(source_hashes), 'source_unchanged': not changed,
        'extra_checkpoint_files': extra_files, 'image_count': len(images), 'annotation_count': len(annotations), 'object_count': sum(classes.values()),
        'class_counts': dict(classes), 'class_meaning': LABELS, 'paper_class_counts': paper_counts,
        'split_counts': {n: len(v) for n,v in splits.items()},
        'split_class_counts': {n: dict(Counter(o['raw_label'] for r in records if r['sample_id'] in set(ids) for o in r['objects'])) for n,ids in splits.items()},
        'objects_per_image': dict(Counter(len(r['objects']) for r in records)),
        'image_sizes': dict(Counter(f'{r.get("width")}x{r.get("height")}' for r in records)),
        'image_modes': dict(Counter(r.get('mode') for r in records)), 'xml_tag_counts': dict(tags),
        'difficult_counts': dict(Counter(o.get('difficult') for r in records for o in r['objects'])),
        'truncated_counts': dict(Counter(o.get('truncated') for r in records for o in r['objects'])),
        'small_box_under32_ids': [r['sample_id'] for r in records if any(min(o.get('roi_width',0),o.get('roi_height',0))<32 for o in r['objects'])],
        'issues_by_kind': dict(Counter(i['kind'] for i in issues)), 'affected_sample_ids': sorted({i['sample_id'] for i in issues if i['sample_id']}),
        'duplicate_group_counts': {k: len(v) for k,v in duplicates.items()},
        'cross_split_duplicate_group_counts': {k: sum(g['cross_split'] for g in v) for k,v in duplicates.items()},
        'patient_nodule_ids': 'No dedicated ID fields or mapping files; numeric image IDs are not patient/nodule IDs.',
        'near_duplicate_detection': 'not performed; exact encoded-file and decoded-RGB comparison only',
        'seed': args.seed, 'eligible_count': len(eligible), 'preview_count': len(selected), 'selected_ids': chosen,
        'bbox_policy': 'Official loader: all four XML coordinates minus 1; Pillow crop half-open using that result, no padding/clamping/resizing in saved ROI.',
        'label_source': 'https://www.nature.com/articles/s41597-025-05757-4#Sec2',
    }
    write_json(out / 'summary.json', summary); write_json(out / 'issues.json', issues)
    write_json(out / 'duplicates.json', duplicates); write_json(out / 'preview_manifest.json', selected)
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    with (out / 'issues.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['kind','sample_id','detail']); writer.writeheader()
        for i in issues: writer.writerow({**i,'detail':json.dumps(i['detail'],ensure_ascii=False)})
    lines = ['# 数据检查实际运行结果', '', '| 检查项 | 状态 | 结果 |', '|---|---|---|']
    def row(name, status, value): lines.append(f'| {name} | {status} | {value} |')
    row('全量检查', '通过', f'{len(images)} 图像 / {len(annotations)} XML / {sum(classes.values())} 框')
    row('来源保持不变', '通过' if not changed else '发现问题', f'{len(source_hashes)} 文件前后 SHA256 比较')
    for title,kinds in [('缺失标注',['missing_annotation','missing_objects']),('缺失图片',['missing_image']),('图片解码',['corrupt_image']),('文件名与尺寸',['filename_mismatch','size_mismatch','depth_mismatch']),('标注格式与标签',['malformed_xml','unknown_or_missing_label']),('框合法性',['nonpositive_box','out_of_bounds_box','raw_coordinate_exceeds_image_bounds','malformed_box','noninteger_or_nonfinite_box']),('官方划分完整性',['missing_split','invalid_split_line','split_missing_file','split_id_overlap','duplicate_id_within_split','trainval_union_mismatch','unassigned_image'])]:
        count=sum(summary['issues_by_kind'].get(k,0) for k in kinds); row(title, '发现问题' if count else '通过', f'{count} 条异常')
    row('类别统计与论文一致性','发现问题' if dict(classes)!=paper_counts else '通过',f'实测 {dict(classes)}；论文 {paper_counts}')
    row('精确重复图像', '发现问题' if any(duplicates.values()) else '通过', summary['duplicate_group_counts'])
    row('近重复图像','尚未验证','未执行感知相似度或多视角配对检查')
    row('患者／结节分组标识','信息缺失','无独立 ID 字段、映射表；无法验证患者级隔离')
    row('固定种子预览','通过' if len(selected)==args.preview_count else '发现问题',f'seed={args.seed}; {len(selected)} 个；等待用户目视验收')
    row('模型、推理、训练','尚未验证','未下载模型，未加载模型，未启动训练')
    lines.extend(['','所有异常详见 [issues.csv](issues.csv)，所有样本见 [manifest.jsonl](manifest.jsonl)。', '', '## 官方划分类别统计', '', '| 划分 | 总数 | 0（良性） | 1（恶性） |','|---|---:|---:|---:|'])
    for n,c in summary['split_class_counts'].items(): lines.append(f'| {n} | {len(splits[n])} | {c.get("0",0)} | {c.get("1",0)} |')
    lines.extend(['', '## 受影响样本', ''])
    for i in issues: lines.append(f'- `{i["kind"]}` / `{i["sample_id"] or "无法定位单一样本"}`：{json.dumps(i["detail"],ensure_ascii=False)}')
    (out / 'DATA_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: summary[k] for k in ['image_count','annotation_count','class_counts','split_class_counts','issues_by_kind','duplicate_group_counts','source_unchanged','preview_count','selected_ids']},ensure_ascii=False,indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
