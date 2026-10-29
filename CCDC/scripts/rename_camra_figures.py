"""Rename current CCDC vector figure labels, without changing plotted data.

Reads immutable pre-rename copies under output/1020/camra_rename_20261029/sources.
Only writes the current four CCDC figures and their local verification products.
Original drawing projects and historical data IDs are deliberately untouched.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
import re

import fitz
import numpy as np
from matplotlib import font_manager

from method_names import METHOD_NAME

CCDC = Path(__file__).resolve().parents[1]
ROOT = CCDC.parent
REVIEW = CCDC / 'output/1020/camra_rename_20261029'
IMAGES = CCDC / 'images/1020'
FIGURES = ('metrics_comparison.pdf', 'trajectory_sync.pdf',
           'ablation_horizontal.pdf', 'budget_sweep_horizontal.pdf')
OLD_NAME = 'HD-MCRA'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_text(text):
    return Counter(re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?', text))


def name_spans(page):
    result = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                text = ''.join(c['c'] for c in span['chars'])
                if OLD_NAME not in text:
                    continue
                start = text.index(OLD_NAME)
                chars = span['chars'][start:start+len(OLD_NAME)]
                rect = fitz.Rect(chars[0]['bbox'])
                for char in chars[1:]:
                    rect |= fitz.Rect(char['bbox'])
                result.append({'bbox': list(rect), 'origin': chars[0]['origin'],
                               'size': span['size'], 'font': span['font'],
                               'color': span['color']})
    return result


def verify_figure(source, target, labels):
    with fitz.open(source) as old, fitz.open(target) as new:
        assert len(old) == len(new) == 1
        a, b = old[0], new[0]
        assert a.rect == b.rect and a.rotation == b.rotation == 0
        old_text, new_text = a.get_text(), b.get_text()
        assert OLD_NAME not in new_text
        assert new_text.count(METHOD_NAME) == old_text.count(OLD_NAME) == len(labels)
        assert numeric_text(old_text) == numeric_text(new_text)
        assert Counter(old_text.replace(OLD_NAME, '').split()) == Counter(new_text.replace(METHOD_NAME, '').split())
        # The redaction operation is explicitly forbidden from touching graphics.
        # Removing a text operation changes later sequence numbers, not paths.
        paths = lambda page: [{k: v for k, v in item.items() if k != 'seqno'}
                              for item in page.get_drawings()]
        assert paths(a) == paths(b), 'Vector paths changed'
        assert a.get_images(full=True) == b.get_images(full=True)
        matrix = fitz.Matrix(2, 2)
        ap, bp = a.get_pixmap(matrix=matrix), b.get_pixmap(matrix=matrix)
        aa = np.frombuffer(ap.samples, dtype=np.uint8).reshape(ap.height, ap.width, ap.n)
        bb = np.frombuffer(bp.samples, dtype=np.uint8).reshape(bp.height, bp.width, bp.n)
        changed = np.any(aa != bb, axis=2)
        outside = changed.copy()
        for item in labels:
            rect = fitz.Rect(item['bbox']) * matrix
            # Two pixels include boundary antialiasing only.
            x0, y0 = max(0, math.floor(rect.x0)-2), max(0, math.floor(rect.y0)-2)
            x1, y1 = min(ap.width, math.ceil(rect.x1)+2), min(ap.height, math.ceil(rect.y1)+2)
            outside[y0:y1, x0:x1] = False
        assert not outside.any(), f'{outside.sum()} changed pixels outside method labels'
        return {'passed': True, 'source_sha256': sha(source), 'output_sha256': sha(target),
                'size_mm': [v*25.4/72 for v in a.rect[2:]], 'labels': labels,
                'numeric_tokens_unchanged': True, 'vector_paths_unchanged': True,
                'changed_pixels_outside_labels': int(outside.sum())}


def render_names():
    records = {}
    for name in FIGURES:
        source, target = REVIEW/'sources'/name, IMAGES/name
        if not source.is_file():
            raise FileNotFoundError(f'Missing pre-rename source snapshot: {source}')
        with fitz.open(source) as doc:
            page = doc[0]
            labels = name_spans(page)
            assert len(labels) == (2 if name == 'budget_sweep_horizontal.pdf' else 1)
            for item in labels:
                page.add_redact_annot(fitz.Rect(item['bbox']), fill=False)
            page.apply_redactions(images=0, graphics=0, text=0)
            for index, item in enumerate(labels):
                weight = 'bold' if 'Bold' in item['font'] else 'normal'
                font = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight=weight))
                resource = f'CaMRAFont{index}'
                page.insert_font(fontname=resource, fontfile=font)
                color = item['color']
                rgb = tuple(((color >> shift) & 255)/255 for shift in (16, 8, 0))
                page.insert_text(item['origin'], METHOD_NAME, fontsize=item['size'],
                                 fontname=resource, color=rgb)
            doc.set_metadata({key: value.replace(OLD_NAME, METHOD_NAME)
                              if isinstance(value, str) else value
                              for key, value in doc.metadata.items()})
            doc.save(target, garbage=0, deflate=True)
        records[name] = verify_figure(source, target, labels)
        with fitz.open(target) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(REVIEW/(target.stem+'.png'))
    (REVIEW/'figure_manifest.json').write_text(json.dumps({
        'method': METHOD_NAME, 'scope': 'current CCDC submission only',
        'operation': 'vector label replacement; values, paths, dimensions and non-label pixels unchanged',
        'figures': records}, indent=2)+'\n', encoding='utf-8')
    return records


def verify_manifest():
    report = json.loads((REVIEW/'figure_manifest.json').read_text(encoding='utf-8'))
    assert report['method'] == METHOD_NAME and set(report['figures']) == set(FIGURES)
    for name, item in report['figures'].items():
        source, target = REVIEW/'sources'/name, IMAGES/name
        assert sha(source) == item['source_sha256']
        assert sha(target) == item['output_sha256']
        verify_figure(source, target, item['labels'])
    return report


def verify_preserved(protocol):
    """Old file hashes remain authoritative, except verified label-only changes."""
    report = verify_manifest()
    for item in protocol['files_preserved']:
        path = ROOT/item['path']
        if path.parent == IMAGES and path.name in FIGURES:
            assert report['figures'][path.name]['source_sha256'] == item['sha256']
        else:
            assert sha(path) == item['sha256'], item['path']
    return {'passed': True, 'files': len(protocol['files_preserved']),
            'label_only_figures': len(FIGURES), 'all_other_files_unchanged': True}


def audit_current():
    verify_manifest()
    tex = (CCDC/'latex/1020/main.tex').read_text(encoding='utf-8')
    assert OLD_NAME not in tex and 'calibrated minimum-cost reach-avoid (CaMRA)' in tex
    assert tex.count('\\begin{figure}') == 7 and tex.count('\\begin{table}') == 3
    assert '\\begin{figure*}' not in tex and '\\begin{table*}' not in tex
    # Reuse numeric audits against unchanged legacy IDs; never rewrite data.
    spec = importlib.util.spec_from_file_location('completion_audit', CCDC/'scripts/audit_manuscript_completion_20261026.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = module.read_json(CCDC/'data/manuscript_completion_20261026/main_results.json')
    ablation = module.read_json(CCDC/'data/manuscript_completion_20261026/ablation_budget_results.json')
    metrics = ['SR', 'Cost', 'CR', 'BVR', 'MC']
    methods = ['HD-MCRA', 'RC-PPO', 'RA', 'PPO_beta_L', 'PPO_beta_H', 'CPPO_beta_L', 'CPPO_beta_H']
    expected = [[round(main['equal_scene_means'][method][k], 2 if k == 'MC' else 1)
                 for k in metrics] for method in methods]
    assert module.table_values(tex, 'tab:main_results') == expected
    expected = [[round(next(r['value'] for r in ablation['ablation'] if r['configuration'] == config
                            and r['metric'] == k), 2 if k == 'MC' else 1)
                 for k in metrics] for config in ['FULL', 'ZERO', 'NO_ALPHA', 'NO_DELTA']]
    assert module.table_values(tex, 'tab:ablation_draft') == expected
    with fitz.open(CCDC/'output/1020/main.pdf') as pdf:
        text = '\n'.join(page.get_text() for page in pdf)
        assert OLD_NAME not in text and METHOD_NAME in text
        pages = len(pdf)
    return {'passed': True, 'main_table_cells': 35, 'ablation_table_cells': 20,
            'renamed_figure_labels': 5, 'pages': pages, 'legacy_data_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='read-only validation after rebuilding the paper')
    args = parser.parse_args()
    result = audit_current() if args.check else render_names()
    print(json.dumps(result, indent=2))
