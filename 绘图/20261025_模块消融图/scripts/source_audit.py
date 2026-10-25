"""Read-only extraction of existing assumed results and source fingerprints."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import runpy
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
METRICS = {'SR': 'SR', 'Cost': 'Average Cost', 'BVR': 'BVR', 'CR': 'CR', 'MC': 'Minimum Clearance'}
SOURCES = [
    'CCDC/latex/1020/main.tex', 'CORL/scripts/generate_experiment_results_v3.py',
    'CORL/data/experiment_results_rcppo_ra_v3.xlsx',
    'CCDC/references/ablation_review_20261025/current_ablation_snapshot.json',
    '绘图/20261023_多方法指标对比/data/assumed_point_estimates.csv',
    '绘图/20261023_多方法指标对比/data/protocol.json',
    '绘图/20261025_预算扫描图/data/reviewed_points.jsonl',
    '绘图/20261025_预算扫描图/data/plot_protocol.json',
    '绘图/20261021_轨迹与同步曲线/data/summary.json',
    '绘图/20261021_轨迹与同步曲线/data/protocol.md',
    '绘图/20261021_部署与校准总览图/data/README.md',
    'CCDC/images/1020/deployment_calibration_overview.pdf',
    'CCDC/images/1020/metrics_comparison_assumed.pdf',
    'CCDC/images/1020/trajectory_sync.pdf',
    'CCDC/images/1020/budget_sweep_horizontal.pdf',
]

def dump(relative, value):
    path = PROJECT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def table_rows(tex, label):
    block = next(b for b in re.findall(r'\\begin\{table\}(.*?)\\end\{table\}', tex, re.S) if label in b)
    rows = []
    for line in block.splitlines():
        cells = line.split(' & ')
        if len(cells) != 6 or cells[0] in {'Method', 'Variant'}:
            continue
        values = [float(re.search(r'\d+(?:\.\d+)?', c).group()) for c in cells[1:]]
        rows.append({'label': cells[0], 'values': values})
    return rows

def workbook_rows(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    result = {}
    with zipfile.ZipFile(path) as book:
        wb = ET.fromstring(book.read('xl/workbook.xml'))
        rel = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in rel}
        for sheet in wb.find('s:sheets', ns):
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = targets[rid]
            xml = ET.fromstring(book.read(target.lstrip('/') if target.startswith('/') else 'xl/' + target))
            rows = []
            for row in xml.findall('.//s:sheetData/s:row', ns):
                cells = []
                for cell in row:
                    text = ''.join(cell.itertext())
                    cells.append({'cell': cell.attrib['r'], 'value': text})
                rows.append(cells)
            result[sheet.attrib['name']] = rows
    return result

def main():
    source = runpy.run_path(str(ROOT / SOURCES[1]))
    tex = (ROOT / SOURCES[0]).read_text(encoding='utf-8')
    ablation = table_rows(tex, 'tab:ablation_draft')
    assert len(ablation) == 5
    for actual, expected in zip(ablation, source['ABLATION_ROWS']):
        assert actual['values'] == list(expected[1:]), (actual, expected)
    anchors = {}
    for cfg, method in [('FULL', 'HD-MCRA'), ('ZERO', 'RC-PPO')]:
        for metric, source_metric in METRICS.items():
            vals = source['MAIN_DATA'][source_metric][method]
            scale = 100 if metric in {'SR', 'CR', 'BVR'} else 1
            anchors[cfg + '_' + metric.upper()] = {
                'value': sum(vals) / len(vals) * scale,
                'scenario_values': [v * scale for v in vals],
                'display_value': round(sum(vals) / len(vals) * scale, 2 if metric == 'MC' else 1),
            }
    hashes = [{'path': str(ROOT / s), 'sha256': hashlib.sha256((ROOT / s).read_bytes()).hexdigest()} for s in SOURCES]
    workbook = workbook_rows(ROOT / SOURCES[2])
    ablation_sheet = {key: rows for key, rows in workbook.items() if 'ablation' in key.lower()}
    git = subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8').stdout
    result = {'data_status': 'hypothetical', 'sources': hashes, 'manuscript_ablation_rows': ablation,
              'manuscript_main_rows': table_rows(tex, 'tab:main_results'), 'anchors': anchors,
              'source_main_data': source['MAIN_DATA'], 'workbook_ablation': ablation_sheet,
              'workbook_main_averages': workbook.get('main_scenario_averages', []),
              'workbook_note': 'Workbook has no ablation sheet; ablation authority is the matching manuscript table and source ABLATION_ROWS. Workbook is checked for main-result anchors only.',
              'workbook_sheets': list(workbook), 'git_status_at_start': git,
              'unresolved_empirical_details': ['cost units/function', 'absolute scenario budgets', 'seed/trial allocation', 'per-trial records and interval estimator']}
    dump('reviews/baseline.json', result)
    dump('data/source_manifest.json', {'data_status': 'hypothetical', 'sources': hashes})
    print(json.dumps({'passed': True, 'source_count': len(hashes), 'table_values_checked': 25,
                      'anchors': anchors, 'workbook_ablation': ablation_sheet,
                      'python': sys.executable, 'packages': {p: importlib.metadata.version(p) for p in ['matplotlib','numpy','PyMuPDF','Pillow']},
                      'pdftoppm': shutil.which('pdftoppm')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
