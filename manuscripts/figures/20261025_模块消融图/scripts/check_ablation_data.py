"""Check assumed centers, review coverage, anchors and plotted mappings."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import math

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
CONFIGS = ['FULL', 'NO_ALPHA', 'NO_DELTA', 'ZERO', 'NO_HIERARCHY']
METRICS = ['SR', 'Cost', 'BVR', 'CR', 'MC']

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def validate(points, protocol, reviews=None, render=None):
    errors = []
    expected = {c + '_' + m.upper() for c in CONFIGS for m in METRICS}
    indexed = {p['point_id']: p for p in points}
    if len(points) != len(indexed):
        errors.append('duplicate point ID')
    if set(indexed) != expected:
        errors.append('missing or extra point ID')
    for p in points:
        metric, cfg = p['metric'], p['configuration']
        if p['point_id'] != cfg + '_' + metric.upper():
            errors.append('point identity mismatch: ' + p['point_id'])
        value = p.get('value', p.get('baseline_value'))
        unit = 'percent' if metric in ['SR', 'CR', 'BVR'] else ('m' if metric == 'MC' else 'draft cost units')
        role = 'support' if metric == 'MC' or cfg == 'NO_HIERARCHY' else 'main'
        if p['unit'] != unit or p['role'] != role:
            errors.append('unit/role mismatch: ' + p['point_id'])
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append('invalid nonnegative center: ' + p['point_id'])
            continue
        if unit == 'percent' and value > 100:
            errors.append('percentage exceeds 100: ' + p['point_id'])
        if p.get('data_status') != 'hypothetical':
            errors.append('unexpected data status: ' + p['point_id'])
        anchor = protocol['anchors'].get(p['point_id'])
        if anchor and not math.isclose(value, anchor['value'], abs_tol=1e-9):
            errors.append('reference anchor mismatch: ' + p['point_id'])
    if set(indexed) == expected:
        for cfg in CONFIGS:
            vals = {m: indexed[cfg + '_' + m.upper()].get('value', indexed[cfg + '_' + m.upper()].get('baseline_value')) for m in METRICS}
            if vals['SR'] + vals['CR'] > 100 + 1e-9:
                errors.append('success/collision exclusion violated: ' + cfg)
    if reviews is not None:
        latest = {}
        for r in reviews:
            if any(k not in r for k in protocol['point_review_required_fields']):
                errors.append('review fields missing: ' + r.get('point_id', '?'))
            old = latest.get(r['point_id'])
            if old and (r['revision'] <= old['revision'] or r['supersedes_revision'] != old['revision']):
                errors.append('review revision order invalid: ' + r['point_id'])
            latest[r['point_id']] = r
        if set(latest) != expected:
            errors.append('review coverage incomplete')
        for pid, p in indexed.items():
            r = latest.get(pid)
            if r and (r['decision'] not in ['retain', 'revise'] or r['adopted_value'] is None or not r['reasoning']):
                errors.append('review unresolved: ' + pid)
            if r and r['adopted_value'] is not None and not math.isclose(r['adopted_value'], p.get('value', p.get('baseline_value')), abs_tol=1e-9):
                errors.append('review/data mismatch: ' + pid)
    if render is not None:
        active = render.get('active_metrics', protocol['metric_order'])
        expected_pairs = {(m, cfg) for m in active for cfg in protocol['row_order']}
        pairs = render['comparisons']
        if len(pairs) != len(expected_pairs) or {(p['metric'], p['configuration']) for p in pairs} != expected_pairs:
            errors.append('plot comparisons missing or duplicated')
        for p in pairs:
            for name, cfg in [('full_value', 'FULL'), ('ablated_value', p['configuration'])]:
                source = indexed[cfg + '_' + p['metric'].upper()]
                if not math.isclose(p[name], source.get('value', source.get('baseline_value')), abs_tol=1e-9):
                    errors.append('rendered value mismatch: ' + str((p['metric'],cfg)))
            if not math.isclose(p['difference'], p['ablated_value'] - p['full_value'], abs_tol=1e-9):
                errors.append('rendered difference mismatch')
    return errors

def summary(points, protocol):
    idx = {p['point_id']: p.get('value',p.get('baseline_value')) for p in points}
    comparisons = [{'metric': m, 'configuration': c, 'full_value': idx['FULL_'+m.upper()], 'ablated_value': idx[c+'_'+m.upper()], 'difference': idx[c+'_'+m.upper()]-idx['FULL_'+m.upper()]} for m in protocol['metric_order'] for c in protocol['row_order']]
    return {'total_points': len(points), 'main_points': sum(p['role']=='main' for p in points), 'support_points': sum(p['role']=='support' for p in points), 'anchors': len(protocol['anchors']), 'comparisons': comparisons}

def self_check(points, protocol):
    cases = {}
    def expect_failure(name, altered):
        errors = validate(altered, protocol)
        assert errors, name
        cases[name] = errors
    assert not validate(points, protocol)
    expect_failure('missing_point', points[:-1])
    expect_failure('duplicate_point', points + [points[0]])
    wrong = copy.deepcopy(points); wrong[0]['baseline_value'] /= 100
    expect_failure('fraction_used_as_percent', wrong)
    wrong = copy.deepcopy(points); wrong[0]['unit'] = 'fraction'
    expect_failure('wrong_unit', wrong)
    wrong = copy.deepcopy(points); wrong[5]['baseline_value'] = 101
    expect_failure('invalid_probability', wrong)
    wrong = copy.deepcopy(points); wrong[6]['baseline_value'] = -1
    expect_failure('negative_cost', wrong)
    mapping = summary(points, protocol)
    mapping['comparisons'][0]['ablated_value'] += 1
    errors = validate(points, protocol, render=mapping)
    assert errors; cases['render_mapping_error'] = errors
    return {'passed': True, 'valid_baseline_passed': True, 'negative_checks': cases}

def cross_sources(protocol):
    baseline = read_json(PROJECT / 'reviews/baseline.json')
    with (ROOT / 'figures/20261023_多方法指标对比/data/assumed_point_estimates.csv').open(encoding='utf-8-sig', newline='') as handle:
        main = list(csv.DictReader(handle))
    budget = read_lines(ROOT / 'figures/20261025_预算扫描图/data/reviewed_points.jsonl')
    wb = { (row[0]['value'], row[1]['value']): float(row[2]['value']) for row in baseline['workbook_main_averages'][1:] }
    source_metric = {'Cost':'Average Cost', 'MC':'Minimum Clearance'}
    checks = []
    for pid, anchor in protocol['anchors'].items():
        cfg, metric_id = pid.split('_')
        method = 'HD-MCRA' if cfg == 'FULL' else 'RC-PPO'
        metric = 'Cost' if metric_id == 'COST' else metric_id
        scale = 100 if metric in ['SR','BVR','CR'] else 1
        expected = anchor['value']
        workbook = wb[(source_metric.get(metric,metric), method)]
        assert math.isclose(workbook, round(expected/scale,4), abs_tol=1e-10), (pid,workbook,expected)
        entry = {'point_id':pid, 'expected':expected, 'workbook_stored':workbook, 'workbook_decimals':4, 'workbook_rounding_passed':True}
        if metric != 'MC':
            vals = [float(p['estimate']) for p in main if p['method']==method and p['metric']==metric]
            assert len(vals)==3, (pid, vals)
            assert math.isclose(sum(vals)/3, expected, abs_tol=1e-9)
            refs = [p for p in budget if p['method']==method and p['metric']==metric and math.isclose(p['budget_ratio'],1)]
            assert len(refs)==1
            assert math.isclose(refs[0]['value']*scale, expected, abs_tol=1e-9), (pid,refs)
            entry.update(main_scene_values=vals, budget_reference_value=refs[0]['value']*scale, passed=True)
        else:
            entry.update(passed=True, note='MC anchor uses source scenario mean and manuscript display, absent from four-metric main chart.')
        checks.append(entry)
    unchanged = []
    for item in baseline['sources']:
        current = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
        assert current == item['sha256'], item['path']
        unchanged.append(item['path'])
    return {'passed':True, 'anchor_cells':len(checks), 'checks':checks, 'unchanged_sources':unchanged,
            'notes':['Workbook stores averages to four decimals; manuscript stores rates/cost to one, MC to two.', 'No ablation sheet exists in original workbook.']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--cross-sources', action='store_true')
    parser.add_argument('--render', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    protocol = read_json(PROJECT / 'data/review_protocol.json')
    if args.cross_sources:
        report = cross_sources(protocol)
    elif args.self_check:
        report = self_check(read_json(PROJECT / 'data/baseline_points.json')['points'], protocol)
    else:
        points = read_json(PROJECT / 'data/baseline_points.json')['points'] if args.baseline else read_lines(PROJECT / 'data/reviewed_points.jsonl')
        reviews = None if args.baseline else read_lines(PROJECT / 'reviews/point_review.jsonl')
        errors = validate(points, protocol, reviews, read_json(args.render) if args.render else None)
        report = {'passed': not errors, 'errors': errors, **summary(points, protocol)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
