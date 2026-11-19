"""Main-agent checks on saved L1 data; no simulator or network inference."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(NAV / 'scripts'))
from audit_nav043_saved_applicability import analyze_trial, load, sha, write_new

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
script = NAV / 'scripts/audit_nav043_exploration_applicability.py'
compile(script.read_text(), str(script), 'exec')
source = NAV / 'scripts/audit_nav043_saved_applicability.py'
baseline_source = subprocess.check_output(['git', 'show', '56b4163:CaMRA/navigation/scripts/audit_nav043_saved_applicability.py'], cwd=ROOT)
assert source.read_bytes() == baseline_source
report_path = NAV / 'reports/nav043_exploration_applicability_v1.json'
assert not report_path.exists()
assert not (NAV / 'reports/nav043_exploration_review_v1.json').exists()
refusal = subprocess.run([sys.executable, '-B', str(script)], cwd=ROOT, text=True, capture_output=True)
assert refusal.returncode != 0 and 'original saved physical-event review is not finished' in refusal.stderr
assert not report_path.exists()
with (HERE / 'applicability-premature-refusal.log').open('x') as f:
    f.write(refusal.stdout + refusal.stderr)
calibration = load(NAV / 'config/calibration_applicability_prereg_v1.json')
geometry = load(NAV / 'config/geometry_margin_closedloop_v1.json')
alpha = calibration['frozen_values']['alpha']
delta = calibration['frozen_values']['delta']
first = load(HERE / 'first-nominal-review.json')
rows = []
for method in ('uncalibrated', 'full'):
    trial = Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1/L1_empty') / method
    row = analyze_trial(trial, 'actual_state_W_selected', 'L1_empty', first['raw_sources'],
                        alpha, delta, geometry['table_q90_by_prefix_s'])
    saved = load(trial / 'substeps.json')
    with np.load(trial / 'energy.npz', allow_pickle=False) as energy:
        increments = np.maximum(energy['tau_applied'].astype(float) * energy['qdot_post'].astype(float), 0.).sum(axis=-1)[:, 0] * .005
    for seconds, prefix in row['prefixes_s'].items():
        n = int(seconds) * 200
        if len(saved) < n:
            assert not prefix['complete'] and 'eps_c' not in prefix
            continue
        actual = float(np.sum(increments[:n]))
        nominal = sum(s['cost_nom_J'] for s in saved[:n])
        deviation = max(s['deviation_m'] for s in saved[19:n:20])
        assert abs(prefix['actual_joint_work_J'] - actual) < 1.e-8
        assert abs(prefix['nominal_command_cost_J'] - nominal) < 1.e-8
        assert abs(prefix['eps_c'] - (actual / nominal - 1.)) < 1.e-10
        assert prefix['max_position_error_10Hz_m'] == deviation
        assert prefix['position_covered_by_original_delta_on_10Hz'] == (deviation <= delta)
        assert not prefix.get('old_table_transfer_accepted', False)
    assert row['budget_mode'] == 'actual_state_guarded_W'
    assert not row['independent_performance_evidence']
    rows.append(row)
for path, digest in first['raw_sources'].items():
    assert sha(path) == digest, path
protocol = load(NAV / 'config/nav043_exploration_v1.json')
for path, digest in protocol['sealed_hashes'].items():
    assert sha(path) == digest, path
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
plan = load(plan_path)
old = json.loads(subprocess.check_output(['git', 'show', '56b4163:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
assert {k:v for k,v in plan.items() if k != 'tasks'} == {k:v for k,v in old.items() if k != 'tasks'}
for a,b in zip(old['tasks'], plan['tasks']):
    if a['id'] == 'NAV-043':
        assert b['work_records'][:len(a['work_records'])] == a['work_records']
        assert {k:v for k,v in a.items() if k != 'work_records'} == {k:v for k,v in b.items() if k != 'work_records'}
    else:
        assert a == b
report = {
    'at': datetime.now().astimezone().isoformat(),
    'status': 'PASS_BASIC_L1_SAVED_PREFIX_CHECKS_COMPLETE_REVIEW_PENDING',
    'host': socket.getfqdn(), 'pane': os.environ.get('TMUX_PANE'),
    'script_sha256': sha(script), 'analyzer_sha256': sha(source),
    'verifier_sha256': sha(__file__), 'premature_production_call_refused_without_output': True,
    'L1_rows': rows, 'new_applications': 0, 'new_rollouts': 0, 'W_queries': 0,
    'training_updates': 0, 'calibration_fits': 0, 'test_subagent_used': False,
    'NAV043_passed': False, 'G5_passed': False,
}
write_new(HERE / 'applicability-ready-v1.json', report)
print(json.dumps({k:v for k,v in report.items() if k != 'L1_rows'}, ensure_ascii=False, indent=2), flush=True)
for row in rows:
    print(row['method'], row['terminal'], flush=True)
