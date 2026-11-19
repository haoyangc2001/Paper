"""Main-agent verification of saved kinematics and preserved project boundaries."""
import ast
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
load = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
output = HERE / 'verification.json'
assert not output.exists(), 'refuse to overwrite verification'
report_path = NAV / 'reports/nav043_saved_tracking_diagnostic_v1.json'
report = load(report_path)
audit_path = NAV / 'reports/nav043_saved_applicability_audit_v1.json'
assert sha(audit_path) == report['source_audit_sha256'] == 'ef070cc069d72cc00fcac61bccd9c819f30dfbec2f0441307bfc1eae83b6a1ef'
assert sha(HERE / 'analyze.py') == report['analysis_source_sha256']
assert report['status'] == 'DESCRIPTIVE_SAVED_KINEMATICS_NOT_CAUSAL_CALIBRATION_OR_ACCEPTANCE'
assert all(report[k] == 0 for k in ('new_applications', 'new_episodes', 'W_queries', 'model_fits', 'optimizer_updates'))
assert not any(report[k] for k in ('final_data_access', 'NAV043_passed', 'G5_passed'))
audit = load(audit_path)
source_files = set()
max_vector_error = 0.
for row in report['rows']:
    prior = next(r for r in audit['rows'] if r['trial'] == row['trial'])
    assert row['source_hashes'] == prior['source_hashes']
    for name, digest in row['source_hashes'].items():
        assert sha(name) == digest, name
        source_files.add(name)
    trial = Path(row['trial'])
    steps = load(trial / 'substeps.json')
    initial = load(trial / 'result.json')['warmup']['state_at_window_start'][0]
    assert row['duration_s'] == len(steps) * .005
    command = np.array([s['applied'] for s in steps])
    measured = np.array([s['state'][3:6] for s in steps])
    for i, axis in enumerate(('vx_mps', 'vy_mps', 'wz_radps')):
        stats = row['axis_statistics_at_saved_200Hz_post_states'][axis]
        # fsum is a separate accumulation path for the signed means.
        assert math.isclose(stats['command_mean_signed'], math.fsum(command[:, i]) / len(steps), abs_tol=1.e-12)
        assert math.isclose(stats['measured_body_mean_signed'], math.fsum(measured[:, i]) / len(steps), abs_tol=1.e-12)
    expected_turn = math.degrees(math.fsum(command[:, 2]) * .005)
    assert math.isclose(expected_turn, row['command_signed_turn_deg'], abs_tol=1.e-9)
    yaw = np.unwrap([initial['yaw']] + [s['state'][2] for s in steps])
    observed_turn = math.degrees(yaw[-1] - yaw[0])
    assert math.isclose(observed_turn, row['actual_unwrapped_signed_yaw_change_deg'], abs_tol=1.e-9)
    reference_yaw = steps[-1]['ref_pose'][2] - initial['yaw']
    wrapped_turn_difference = math.atan2(math.sin(reference_yaw - math.radians(expected_turn)), math.cos(reference_yaw - math.radians(expected_turn)))
    assert abs(wrapped_turn_difference) < 1.e-9
    endpoint = np.array(steps[-1]['state'][:2]) - np.array(steps[-1]['ref_pose'][:2])
    np.testing.assert_allclose(endpoint, row['endpoint_actual_minus_reference_xy_m'], rtol=0., atol=1.e-9)
    assert math.isclose(float(np.linalg.norm(endpoint)), row['endpoint_error_m'], abs_tol=1.e-9)
    assert math.isclose(row['endpoint_error_m'], prior['terminal']['terminal_position_error_m'], abs_tol=1.e-9)
    vector_error = float(np.max(np.abs(endpoint - np.array(row['translation_tracking_term_xy_m']) - np.array(row['heading_displacement_term_xy_m']))))
    max_vector_error = max(max_vector_error, vector_error)
    assert vector_error < 1.e-9 and row['translation_axes_identity_max_abs_error_m'] < 1.e-9
assert len(report['rows']) == 12 and len(source_files) == 60
for method in ('full', 'uncalibrated'):
    assert len({r['level'] for r in report['rows'] if r['method'] == method and r['kind'] == 'baseline'}) == 3
prereg = load(NAV / 'config/nav066_decision_prereg_v2.json')
protected = {str(NAV / k): v for k, v in prereg['implementation_hashes'].items()}
protected.update(prereg['protected_hashes'])
protected.update({c['checkpoint']: c['checkpoint_sha256'] for c in prereg['six_candidates']})
assert len(protected) == 295
for name, digest in protected.items():
    assert sha(name) == digest, name
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
plan = load(plan_path)
base = json.loads(subprocess.check_output(['git', 'show', '24b0e35:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in base.items() if k != 'tasks'}
assert len(plan['tasks']) == len(base['tasks']) == 67
for old, new in zip(base['tasks'], plan['tasks']):
    if old['id'] == 'NAV-043':
        assert old['work_records'] == new['work_records'][:len(old['work_records'])]
        assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
    else:
        assert old == new
assert sum(t['status'] for t in plan['tasks']) == 55
states = {t['id']: t['status'] for t in plan['tasks']}
assert next(t['id'] for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])) == 'NAV-043'
assert not subprocess.check_output(['git', 'diff', '24b0e35', '--name-only', '--', 'manuscripts', '*.tex', '*.bib', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration', 'CaMRA/navigation/src', 'CaMRA/navigation/config'], cwd=ROOT, text=True).strip()
for script in (HERE / 'analyze.py', Path(__file__)):
    ast.parse(script.read_text())
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(plan_path), '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout + validation.stderr
out = {'at': datetime.now().astimezone().isoformat(), 'status': 'PASS_BASIC_SAVED_KINEMATICS_NOT_NAV043_ACCEPTANCE',
       'report_sha256': sha(report_path), 'analysis_sha256': sha(HERE / 'analyze.py'), 'verifier_sha256': sha(Path(__file__)),
       'raw_source_files_checked': len(source_files), 'protected_identities_checked': len(protected),
       'saved_trials': 12, 'distinct_configurations_per_method': 3,
       'turn_sums_and_saved_reference_checked': True, 'body_velocities_and_Euler_yaw_kept_separate': True,
       'max_vector_identity_error_m': max_vector_error, 'append_only_plan_and_protected_paths_checked': True,
       'validator': validation.stdout, 'completed': 55, 'NAV043_status': False,
       'new_applications': 0, 'new_episodes': 0, 'W_queries': 0, 'fitting_updates': 0,
       'test_subagent_used': False, 'remote_push': False}
with output.open('x') as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(out, ensure_ascii=False, indent=2))
