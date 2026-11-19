"""Descriptive tracking from sealed saved states; no fitting or inference."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAV = ROOT / 'CaMRA/navigation'
sys.path.insert(0, str(ROOT / 'CaMRA/calibration/src'))
from camra_calibration.nominal import PlanarState, step_exact

HERE = Path(__file__).resolve().parent
DT = .005
load = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
source = NAV / 'reports/nav043_saved_applicability_audit_v1.json'
assert sha(source) == 'ef070cc069d72cc00fcac61bccd9c819f30dfbec2f0441307bfc1eae83b6a1ef'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
audit = load(source)
rows = []
for trial in audit['rows']:
    for name, digest in trial['source_hashes'].items():
        assert sha(name) == digest, name
    directory = Path(trial['trial'])
    result, steps = load(directory / 'result.json'), load(directory / 'substeps.json')
    initial = result['warmup']['state_at_window_start'][0]
    old_actual = np.array([initial['x'], initial['y']])
    old_yaw = initial['yaw']
    old_reference = old_actual.copy()
    translation = np.zeros(2)
    heading = np.zeros(2)
    forward = np.zeros(2)
    lateral = np.zeros(2)
    actual_turn = 0.
    for step in steps:
        # Exact identity: observed minus nominal displacement equals
        # (observed minus desired at actual heading) plus
        # (desired at actual heading minus desired at nominal heading).
        pose, command = step['state'], step['applied']
        actual = np.array(pose[:2])
        reference = np.array(step['ref_pose'][:2])
        desired_at_actual = step_exact(PlanarState(0., 0., old_yaw), command, DT)
        desired = np.array([desired_at_actual.px, desired_at_actual.py])
        residual = actual - old_actual - desired
        c, s = math.cos(old_yaw), math.sin(old_yaw)
        forward_axis, side_axis = np.array([c, s]), np.array([-s, c])
        forward += np.dot(residual, forward_axis) * forward_axis
        lateral += np.dot(residual, side_axis) * side_axis
        translation += residual
        heading += desired - (reference - old_reference)
        actual_turn += math.atan2(math.sin(pose[2] - old_yaw), math.cos(pose[2] - old_yaw))
        old_actual, old_yaw, old_reference = actual, pose[2], reference
    endpoint_error = old_actual - old_reference
    decomposition_error = float(np.max(np.abs(endpoint_error - translation - heading)))
    axes_error = float(np.max(np.abs(translation - forward - lateral)))
    assert decomposition_error < 1.e-9 and axes_error < 1.e-9
    np.testing.assert_allclose(np.linalg.norm(endpoint_error), trial['terminal']['terminal_position_error_m'], rtol=0., atol=1.e-9)
    commands = np.array([step['applied'] for step in steps])
    measured = np.array([step['state'][3:6] for step in steps])
    positions = np.array([[initial['x'], initial['y']]] + [step['state'][:2] for step in steps])
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    duration = len(steps) * DT
    command_turn = float(commands[:, 2].sum() * DT)
    axis_stats = {}
    for i, axis in enumerate(('vx_mps', 'vy_mps', 'wz_radps')):
        axis_stats[axis] = {'command_mean_signed': float(commands[:, i].mean()),
            'measured_body_mean_signed': float(measured[:, i].mean()),
            'command_mean_abs': float(np.abs(commands[:, i]).mean()),
            'measured_body_mean_abs': float(np.abs(measured[:, i]).mean()),
            'mean_abs_tracking_difference': float(np.abs(measured[:, i] - commands[:, i]).mean())}
    rows.append({'level': trial['level'], 'method': trial['method'], 'kind': trial['trial_kind'],
        'trial': str(directory), 'source_hashes': trial['source_hashes'], 'duration_s': duration,
        'axis_statistics_at_saved_200Hz_post_states': axis_stats,
        'command_path_length_m': float(np.linalg.norm(commands[:, :2], axis=1).sum() * DT),
        'actual_xy_path_length_m': path_length,
        'command_signed_turn_deg': math.degrees(command_turn),
        'actual_unwrapped_signed_yaw_change_deg': math.degrees(actual_turn),
        'actual_body_wz_integral_deg': math.degrees(float(measured[:, 2].sum() * DT)),
        'endpoint_actual_minus_reference_xy_m': endpoint_error.tolist(),
        'endpoint_error_m': float(np.linalg.norm(endpoint_error)),
        'heading_displacement_term_xy_m': heading.tolist(),
        'translation_tracking_term_xy_m': translation.tolist(),
        'forward_translation_term_xy_m': forward.tolist(),
        'lateral_translation_term_xy_m': lateral.tolist(),
        'vector_identity_max_abs_error_m': decomposition_error,
        'translation_axes_identity_max_abs_error_m': axes_error,
        'independent_validation': False})
assert len(rows) == 12
for row in rows:
    if row['kind'] == 'baseline':
        repeat = next(r for r in rows if r['kind'] == 'repeat' and r['method'] == row['method'] and r['level'] == row['level'])
        for key in ('axis_statistics_at_saved_200Hz_post_states', 'endpoint_error_m',
                    'heading_displacement_term_xy_m', 'translation_tracking_term_xy_m'):
            assert row[key] == repeat[key]
report = {'schema': 'camra-nav043-saved-tracking-decomposition-v1',
    'at': datetime.now().astimezone().isoformat(), 'task': 'NAV-043',
    'status': 'DESCRIPTIVE_SAVED_KINEMATICS_NOT_CAUSAL_CALIBRATION_OR_ACCEPTANCE',
    'source_audit_sha256': sha(source), 'analysis_source_sha256': sha(Path(__file__)),
    'rows': rows, 'new_applications': 0, 'new_episodes': 0, 'W_queries': 0,
    'model_fits': 0, 'optimizer_updates': 0, 'final_data_access': False,
    'NAV043_passed': False, 'G5_passed': False,
    'limits': ['Same exposed k=1 traces; no new physical or nominal run.',
        'Body-frame velocities are true 3D body projections. Body wz is not exactly Euler yaw rate under tilt; both are reported, never substituted.',
        'The vector decomposition is an exact geometric accounting identity using recorded heading, not a physical counterfactual or unique causal attribution.',
        'Forward/lateral/heading vectors can cancel; their norms must not be summed or interpreted as causal percentages.',
        'Mean signed tracking values can cancel across commands; absolute values and differences are retained.',
        'No gain fitting, controller tuning, new dynamics, margins or acceptance changes.',
        'Identical repeated records do not provide independent population reliability.']}
out = NAV / 'reports/nav043_saved_tracking_diagnostic_v1.json'
with out.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False))
for row in rows:
    if row['kind'] == 'baseline':
        print(json.dumps(row, ensure_ascii=False))
