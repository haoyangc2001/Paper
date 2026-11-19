"""Describe completed L2 tracking and time slack; no inference or simulation."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
RAW = Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')
OUT = HERE / 'L2-tracking-diagnostic.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not OUT.exists()
sources = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    data = Path(path).read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


snapshot = load(HERE / 'L2-completed-progress.json')
assert len(snapshot['source_hashes']) == 22
for path, digest in snapshot['source_hashes'].items():
    assert sha(path) == digest, path
old = load(ROOT / 'CaMRA/navigation/reports/nav043_saved_tracking_diagnostic_v1.json')
assert old['status'] == 'DESCRIPTIVE_SAVED_KINEMATICS_NOT_CAUSAL_CALIBRATION_OR_ACCEPTANCE'
protocol = load(ROOT / 'CaMRA/navigation/config/nav043_exploration_v1.json')
for path, digest in protocol['sealed_hashes'].items():
    assert sha(path) == digest, path
rows = []
for method in ('uncalibrated', 'full'):
    trial = RAW / 'L2_single_detour' / method
    result = load(trial / 'result.json')
    nominal = load(trial / 'nominal.json')
    steps = load(trial / 'substeps.json')
    nav = load(trial / 'navigation.json')
    decision = load(trial / 'decision.json')
    pin = load(trial / 'decision.pin.json')
    assert sha(trial / 'decision.json') == pin['sha256']
    assert result['z0_J'] == decision['decision']['selected_budget_J'] == nominal['z0_J']
    assert len(steps) == result['physics_steps'] == 9000
    assert len(nav) == result['navigation_steps'] == 450
    assert nominal['initial_state'] == result['warmup']['state_at_window_start'][0]
    initial = nominal['initial_state']
    commands = np.asarray([s['applied'] for s in steps], dtype=float)
    state = np.asarray([s['state'] for s in steps], dtype=float)
    times = np.asarray([s['t_s'] for s in steps], dtype=float)
    np.testing.assert_allclose(times, np.arange(1, 9001) * .005, rtol=0., atol=1.e-8)
    assert [s['physics_tick'] for s in steps] == list(range(1, 9001))
    expected_commands = np.asarray([n['command'][0] for n in nav], dtype=float)
    np.testing.assert_array_equal(commands, np.repeat(expected_commands, 20, axis=0))
    assert [n['physics_tick'] for n in nav] == list(range(0, 9000, 20))
    yaw = np.r_[initial['yaw'], state[:, 2]]
    yaw_diffs = np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw)))
    actual_yaw_turn = float(yaw_diffs.sum())
    terminal_yaw_turn = math.atan2(math.sin(state[-1, 2] - yaw[0]), math.cos(state[-1, 2] - yaw[0]))
    assert abs(math.atan2(math.sin(actual_yaw_turn-terminal_yaw_turn), math.cos(actual_yaw_turn-terminal_yaw_turn))) < 1.e-10
    stats = {}
    for i, axis in enumerate(('vx_mps', 'vy_mps', 'wz_radps')):
        stats[axis] = {
            'command_mean_signed': float(commands[:, i].mean()),
            'actual_body_mean_signed': float(state[:, i+3].mean()),
            'command_mean_abs': float(np.abs(commands[:, i]).mean()),
            'actual_body_mean_abs': float(np.abs(state[:, i+3]).mean()),
            'mean_abs_tracking_difference': float(np.abs(commands[:, i]-state[:, i+3]).mean()),
        }
    xy = np.vstack(([initial['x'], initial['y']], state[:, :2]))
    nominal_xy = np.vstack(([initial['x'], initial['y']], [r['pose'][:2] for r in nominal['trace']]))
    prefixes = {}
    for seconds in (5, 10, 20, 30, 40, 45):
        physical = steps[seconds*200-1]
        n = seconds*10
        prefix = {
            'physical_goal_distance_m': physical['goal_distance_m'],
            'physical_budget_remaining_J': physical['z_J'],
            'same_command_reference_position_error_m': physical['deviation_m'],
            'nominal_complete_to_this_time': n <= len(nominal['trace']),
        }
        if n <= len(nominal['trace']):
            prefix['nominal_goal_distance_m'] = nominal['trace'][n-1]['goal_distance_m']
        else:
            prefix['nominal_missing_reason'] = 'terminated before prefix; no endpoint padding'
        prefixes[str(seconds)] = prefix
    previous = next(r for r in old['rows'] if r['level']=='L2_single_detour' and r['method']==method and r['kind']=='baseline')
    for path, digest in previous['source_hashes'].items():
        assert sha(path) == digest, path
    rows.append({
        'method': method, 'trial': str(trial), 'selected_k': decision['decision']['selected_k'],
        'z0_J': result['z0_J'], 'physical_label': result['label'], 'physical_t_s': 45.,
        'measured_budget_utilization': result['measured_budget_utilization'],
        'nominal_label': nominal['label'], 'nominal_t_s': nominal['t_s'],
        'nominal_slack_to_45s': 45.-nominal['t_s'],
        'axis_statistics_at_saved_200Hz_post_states': stats,
        'command_signed_turn_deg': math.degrees(float(commands[:, 2].sum()*.005)),
        'actual_unwrapped_signed_yaw_change_deg': math.degrees(actual_yaw_turn),
        'actual_body_wz_integral_deg': math.degrees(float(state[:, 5].sum()*.005)),
        'command_xy_path_length_m': float(np.linalg.norm(commands[:, :2], axis=1).sum()*.005),
        'actual_xy_path_length_m': float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        'nominal_closed_loop_xy_path_length_m': float(np.linalg.norm(np.diff(nominal_xy, axis=0), axis=1).sum()),
        'physical_goal_distance_m': result['terminal']['goal_distance_m'],
        'same_command_reference_endpoint_position_error_m': result['terminal']['deviation_m'],
        'prefixes_s': prefixes,
        'historical_k1_descriptive_comparison': {
            'trial': previous['trial'], 'source_hashes': previous['source_hashes'],
            'duration_s': previous['duration_s'],
            'command_signed_turn_deg': previous['command_signed_turn_deg'],
            'actual_unwrapped_signed_yaw_change_deg': previous['actual_unwrapped_signed_yaw_change_deg'],
            'vx_command_mean': previous['axis_statistics_at_saved_200Hz_post_states']['vx_mps']['command_mean_signed'],
            'vx_actual_body_mean': previous['axis_statistics_at_saved_200Hz_post_states']['vx_mps']['measured_body_mean_signed'],
            'causal_budget_effect_identified': False,
        },
    })
for path, digest in sources.items():
    assert sha(path) == digest, path
report = {
    'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
    'pane': os.environ.get('TMUX_PANE'), 'status': 'DESCRIPTIVE_COMPLETED_L2_TRACKING_NOT_FINAL_REVIEW',
    'task': 'NAV-043', 'rows': rows, 'source_hashes': sources,
    'analysis_sha256': sha(__file__), 'new_applications': 0, 'new_episodes': 0,
    'W_queries': 0, 'training_updates': 0, 'model_fits': 0,
    'NAV043_passed': False, 'G5_passed': False,
    'limitations': [
        'Saved online results and kinematics only; full physical-event and applicability audits remain on original waiters.',
        'Single exposed configuration per method; no population reliability or causal budget/parameter attribution.',
        'Same-command reference and separately executed nominal closed loop are different trajectories.',
        'Body-frame wz and unwrapped Euler yaw change differ under tilt and are reported separately.',
        'Signed averages can cancel; absolute averages and tracking differences are retained.',
        'Historical k1 and actual-state selected-budget executions are different runs, not a controlled budget-only experiment.',
        'No fitted gains, hypothetical corrected trajectories, new margins, deadline changes or deployment selection.',
    ],
}
with OUT.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print({'status': report['status'], 'rows': len(rows), 'sources': len(sources)}, flush=True)
for row in rows:
    print(json.dumps(row, ensure_ascii=False), flush=True)
