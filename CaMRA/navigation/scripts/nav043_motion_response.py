"""Isolated, approval-gated saved-command response probe; never deploys a model."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parents[1]
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
PROTOCOL = NAV / 'config/nav043_motion_response_proposal_v1.json'
APPROVAL = NAV / 'config/nav043_motion_response_approval_v1.json'
PHYSICS_DT = .005
NAV_TICKS = 20
PREFIXES = (5, 10, 20, 30, 45)
LEVELS = ('L1_empty', 'L2_single_detour', 'L3_multi_path')
AXES = ('vx', 'vy', 'yaw_rate')


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def ab(theta):
    if abs(theta) < 1.e-6:
        return 1. - theta**2 / 6. + theta**4 / 120., theta / 2. - theta**3 / 24.
    return math.sin(theta) / theta, 2. * math.sin(theta / 2.)**2 / theta


def step(pose, twist, dt):
    x, y, yaw = pose
    vx, vy, rate = twist
    a, b = ab(float(rate * dt))
    dx, dy = dt * (a * vx - b * vy), dt * (b * vx + a * vy)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * dx - s * dy, y + s * dx + c * dy, yaw + rate * dt])


def interval_twist(start, end, dt=.1):
    """SE(2) log of the actual pose increment, not body angular velocity."""
    theta = float(wrap(end[2] - start[2]))
    if abs(theta) >= math.pi / 2.:
        raise ValueError('ambiguous large planar rotation in one navigation interval')
    c, s = math.cos(start[2]), math.sin(start[2])
    dx, dy = end[:2] - start[:2]
    bx, by = c * dx + s * dy, -s * dx + c * dy
    a, b = ab(theta)
    denominator = dt * (a * a + b * b)
    return np.array([(a * bx + b * by) / denominator,
                     (-b * bx + a * by) / denominator, theta / dt])


@dataclass
class Trace:
    identity: dict
    initial: np.ndarray
    actual: np.ndarray
    commands: np.ndarray
    interval_commands: np.ndarray
    observed_twists: np.ndarray

    @property
    def ticks(self):
        return len(self.actual)


def read_trace(entry):
    trial = Path(entry['trial'])
    result, rows, nav = (load(trial / n) for n in
                         ('result.json', 'substeps.json', 'navigation.json'))
    if result['method'] != entry['method'] or result['map_id'] != entry['map_id']:
        raise ValueError('method/map identity mismatch')
    n = result['physics_steps']
    if n != len(rows) or n < 2 * NAV_TICKS or len(nav) != (n + NAV_TICKS - 1) // NAV_TICKS:
        raise ValueError('missing physics/navigation intervals')
    if result['warmup']['physics_steps'] != 200 or rows[-1] != result['terminal']:
        raise ValueError('warmup/terminal mismatch')
    if abs(result['t_s'] - n * PHYSICS_DT) > 1.e-8:
        raise ValueError('duration mismatch')
    initial = result['warmup']['state_at_window_start'][0]
    initial = np.array([initial[k] for k in ('x', 'y', 'yaw')], dtype=float)
    actual = np.asarray([r['state'][:3] for r in rows], dtype=float)
    commands = np.asarray([r['applied'] for r in rows], dtype=float)
    interval_commands = np.asarray([r['command'][0] for r in nav], dtype=float)
    if actual.shape != (n, 3) or commands.shape != (n, 3):
        raise ValueError('invalid saved shape')
    if not all(np.isfinite(x).all() for x in (initial, actual, commands, interval_commands)):
        raise ValueError('nonfinite saved data')
    if np.any(np.abs(commands) > np.array([1., .4, 1.]) + 1.e-7):
        raise ValueError('outside frozen command domain')
    for i, row in enumerate(rows):
        if row['physics_tick'] != i + 1 or row['env_id'] != 0:
            raise ValueError('physics tick/environment mismatch')
        if abs(row['t_s'] - (i + 1) * PHYSICS_DT) > 1.e-8:
            raise ValueError('physics clock mismatch')
        if not np.allclose(commands[i], interval_commands[i // NAV_TICKS], atol=1.e-7, rtol=0.):
            raise ValueError('command not held for its genuine navigation interval')
    if any(r['physics_tick'] != i * NAV_TICKS or r['navigation_step'] != i
           for i, r in enumerate(nav)):
        raise ValueError('navigation boundary mismatch')
    # First full interval supplies the previous response only in fitting.
    # Its missing pre-window response is never inferred or padded.
    endpoints = np.vstack([initial, actual[NAV_TICKS - 1::NAV_TICKS]])
    twists = np.asarray([interval_twist(a, b) for a, b in
                         zip(endpoints[:-1], endpoints[1:])])
    return Trace(entry, initial, actual, commands, interval_commands, twists)


def protected_identities(protocol):
    registry = protocol['protected_registry']
    if sha(registry['path']) != registry['sha256']:
        raise ValueError('changed protected identity registry')
    pair = load(registry['path'])
    protected = {str(NAV / k): v for k, v in pair['implementation_hashes'].items()}
    protected.update(pair['protected_hashes'])
    protected.update({c['checkpoint']: c['checkpoint_sha256'] for c in pair['six_candidates']})
    if len(protected) != 295:
        raise ValueError('protected identity count changed')
    return protected


def verify_protocol():
    pin = Path(str(PROTOCOL) + '.sha256')
    if not pin.is_file() or sha(PROTOCOL) != pin.read_text().strip():
        raise ValueError('proposal identity/pin mismatch')
    protocol = load(PROTOCOL)
    if protocol['status'] != 'PROPOSED' or protocol['schema'] != 'nav043-motion-response-v1':
        raise ValueError('invalid proposal')
    if protocol['limits'] != {'models': 3, 'axis_solves': 9, 'cpu_wall_s': 3600,
                              'new_physical_episodes': 0, 'new_nominal_episodes': 0,
                              'W_queries': 0, 'actor_critic_updates': 0, 'calibration_fits': 0}:
        raise ValueError('unregistered resource limits')
    for group in ('report_hashes', 'implementation_hashes'):
        for path, digest in protocol[group].items():
            if sha(path) != digest:
                raise ValueError(f'changed {group}: {path}')
    for path, digest in protected_identities(protocol).items():
        if sha(path) != digest:
            raise ValueError(f'changed protected source: {path}')
    entries = protocol['traces']
    if len(entries) != 18 or len({e['trial'] for e in entries}) != 18:
        raise ValueError('must retain the same eighteen whole traces')
    for level in LEVELS:
        subset = [e for e in entries if e['level'] == level]
        if len(subset) != 6 or {e['method'] for e in subset} != {'full', 'uncalibrated'}:
            raise ValueError('incorrect map grouping')
        if len({e['map_id'] for e in subset}) != 1:
            raise ValueError('mixed map identity')
        for method in ('full', 'uncalibrated'):
            if {e['kind'] for e in subset if e['method'] == method} != {'baseline', 'repeat', 'W_selected'}:
                raise ValueError('missing original, repeat or W-selected trace')
    for entry in entries:
        if not entry['trial'].startswith('/pub/data/caohy/work/camra_navigation/') or 'final_' in entry['trial']:
            raise ValueError('data scope mismatch')
        for path, digest in entry['hashes'].items():
            if Path(path).parent != Path(entry['trial']) or sha(path) != digest:
                raise ValueError(f'changed trace: {path}')
    return protocol


def require_approval(protocol):
    if not APPROVAL.is_file():
        raise ValueError('formal response-model fitting is not approved')
    pin = Path(str(APPROVAL) + '.sha256')
    if not pin.is_file() or sha(APPROVAL) != pin.read_text().strip():
        raise ValueError('approval pin missing or changed')
    approval = load(APPROVAL)
    if (approval.get('status') != 'APPROVED' or approval.get('protocol_sha256') != sha(PROTOCOL)
            or approval.get('limits') != protocol['limits']
            or approval.get('output_root') != protocol['output_root']):
        raise ValueError('approval does not authorize this exact proposal')
    return approval


def fit_axis(x, y):
    """Exactly one two-variable least-squares solve; no retries or clipping."""
    theta, residuals, rank, singular = np.linalg.lstsq(x, y, rcond=1.e-12)
    condition = float(singular[0] / singular[-1]) if len(singular) == 2 and singular[-1] > 0. else None
    valid = (rank == 2 and condition is not None and condition <= 1.e8
             and np.isfinite(theta).all() and 0. <= theta[0] < 1. and theta[1] > 0.)
    return {'a': float(theta[0]), 'b': float(theta[1]), 'rank': int(rank),
            'singular_values': singular.tolist(), 'condition': condition,
            'valid': bool(valid), 'residual_sum': residuals.tolist(),
            'steady_gain': float(theta[1] / (1. - theta[0])) if theta[0] != 1. else None}


def free_replay(trace, parameters=None):
    """Initial pose aligned once; response starts at zero in every heldout trace."""
    pose, response = trace.initial.copy(), np.zeros(3)
    poses, responses = [], []
    for i, command in enumerate(trace.interval_commands):
        if parameters is None:
            response = command
        else:
            response = parameters[:, 0] * response + parameters[:, 1] * command
        responses.append(response.copy())
        for _ in range(min(NAV_TICKS, trace.ticks - i * NAV_TICKS)):
            pose = step(pose, response, PHYSICS_DT)
            poses.append(pose.copy())
    return np.asarray(poses), np.asarray(responses)


def metrics(trace, predicted, responses, ticks):
    position_error = np.linalg.norm(predicted[:ticks, :2] - trace.actual[:ticks, :2], axis=1)
    heading = wrap(predicted[:ticks, 2] - trace.actual[:ticks, 2])
    full = ticks // NAV_TICKS
    velocity_error = responses[:full] - trace.observed_twists[:full]
    return {'ticks': ticks, 'duration_s': ticks * PHYSICS_DT,
            'position_squared_error_sum': float(position_error @ position_error),
            'position_RMSE_m': float(np.sqrt(np.mean(position_error**2))),
            'position_max_m': float(position_error.max()), 'position_terminal_m': float(position_error[-1]),
            'heading_RMSE_rad': float(np.sqrt(np.mean(heading**2))),
            'heading_terminal_rad': float(heading[-1]),
            'interval_twist_RMSE_vx_vy_yaw_rate': np.sqrt(np.mean(velocity_error**2, axis=0)).tolist()
            if full else None, 'full_velocity_intervals': full}


def compare_trace(trace, parameters):
    ideal, ideal_responses = free_replay(trace)
    predicted, responses = free_replay(trace, parameters)
    def compare(ticks):
        return {'ideal': metrics(trace, ideal, ideal_responses, ticks),
                'response': metrics(trace, predicted, responses, ticks)}
    prefixes = {}
    for seconds in PREFIXES:
        ticks = round(seconds / PHYSICS_DT)
        prefixes[str(seconds)] = {'complete': True, **compare(ticks)} if ticks <= trace.ticks else {
            'complete': False, 'available_duration_s': trace.ticks * PHYSICS_DT}
    return {'trial': trace.identity['trial'], 'method': trace.identity['method'],
            'level': trace.identity['level'], 'kind': trace.identity['kind'],
            'terminal': compare(trace.ticks), 'prefixes': prefixes}


def grouped_gate(rows):
    def rms(subset, kind):
        m = [r['terminal'][kind] for r in subset]
        return math.sqrt(sum(r['position_squared_error_sum'] for r in m) / sum(r['ticks'] for r in m))
    groups = {}
    for level in LEVELS:
        subset = [r for r in rows if r['level'] == level]
        if len(subset) != 6:
            raise ValueError('incomplete heldout map; no dropping failed traces')
        ideal, response = rms(subset, 'ideal'), rms(subset, 'response')
        groups[level] = {'ideal_RMSE_m': ideal, 'response_RMSE_m': response,
                         'not_worse': response <= ideal + 1.e-10}
    ideal, response = rms(rows, 'ideal'), rms(rows, 'response')
    return {'by_heldout_map': groups, 'all_heldout_ideal_RMSE_m': ideal,
            'all_heldout_response_RMSE_m': response,
            'diagnostic_pass': all(g['not_worse'] for g in groups.values()) and response < ideal - 1.e-10,
            'independent_unseen_validation': False, 'NAV043_passed': False, 'G5_passed': False}


def run_probe(protocol, process_started):
    approval = require_approval(protocol)
    output = Path(protocol['output_root'])
    output.mkdir(parents=False, exist_ok=False)
    started = process_started
    solves, rows = [], []
    def deadline(*_):
        raise TimeoutError('registered CPU wall limit reached')
    def check_time():
        if time.monotonic() - started >= protocol['limits']['cpu_wall_s']:
            deadline()
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1, math.ceil(protocol['limits']['cpu_wall_s'] - (time.monotonic() - started))))
    write_new(output / 'STARTED.json', {'at': datetime.now().astimezone().isoformat(),
              'protocol_sha256': sha(PROTOCOL), 'approval_sha256': sha(APPROVAL),
              'host': socket.gethostname(), 'pane': os.environ.get('TMUX_PANE')})
    try:
        check_time()
        traces = []
        for entry in protocol['traces']:
            check_time()
            traces.append(read_trace(entry))
        for level in LEVELS:
            training = [t for t in traces if t.identity['level'] != level]
            heldout = [t for t in traces if t.identity['level'] == level]
            if len(training) != 12 or len(heldout) != 6:
                raise ValueError('invalid whole-map fold')
            fold = output / level
            fold.mkdir()
            parameters = []
            for axis, name in enumerate(AXES):
                check_time()
                x = np.concatenate([np.column_stack([t.observed_twists[:-1, axis],
                      t.interval_commands[1:len(t.observed_twists), axis]]) for t in training])
                y = np.concatenate([t.observed_twists[1:, axis] for t in training])
                marker = fold / (name + '-solve-started.json')
                write_new(marker, {'axis': name, 'ordinal': len(solves) + 1, 'rows': len(y)})
                item = {'level': level, 'axis': name, 'ordinal': len(solves) + 1}
                solves.append(item)
                if len(solves) > protocol['limits']['axis_solves']:
                    raise ValueError('solve allocation exhausted')
                fitted = fit_axis(x, y)
                item.update(fitted)
                write_new(fold / (name + '-solve-result.json'), item)
                check_time()
                if not fitted['valid']:
                    raise ValueError('unidentifiable, ill-conditioned or non-stable/nonpositive response')
                parameters.append([fitted['a'], fitted['b']])
            parameters = np.asarray(parameters)
            write_new(fold / 'model.json', {'parameters_a_b': parameters.tolist(),
                      'training_trials': [t.identity['trial'] for t in training],
                      'heldout_trials': [t.identity['trial'] for t in heldout],
                      'deployment_model': False})
            for trace in heldout:
                check_time()
                row = compare_trace(trace, parameters)
                rows.append(row)
                write_new(fold / f"{trace.identity['kind']}-{trace.identity['method']}.json", row)
        check_time()
        # Check unchanged inputs again before recording any diagnostic result.
        verify_protocol()
        gate = grouped_gate(rows)
        write_new(output / 'FINISHED.json', {'status': 'SAVED_RESPONSE_DIAGNOSTIC_ONLY',
                  'solves': solves, 'rows': rows, 'gate': gate,
                  'cpu_wall_s': time.monotonic() - started, 'protocol_sha256': sha(PROTOCOL),
                  'approval': approval, 'new_physical_episodes': 0, 'new_nominal_episodes': 0,
                  'W_queries': 0, 'actor_critic_updates': 0, 'calibration_fits': 0})
    except BaseException as error:
        write_new(output / 'HELD.json', {'error': repr(error), 'solves': solves,
                  'heldout_rows_completed': len(rows), 'cpu_wall_s': time.monotonic() - started,
                  'NAV043_passed': False, 'G5_passed': False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main():
    process_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', type=Path)
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX') or sys.executable != PYTHON:
        raise ValueError('requires tiger11/tmux/registered absolute interpreter')
    protocol = verify_protocol()
    # Refuse unapproved execution before constructing regression data.
    if args.run:
        require_approval(protocol)
        if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
            raise ValueError('requires three registered CPU thread limits of one')
    if args.preflight:
        if args.preflight.parent.resolve() != ROOT / '.worklog-testing/NAV-043/motion-response-v1':
            raise ValueError('preflight output must use the isolated preparation directory')
        traces = [read_trace(entry) for entry in protocol['traces']]
        write_new(args.preflight, {'status': 'PREFLIGHT_ONLY_NO_MODEL_FITS',
                  'protocol_sha256': sha(PROTOCOL), 'host': socket.gethostname(),
                  'pane': os.environ.get('TMUX_PANE'), 'traces': [
                      {'trial': t.identity['trial'], 'level': t.identity['level'],
                       'physics_ticks': t.ticks, 'full_intervals': len(t.observed_twists),
                       'fit_pairs': len(t.observed_twists) - 1,
                       'partial_ticks': t.ticks % NAV_TICKS} for t in traces],
                  'protected_hashes_checked': len(protected_identities(protocol)),
                  'actual_response_model_solves': 0, 'new_physical_episodes': 0,
                  'new_nominal_episodes': 0, 'W_queries': 0, 'NAV043_passed': False, 'G5_passed': False})
    else:
        run_probe(protocol, process_started)


if __name__ == '__main__':
    main()
