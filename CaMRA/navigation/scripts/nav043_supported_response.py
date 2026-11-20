"""Isolated response-v3 candidate, data support and approval-gated diagnostic.

No production consumer. q denotes a completed 0.1s SE(2) increment.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import signal
import socket
import sys
import time

import numpy as np

import nav043_motion_response as base

PROPOSAL = base.NAV/'config/nav043_supported_response_proposal_v3.json'
APPROVAL = base.NAV/'config/nav043_supported_response_approval_v3.json'
READINESS = base.NAV/'config/nav043_supported_response_readiness_v3.json'
HERE = base.ROOT/'.worklog-testing/NAV-043/motion-candidate-v3'
FEATURE_VERSION = 'origin-preserving-hinge-v1'
DOMAIN = np.array([1., .4, 1.])
LIMITS = dict(models=3, axis_solves=9, cpu_wall_s=3600, applications=0,
              new_physical_episodes=0, new_nominal_episodes=0, W_queries=0,
              actor_critic_updates=0, calibration_fits=0, automatic_retries=0)


def features(commands):
    """Continuous fixed basis; no learned knots, intercept or command clipping."""
    u = np.asarray(commands, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3 or not np.isfinite(u).all():
        raise ValueError('finite Nx3 commands required')
    if np.any(np.abs(u) > DOMAIN + 1.e-7):
        raise ValueError('outside frozen float32 command domain')
    hinge = np.sign(u[:, 0]) * np.maximum(np.abs(u[:, 0]) - .1, 0.)
    gate = np.minimum(np.max(np.abs(u / DOMAIN), axis=1) / .2, 1.)
    return np.column_stack((u, hinge, gate))


@dataclass
class Case:
    trace: base.Trace
    cohort: str
    initial_q: np.ndarray

    @property
    def key(self):
        return self.trace.identity['trial']

    def pairs(self):
        t = self.trace
        n = len(t.observed_twists)
        if self.cohort == 'probe':
            previous = np.vstack((self.initial_q, t.observed_twists[:-1]))
            return previous, t.interval_commands[:n], t.observed_twists
        if self.cohort != 'navigation':
            raise ValueError('unknown cohort')
        # The pre-window q was not captured in old navigation: exclude pair 0.
        return t.observed_twists[:-1], t.interval_commands[1:n], t.observed_twists[1:]


def read_probe(folder):
    folder = Path(folder)
    result = base.load(folder/'result.json')
    q = base.load(folder/'completed_q.json')
    with (folder/'actual.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    if (result['case_id'] != folder.name or result['ticks'] != 2200
            or len(rows) != 2200 or q['partial_tail_ticks'] != 0
            or len(q['completed_intervals']) != 101):
        raise ValueError('incomplete controlled case')
    if any(r['physics_tick'] != i + 1 or r['phase'] != ('warmup' if i < 200 else 'measurement')
           for i, r in enumerate(rows)):
        raise ValueError('gapped/reordered probe')
    poses = np.array([[r['state'][k] for k in ('x', 'y', 'yaw')] for r in rows])
    commands = np.array([r['requested_command'] for r in rows[200:]])
    features(commands)
    if not np.allclose(commands, result['requested_command'], rtol=0., atol=1.e-7):
        raise ValueError('probe command changed')
    observed = []
    for i, interval in enumerate(q['completed_intervals']):
        a, b = (180 + i*20, 200 + i*20)
        if (interval['start_tick'] != a or interval['end_tick'] != b
                or interval['state_version'] != 'se2-completed-interval-v1'
                or interval['duration_s'] != .1):
            raise ValueError('different completed interval semantics')
        actual_q = base.interval_twist(poses[a-1], poses[b-1])
        if not np.allclose(actual_q, interval['q_vx_vy_yaw_rate'], atol=1.e-10, rtol=0.):
            raise ValueError('q does not match actual endpoints')
        observed.append(actual_q)
    identity = dict(trial=str(folder), level='controlled', method='fixed_v2', kind=folder.name)
    trace = base.Trace(identity, poses[199], poses[200:], commands,
                       commands[::20], np.asarray(observed[1:]))
    return Case(trace, 'probe', np.asarray(observed[0]))


def load_cases():
    """Read only the previously exposed eighteen navigation and twelve probes."""
    protocol = base.verify_protocol()
    review = base.load(base.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json')
    if not review['case_integrity_passed'] or review['strict_original_completion_gate_passed']:
        raise ValueError('saved data review scope changed')
    index = base.load(base.ROOT/review['raw_file_index'])
    for item in index['files']:
        path = Path(item['path'])
        if path.stat().st_size != item['bytes'] or base.sha(path) != item['sha256']:
            raise ValueError('changed controlled raw input')
    if len(index['files']) != 108:
        raise ValueError('controlled input index changed')
    cases = [Case(base.read_trace(e), 'navigation', np.zeros(3)) for e in protocol['traces']]
    root = Path(review['output_root'])
    if not (root/'HELD.json').is_file() or (root/'FINISHED.json').exists():
        raise ValueError('strict original HELD must be retained')
    cases += [read_probe(root/f'M{i:02}') for i in range(1, 13)]
    if len(cases) != 30 or len({c.key for c in cases}) != 30:
        raise ValueError('unexpected or duplicated source')
    return cases


def folds(cases):
    """Whole-map navigation folds; fixed ordinal-modulo-three probe grouping."""
    if len(cases) != 30 or len({c.key for c in cases}) != 30:
        raise ValueError('thirty distinct complete cases required')
    result = []
    for i, level in enumerate(base.LEVELS):
        held = [c for c in cases if (c.cohort == 'navigation' and c.trace.identity['level'] == level)
                or (c.cohort == 'probe' and (int(c.trace.identity['kind'][1:])-1) % 3 == i)]
        keys = {c.key for c in held}
        train = [c for c in cases if c.key not in keys]
        if ([sum(c.cohort == k for c in train) for k in ('navigation', 'probe')] != [12, 8]
                or [sum(c.cohort == k for c in held) for k in ('navigation', 'probe')] != [6, 4]):
            raise ValueError('invalid whole-case fold')
        result.append((level, train, held))
    return result


def design(cases, axis):
    if axis not in range(3):
        raise ValueError('unknown axis')
    x, y, weights = [], [], []
    for cohort in ('navigation', 'probe'):
        subset = [c for c in cases if c.cohort == cohort]
        if not subset:
            raise ValueError('both cohorts required')
        for case in subset:
            prev, u, target = case.pairs()
            if len(target) == 0:
                raise ValueError('empty fitting case')
            x.append(np.column_stack((prev[:, axis], features(u))))
            y.append(target[:, axis])
            weights.append(np.full(len(target), .5 / len(subset) / len(target)))
    weights = np.concatenate(weights)
    root = np.sqrt(weights)
    return np.concatenate(x)*root[:, None], np.concatenate(y)*root, weights


def rank_support(x):
    singular = np.linalg.svd(x, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0]*1.e-12))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0. else None
    return dict(rank=rank, columns=6, singular_values=singular.tolist(), condition=condition,
                valid=rank == 6 and condition is not None and condition <= 1.e8)


def fit_axis(x, y):
    """One solve. Caller must hold exact approval; no regularization or retry."""
    theta, residuals, rank, singular = np.linalg.lstsq(x, y, rcond=1.e-12)
    valid = (rank == 6 and singular[-1] > 0. and singular[0]/singular[-1] <= 1.e8
             and np.isfinite(theta).all() and 0. <= theta[0] < 1.)
    return dict(theta=theta.tolist(), rank=int(rank), singular_values=singular.tolist(),
                residual_sum=residuals.tolist(), valid=bool(valid))


def free_replay(case, theta, affine=False):
    t = case.trace
    pose, response = t.initial.copy(), case.initial_q.copy()
    poses, responses = [], []
    for i, u in enumerate(t.interval_commands):
        response = (theta[:, 0]*response + theta[:, 1]*u + theta[:, 2] if affine else
                    theta[:, 0]*response + theta[:, 1:] @ features(u[None, :])[0])
        if not np.isfinite(response).all():
            raise ValueError('nonfinite free response')
        responses.append(response.copy())
        for _ in range(min(20, t.ticks-i*20)):
            pose = base.step(pose, response, .005)
            poses.append(pose.copy())
    return np.asarray(poses), np.asarray(responses)


def context():
    if (socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX')
            or Path(sys.executable).resolve() != Path(base.PYTHON).resolve()
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or any(os.environ.get(k) != '1' for k in
                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))):
        raise ValueError('requires tiger11/tmux/registered CPU-only Python and threads=1')


def pin(path):
    if not path.is_file() or base.sha(path) != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('missing or changed pinned registration: '+str(path))
    return base.load(path)


def approved():
    # Check before loading trajectories, creating markers or importing simulator code.
    if not APPROVAL.is_file():
        raise ValueError('response-v3 finite scientific fits require separate APPROVED registration')
    p, a, ready = pin(PROPOSAL), pin(APPROVAL), pin(READINESS)
    if (p['schema'] != 'nav043-supported-response-v3' or p['status'] != 'PROPOSED'
            or p['limits'] != LIMITS or a.get('status') != 'APPROVED'
            or a.get('proposal_sha256') != base.sha(PROPOSAL)
            or a.get('readiness_sha256') != base.sha(READINESS)
            or a.get('limits') != LIMITS or a.get('output_root') != p['output_root']
            or ready['status'] != 'PREPARED_ONLY'
            or ready['proposal_sha256'] != base.sha(PROPOSAL)):
        raise ValueError('approval does not bind exact finite diagnostic and support')
    for group in (p['source_hashes'], ready['source_hashes']):
        for path, digest in group.items():
            if base.sha(path) != digest:
                raise ValueError('changed diagnostic source: '+path)
    return p


def diagnostic_gate(rows):
    def rms(items, key):
        return math.sqrt(sum(r[key]['position_squared_error_sum'] for r in items)
                         / sum(r[key]['ticks'] for r in items))
    nav = [r for r in rows if r['cohort'] == 'navigation']
    probe = [r for r in rows if r['cohort'] == 'probe']
    if len(nav) != 18 or len(probe) != 12:
        raise ValueError('all heldout cases must be reported')
    groups = {}
    for name, subset in [('navigation', nav), ('controlled', probe)] + [
            (level, [r for r in nav if r['level'] == level]) for level in base.LEVELS]:
        groups[name] = dict(candidate_RMSE_m=rms(subset, 'candidate'),
                            affine_RMSE_m=rms(subset, 'affine'))
    checks = {
        'navigation_no_worse_global_and_each_map': all(
            v['candidate_RMSE_m'] <= v['affine_RMSE_m'] + 1.e-10
            for k, v in groups.items() if k != 'controlled'),
        'controlled_RMSE_lower_than_affine': groups['controlled']['candidate_RMSE_m']
                                              < groups['controlled']['affine_RMSE_m'] - 1.e-10,
        'controlled_RMSE_lower_than_ideal': rms(probe, 'candidate') < rms(probe, 'ideal') - 1.e-10,
        'controlled_terminal_heading_no_worse_than_affine': sum(
            abs(r['candidate']['heading_terminal_rad']) for r in probe) <= sum(
            abs(r['affine']['heading_terminal_rad']) for r in probe) + 1.e-10,
        'M01_M02_each_position_RMSE_no_worse_than_ideal': all(
            r['candidate']['position_RMSE_m'] <= r['ideal']['position_RMSE_m'] + 1.e-10
            for r in probe if r['kind'] in ('M01', 'M02'))}
    return dict(checks=checks, groups=groups, diagnostic_pass=all(checks.values()),
                independent_validation=False, deployment_model=False, NAV043_passed=False, G5_passed=False)


def run(output):
    """Invoked only by the durable once-only launcher; no direct CLI fit mode."""
    context()
    p = approved()
    if Path(output) != Path(p['output_root']) or not (HERE/'launch.lock').is_file():
        raise ValueError('requires registered once-only launcher')
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    solves, rows = [], []
    def timeout(*_):
        raise TimeoutError('CPU wall limit reached')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(LIMITS['cpu_wall_s'])
    try:
        base.write_new(output/'STARTED.json', dict(proposal_sha256=base.sha(PROPOSAL),
            approval_sha256=base.sha(APPROVAL), pane=os.environ['TMUX_PANE'], pid=os.getpid()))
        for level, train, held in folds(load_cases()):
            folder = output/level
            folder.mkdir()
            theta = []
            for axis, name in enumerate(base.AXES):
                x, y, weights = design(train, axis)
                ordinal = len(solves)+1
                if ordinal > LIMITS['axis_solves']:
                    raise ValueError('solve allocation exhausted')
                marker = dict(ordinal=ordinal, axis=name, rows=len(y))
                base.write_new(folder/(name+'-solve-started.json'), marker)
                solves.append(dict(level=level, **marker))
                fitted = fit_axis(x, y)
                solves[-1].update(fitted)
                base.write_new(folder/(name+'-solve-result.json'), solves[-1])
                if not fitted['valid']:
                    raise ValueError('invalid/unstable model; no clipping or retry')
                theta.append(fitted['theta'])
            theta = np.asarray(theta)
            base.write_new(folder/'model.json', dict(theta=theta.tolist(),
                training=[c.key for c in train], heldout=[c.key for c in held],
                feature_version=FEATURE_VERSION, deployment_model=False))
            old = base.load(Path(p['affine_output_root'])/level/'model.json')
            affine = np.asarray(old['parameters_a_b_d'])
            for case in held:
                t = case.trace
                row = dict(cohort=case.cohort, **t.identity,
                           initial_q=case.initial_q.tolist(),
                           initialization='actual warmup q' if case.cohort == 'probe' else 'zero; missing historical warmup q')
                for key, params, is_affine in [('candidate', theta, False), ('affine', affine, True)]:
                    poses, responses = free_replay(case, params, affine=is_affine)
                    row[key] = base.metrics(t, poses, responses, t.ticks)
                    row[key+'_prefixes'] = {str(s): base.metrics(t, poses, responses, s*200)
                        for s in (5, 10, 20, 30, 45) if s*200 <= t.ticks}
                    if key == 'candidate':
                        np.savez(folder/(t.identity['kind']+'-'+t.identity['method']+'.npz'),
                                 predicted=poses, responses=responses)
                poses, responses = base.free_replay(t)
                row['ideal'] = base.metrics(t, poses, responses, t.ticks)
                rows.append(row)
                base.write_new(folder/(t.identity['kind']+'-'+t.identity['method']+'.json'), row)
        load_cases()
        approved()
        gate = diagnostic_gate(rows)
        base.write_new(output/'FINISHED.json', dict(status='EXPOSED_DIAGNOSTIC_ONLY',
            solves=solves, rows=rows, gate=gate, cpu_wall_s=time.monotonic()-started,
            limits=LIMITS, extra_all_data_fit=False, calibration_passed=False))
    except BaseException as error:
        base.write_new(output/'HELD.json', dict(error=repr(error), solves=solves,
            heldout_rows=len(rows), cpu_wall_s=time.monotonic()-started, NAV043_passed=False))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
