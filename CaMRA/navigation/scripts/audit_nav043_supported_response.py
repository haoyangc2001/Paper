"""Read-only verification of completed response-v3 output; never fit parameters."""
from datetime import datetime
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

import nav043_supported_response as c

base = c.base
REPORT = base.NAV/'reports/nav043_supported_response_review_v3.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equivalent(expected, saved):
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and expected.keys() == saved.keys(), 'changed fields')
        for key in expected:
            equivalent(expected[key], saved[key])
    elif isinstance(expected, list):
        require(isinstance(saved, list) and len(expected) == len(saved), 'changed list length')
        for a, b in zip(expected, saved):
            equivalent(a, b)
    elif isinstance(expected, float):
        require(isinstance(saved, (int, float)) and not isinstance(saved, bool)
                and math.isfinite(expected) and math.isfinite(saved)
                and math.isclose(expected, saved, rel_tol=1.e-10, abs_tol=1.e-10), 'changed number')
    else:
        require(type(expected) is type(saved) and expected == saved, 'changed saved value')


def verify_solution(x, y, record):
    """Normal equations and spectrum check; SVD computes no response coefficients."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    theta = np.asarray(record['theta'], dtype=float)
    singular = np.asarray(record['singular_values'], dtype=float)
    require(x.ndim == 2 and x.shape[1] == 6 and len(x) > 6
            and y.shape == (len(x),) and np.isfinite(x).all() and np.isfinite(y).all(), 'bad data')
    require(theta.shape == (6,) and np.isfinite(theta).all() and 0. <= theta[0] < 1.
            and record['rows'] == len(y) and record['valid'] is True
            and type(record['rank']) is int and record['rank'] == 6, 'invalid saved solution')
    require(singular.shape == (6,) and np.isfinite(singular).all()
            and singular[-1] > 0. and singular[0]/singular[-1] <= 1.e8, 'invalid spectrum')
    actual_singular = np.linalg.svd(x, compute_uv=False)
    require(np.count_nonzero(actual_singular > actual_singular[0]*1.e-12) == 6
            and np.allclose(actual_singular, singular, rtol=1.e-10, atol=1.e-10), 'changed spectrum')
    residual = x@theta-y
    normal = float(np.max(np.abs(x.T@residual)))
    require(normal <= 1.e-8, 'saved parameters fail normal equations')
    equivalent([float(residual@residual)], record['residual_sum'])
    return dict(rows=len(y), normal_residual_inf=normal,
                residual_squared_sum=float(residual@residual), spectrum_checked=True)


def verify_prediction(path, poses, responses):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'predicted', 'responses'}, 'changed prediction fields')
        for key, expected in [('predicted', poses), ('responses', responses)]:
            actual = saved[key]
            require(actual.shape == expected.shape and np.isfinite(actual).all()
                    and np.allclose(actual, expected, rtol=1.e-10, atol=1.e-10), 'changed '+key)


def replay_row(case, theta, affine):
    t = case.trace
    row = dict(cohort=case.cohort, **t.identity, initial_q=case.initial_q.tolist(),
               initialization='actual warmup q' if case.cohort == 'probe'
               else 'zero; missing historical warmup q')
    candidate_prediction = None
    for key, parameters, is_affine in [('candidate', theta, False), ('affine', affine, True)]:
        poses, responses = c.free_replay(case, parameters, affine=is_affine)
        row[key] = base.metrics(t, poses, responses, t.ticks)
        row[key+'_prefixes'] = {str(s): base.metrics(t, poses, responses, s*200)
                               for s in (5, 10, 20, 30, 45) if s*200 <= t.ticks}
        if key == 'candidate':
            candidate_prediction = poses, responses
    poses, responses = base.free_replay(t)
    row['ideal'] = base.metrics(t, poses, responses, t.ticks)
    return row, candidate_prediction


def expected_files(folds):
    files = {'STARTED.json', 'FINISHED.json'}
    for level, _, held in folds:
        files.add(f'{level}/model.json')
        for axis in base.AXES:
            files.update(f'{level}/{axis}-solve-{suffix}.json' for suffix in ('started', 'result'))
        for case in held:
            stem = case.trace.identity['kind']+'-'+case.trace.identity['method']
            files.update(f'{level}/{stem}.{suffix}' for suffix in ('json', 'npz'))
    require(len(files) == 83, 'colliding or incomplete expected output names')
    return files


def main():
    c.context()
    p = c.approved()
    output = Path(p['output_root'])
    # Refuse before reading fitting data, creating reports or replaying commands.
    require((output/'FINISHED.json').is_file() and not (output/'HELD.json').exists(),
            'no complete response-v3 output to audit')
    require(not REPORT.exists(), 'review already exists; do not overwrite')
    files = sorted(path for path in output.rglob('*') if path.is_file())
    before = {str(path): base.sha(path) for path in files}
    result, started = base.load(output/'FINISHED.json'), base.load(output/'STARTED.json')
    launch, exited = base.load(c.HERE/'launch.json'), base.load(c.HERE/'exit.json')
    command = [base.PYTHON, '-B', '-c',
               'import nav043_supported_response as c; c.run(c.approved()["output_root"])']
    require(exited['exit_code'] == 0 and exited['error'] is None
            and exited['retry_performed'] is False and exited['finished'] is True
            and exited['held'] is False
            and math.isfinite(exited['wrapper_wall_s'])
            and 0. <= exited['wrapper_wall_s'] <= c.LIMITS['cpu_wall_s'], 'failed launch')
    require(launch['proposal_sha256'] == base.sha(c.PROPOSAL)
            and launch['approval_sha256'] == base.sha(c.APPROVAL)
            and launch['output_root'] == str(output) and launch['pane']
            and launch['command'] == command
            and int((c.HERE/'launch.lock').read_text()) == launch['wrapper_pid']
            and type(launch['child_pid']) is int and launch['child_pid'] > 0
            and launch['wrapper_pid'] != launch['child_pid'], 'changed launch identity')
    equivalent(started, dict(proposal_sha256=base.sha(c.PROPOSAL),
        approval_sha256=base.sha(c.APPROVAL), pane=launch['pane'], pid=launch['child_pid']))
    require(result['status'] == 'EXPOSED_DIAGNOSTIC_ONLY'
            and result['limits'] == c.LIMITS and result['extra_all_data_fit'] is False
            and result['calibration_passed'] is False and math.isfinite(result['cpu_wall_s'])
            and 0. <= result['cpu_wall_s'] <= exited['wrapper_wall_s'], 'changed result scope')
    require(len(result['solves']) == 9 and len(result['rows']) == 30
            and [r['ordinal'] for r in result['solves']] == list(range(1, 10)), 'changed counts')
    cases = c.load_cases()
    folds = c.folds(cases)
    require({str(path.relative_to(output)) for path in files} == expected_files(folds),
            'missing or extra output files')
    rows, checks = [], []
    for level, train, held in folds:
        folder = output/level
        model = base.load(folder/'model.json')
        theta = np.asarray(model['theta'], dtype=float)
        require(theta.shape == (3, 6) and np.isfinite(theta).all(), 'invalid model')
        equivalent(model, dict(theta=theta.tolist(), training=[case.key for case in train],
            heldout=[case.key for case in held], feature_version=c.FEATURE_VERSION, deployment_model=False))
        require(set(model['training']).isdisjoint(model['heldout']), 'leaked fold')
        for axis, name in enumerate(base.AXES):
            ordinal = len(checks)+1
            record = base.load(folder/f'{name}-solve-result.json')
            equivalent(record, result['solves'][ordinal-1])
            require(record['level'] == level and record['axis'] == name
                    and record['ordinal'] == ordinal, 'changed solve order')
            equivalent(base.load(folder/f'{name}-solve-started.json'),
                       {k: record[k] for k in ('ordinal', 'axis', 'rows')})
            equivalent(theta[axis].tolist(), record['theta'])
            x, y, _ = c.design(train, axis)
            checks.append(dict(level=level, axis=name, **verify_solution(x, y, record)))
        old = base.load(Path(p['affine_output_root'])/level/'model.json')
        affine = np.asarray(old['parameters_a_b_d'])
        require(affine.shape == (3, 3) and np.isfinite(affine).all(), 'invalid baseline')
        for case in held:
            row, prediction = replay_row(case, theta, affine)
            stem = case.trace.identity['kind']+'-'+case.trace.identity['method']
            equivalent(row, base.load(folder/(stem+'.json')))
            verify_prediction(folder/(stem+'.npz'), *prediction)
            rows.append(row)
    equivalent(rows, result['rows'])
    gate = c.diagnostic_gate(rows)
    equivalent(gate, result['gate'])
    c.load_cases()
    c.approved()
    require(before == {str(path): base.sha(path) for path in output.rglob('*') if path.is_file()},
            'output changed during read-only audit')
    base.write_new(REPORT, dict(status='SAVED_SUPPORTED_RESPONSE_EXECUTION_AUDIT_PASS',
        at=datetime.now().astimezone().isoformat(), host=socket.gethostname(),
        pane=os.environ['TMUX_PANE'], python=sys.executable,
        proposal_sha256=base.sha(c.PROPOSAL), readiness_sha256=base.sha(c.READINESS),
        approval_sha256=base.sha(c.APPROVAL), launch=launch, exit=exited,
        actual_axis_solves=9, extra_response_fits=0, verification_spectra=9,
        heldout_cases=30, saved_measurement_ticks=sum(case.trace.ticks for case in cases),
        cases_reloaded_and_source_pins_rechecked=True, output_unchanged=True,
        normal_equation_checks=checks, rows=rows, diagnostic_gate=gate,
        scientific_diagnostic_pass=gate['diagnostic_pass'],
        output_index=[dict(path=str(path), sha256=before[str(path)], bytes=path.stat().st_size)
                      for path in files],
        new_applications=0, new_physical_episodes=0, new_nominal_episodes=0, W_queries=0,
        actor_critic_updates=0, calibration_fits=0, deployment_model_selected=False,
        ninety_second_evidence=False, NAV043_passed=False, G5_passed=False))


if __name__ == '__main__':
    main()
