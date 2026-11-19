"""Audit saved affine outputs without solving, collecting data, or choosing a model."""
from datetime import datetime
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

import nav043_affine_response as affine

base = affine.base
HERE = base.ROOT/'.worklog-testing/NAV-043/affine-response-v2'
REPORT = base.NAV/'reports/nav043_affine_response_review_v2.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equivalent(a, b):
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), 'changed dictionary fields')
        for key in a:
            equivalent(a[key], b[key])
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), 'changed list length')
        for x, y in zip(a, b):
            equivalent(x, y)
    elif isinstance(a, float):
        require(isinstance(b, (int, float)) and not isinstance(b, bool)
                and math.isfinite(a) and math.isfinite(b)
                and math.isclose(a, b, rel_tol=1.e-10, abs_tol=1.e-10), 'changed numeric value')
    else:
        require(type(a) is type(b) and a == b, 'changed saved value')


def verify_solution(x, y, record):
    """Check a saved solution by multiplication only; never run another fit."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    theta = np.asarray([record[k] for k in ('a', 'b', 'd')])
    singular = np.asarray(record['singular_values'])
    require(x.ndim == 2 and x.shape[1] == 3 and y.shape == (len(x),)
            and len(x) > 3 and np.isfinite(x).all() and np.isfinite(y).all(), 'bad fit data')
    require(record['rows'] == len(y) and record['valid'] is True and record['rank'] == 3
            and np.isfinite(theta).all() and 0. <= theta[0] < 1. and theta[1] > 0., 'invalid saved fit')
    require(singular.shape == (3,) and np.isfinite(singular).all()
            and np.all(singular[:-1] >= singular[1:]) and singular[-1] > 0., 'invalid singular values')
    condition = float(singular[0]/singular[-1])
    require(condition <= 1.e8 and math.isclose(condition, record['condition'], rel_tol=1.e-10),
            'changed saved condition')
    # Verify all three spectral moments against the Gram matrix, without SVD.
    gram = x.T@x
    squared = singular**2
    moments = (float(np.trace(gram)),
               float(sum(gram[i, i]*gram[j, j]-gram[i, j]**2
                         for i, j in ((0, 1), (0, 2), (1, 2)))),
               float(gram[0, 0]*(gram[1, 1]*gram[2, 2]-gram[1, 2]*gram[2, 1])
                     - gram[0, 1]*(gram[1, 0]*gram[2, 2]-gram[1, 2]*gram[2, 0])
                     + gram[0, 2]*(gram[1, 0]*gram[2, 1]-gram[1, 1]*gram[2, 0])))
    saved = (float(sum(squared)), float(squared[0]*squared[1]+squared[0]*squared[2]
                                       + squared[1]*squared[2]), float(np.prod(squared)))
    require(all(a > 0. and math.isclose(a, b, rel_tol=1.e-7, abs_tol=1.e-10)
                for a, b in zip(moments, saved)), 'singular values disagree with fit data')
    residual = x@theta-y
    normal = float(np.max(np.abs(x.T@residual)))
    require(normal <= 1.e-8, 'saved parameters fail normal equations')
    equivalent([float(residual@residual)], record['residual_sum'])
    return {'rows': len(y), 'normal_residual_inf': normal,
            'residual_squared_sum': float(residual@residual)}


def replay_row(trace, parameters):
    poses, responses = affine.free_replay(trace, parameters)
    return {**{k: trace.identity[k] for k in ('trial', 'level', 'method', 'kind')},
            'metrics': base.metrics(trace, poses, responses, trace.ticks),
            'prefixes': {str(s): base.metrics(trace, poses, responses, round(s/base.PHYSICS_DT))
                         if round(s/base.PHYSICS_DT) <= trace.ticks else {'complete': False}
                         for s in base.PREFIXES}}


def expected_files(protocol):
    files = {'STARTED.json', 'FINISHED.json'}
    for level in base.LEVELS:
        files.add(f'{level}/model.json')
        for name in base.AXES:
            files.update(f'{level}/{name}-solve-{suffix}.json' for suffix in ('started', 'result'))
        for entry in protocol['traces']:
            if entry['level'] == level:
                files.add(f"{level}/{entry['kind']}-{entry['method']}.json")
    return files


def main():
    require(socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
            and Path(sys.executable).resolve() == Path(base.PYTHON).resolve(),
            'requires tiger11/tmux/registered Python')
    # Approval and a finished output are required before loading any trajectory.
    proposal, approval = affine.approval()
    output = Path(proposal['output_root'])
    require((output/'FINISHED.json').is_file() and not (output/'HELD.json').exists(),
            'no complete affine output to audit')
    protocol = base.verify_protocol()
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
            == expected_files(protocol), 'missing or extra affine output files')
    result, started = base.load(output/'FINISHED.json'), base.load(output/'STARTED.json')
    launch, exit_record = base.load(HERE/'launch.json'), base.load(HERE/'exit.json')
    require(exit_record['exit_code'] == 0 and exit_record['retry_performed'] is False
            and exit_record['finished'] is True and exit_record['held'] is False, 'failed launch')
    require(launch['proposal_sha256'] == base.sha(affine.PROPOSAL)
            and launch['approval_sha256'] == base.sha(affine.APPROVAL)
            and launch['output_root'] == str(output)
            and launch['source_sha256'] == proposal['source_hashes'][
                'CaMRA/navigation/scripts/nav043_affine_response.py']
            and launch['host'].split('.')[0] == 'tiger11' and launch['pane']
            and launch['GPU_visible_devices'] == ''
            and launch['thread_limits'] == dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
            'launch scope changed')
    require(started['proposal_sha256'] == base.sha(affine.PROPOSAL)
            and started['approval_sha256'] == base.sha(affine.APPROVAL), 'start scope changed')
    require(result['proposal_sha256'] == base.sha(affine.PROPOSAL)
            and result['approval'] == approval and 0. <= result['cpu_wall_s'] <= 3600.,
            'result scope or allocation changed')
    for key in ('new_physical_episodes', 'new_nominal_episodes', 'W_queries',
                'actor_critic_updates', 'calibration_fits'):
        require(result[key] == 0, 'unexpected new experiment work')
    require(len(result['solves']) == 9 and [r['ordinal'] for r in result['solves']] == list(range(1, 10))
            and len(result['rows']) == 18, 'changed solve/row counts')
    require(result['status'] == 'EXPOSED_SAVED_AFFINE_RESPONSE_DIAGNOSTIC_ONLY', 'changed diagnostic status')
    traces = [base.read_trace(e) for e in protocol['traces']]
    rows, checks = [], []
    for level in base.LEVELS:
        training = [t for t in traces if t.identity['level'] != level]
        heldout = [t for t in traces if t.identity['level'] == level]
        require(len(training) == 12 and len(heldout) == 6, 'changed whole-map fold')
        model = base.load(output/level/'model.json')
        equivalent(model['training_trials'], [t.identity['trial'] for t in training])
        equivalent(model['heldout_trials'], [t.identity['trial'] for t in heldout])
        require(set(model['training_trials']).isdisjoint(model['heldout_trials'])
                and model['deployment_model'] is False
                and model['initial_response'] == 'zero, v1 matched diagnostic only', 'changed model scope')
        parameters = np.asarray(model['parameters_a_b_d'])
        require(parameters.shape == (3, 3) and np.isfinite(parameters).all(), 'bad model parameters')
        for axis, name in enumerate(base.AXES):
            record = base.load(output/level/f'{name}-solve-result.json')
            matches = [r for r in result['solves'] if r['level'] == level and r['axis'] == name]
            require(len(matches) == 1, 'duplicate or missing solve identity')
            equivalent(record, matches[0])
            start = base.load(output/level/f'{name}-solve-started.json')
            equivalent(start, {k: record[k] for k in ('level', 'axis', 'ordinal', 'rows')})
            equivalent(parameters[axis].tolist(), [record[k] for k in ('a', 'b', 'd')])
            x = np.concatenate([np.column_stack((t.observed_twists[:-1, axis],
                t.interval_commands[1:len(t.observed_twists), axis],
                np.ones(len(t.observed_twists)-1))) for t in training])
            y = np.concatenate([t.observed_twists[1:, axis] for t in training])
            checks.append({'level': level, 'axis': name, **verify_solution(x, y, record)})
        for trace in heldout:
            row = replay_row(trace, parameters)
            equivalent(row, base.load(output/level/f"{trace.identity['kind']}-{trace.identity['method']}.json"))
            matches = [r for r in result['rows'] if r['trial'] == trace.identity['trial']]
            require(len(matches) == 1, 'duplicate or missing heldout identity')
            equivalent(row, matches[0])
            rows.append(row)
    old = base.load(base.NAV/'reports/nav043_motion_response_review_v1.json')
    gate = affine.gate(rows, old)
    equivalent(gate, result['gate'])
    base.verify_protocol()
    affine.approval()
    base.write_new(REPORT, {'status': 'SAVED_AFFINE_EXECUTION_AUDIT_PASS',
        'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
        'pane': os.environ.get('TMUX_PANE'), 'python': sys.executable,
        'proposal_sha256': base.sha(affine.PROPOSAL), 'launch': launch, 'exit': exit_record,
        'actual_axis_solves': 9, 'extra_verification_fits': 0, 'heldout_rows': 18,
        'saved_measurement_ticks': sum(t.ticks for t in traces),
        'protected_identities_checked': 295, 'raw_input_hashes_checked': 90,
        'normal_equation_checks': checks, 'gate': gate, 'rows': rows,
        'output_index': [{'path': str(p), 'sha256': base.sha(p), 'bytes': p.stat().st_size}
                         for p in sorted(output.rglob('*.json'))],
        'deployment_model_selected': False, 'ninety_second_evidence': False,
        'NAV043_passed': False, 'G5_passed': False})


if __name__ == '__main__':
    main()
