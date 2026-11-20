"""One finite, approval-gated, matched-data diagnostic. No production consumer.

All twelve exposed probes are TRAINING data in every fold. Only the six
navigation cases on the omitted map are held out. This is a new diagnostic,
never a repair of the old gate or an independent validation of probe accuracy.
"""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import numpy as np
import nav043_coupled_response as coupled
import nav043_supported_response as diagonal
import audit_nav043_coupled_response as prior_audit

base = coupled.base
context = diagonal.context
HERE = base.ROOT/'.worklog-testing/NAV-043/matched-support-v5'
PROPOSAL = base.NAV/'config/nav043_matched_support_proposal_v5.json'
READINESS = base.NAV/'config/nav043_matched_support_readiness_v5.json'
APPROVAL = base.NAV/'config/nav043_matched_support_approval_v5.json'
REVIEW = base.NAV/'reports/nav043_matched_support_review_v5.json'
LIMITS = dict(models=6, axis_solves=18, saved_case_predictions=108, cpu_wall_s=3600,
              applications=0, new_physical_episodes=0, new_nominal_episodes=0,
              W_queries=0, actor_critic_updates=0, calibration_fits=0,
              automatic_retries=0)
KINDS = ('diagonal_v3', 'coupled_v4')


def approved():
    # No input loading, output creation or fit before exact approval.
    if not APPROVAL.is_file():
        raise ValueError('matched-support-v5 requires separate APPROVED registration')
    p, ready, approval = [diagonal.pin(path) for path in (PROPOSAL, READINESS, APPROVAL)]
    if (p.get('schema') != 'nav043-matched-support-v5' or p.get('status') != 'PROPOSED'
            or p.get('limits') != LIMITS or ready.get('status') != 'PREPARED_ONLY'
            or ready.get('proposal_sha256') != base.sha(PROPOSAL)
            or approval.get('status') != 'APPROVED'
            or approval.get('proposal_sha256') != base.sha(PROPOSAL)
            or approval.get('readiness_sha256') != base.sha(READINESS)
            or approval.get('limits') != LIMITS
            or approval.get('output_root') != p['output_root']):
        raise ValueError('approval does not bind the exact new diagnostic')
    for mapping in (p['source_hashes'], ready['source_hashes']):
        for path, digest in mapping.items():
            if base.sha(path) != digest:
                raise ValueError('changed source: '+path)
    return p


def partitions(cases):
    if len(cases) != 30 or len({c.key for c in cases}) != 30:
        raise ValueError('exactly thirty distinct exposed cases required')
    probes = [c for c in cases if c.cohort == 'probe']
    if len(probes) != 12 or {c.trace.identity['kind'] for c in probes} != {
            f'M{i:02}' for i in range(1, 13)}:
        raise ValueError('all twelve known probes must be training data')
    result = []
    for level in base.LEVELS:
        train = [c for c in cases if c.cohort == 'probe'
                 or c.trace.identity['level'] != level]
        held = [c for c in cases if c.cohort == 'navigation'
                and c.trace.identity['level'] == level]
        if (len(train) != 24 or len(held) != 6
                or {c.key for c in train} & {c.key for c in held}
                or any(c.cohort != 'navigation' for c in held)
                or len([c for c in train if c.cohort == 'probe']) != 12):
            raise ValueError('invalid matched training / navigation-held split')
        result.append((level, train, held, probes))
    return result


def width(kind):
    if kind not in KINDS:
        raise ValueError('unknown fixed structure')
    return 6 if kind == 'diagonal_v3' else 11


def design(cases, kind, axis):
    width(kind)
    if kind == 'diagonal_v3':
        return diagonal.design(cases, axis)
    x, y, weights = coupled.design(cases)
    return x, y[:, axis], weights


def check_model(theta, kind):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3, width(kind)) or not np.isfinite(theta).all():
        raise ValueError('incorrect saved model shape')
    if kind == 'coupled_v4':
        return coupled.check_model(theta)
    if np.any(theta[:, 0] < 0.) or np.any(theta[:, 0] >= 1.):
        raise ValueError('unstable diagonal memory; no clipping or retry')
    return dict(diagonal_memory=theta[:, 0].tolist(),
                spectral_radius=float(np.max(theta[:, 0])))


def replay(case, theta, kind):
    check_model(theta, kind)
    return (diagonal.free_replay(case, theta) if kind == 'diagonal_v3'
            else coupled.free_replay(case, theta))


def row(case, theta, kind, role):
    if role not in ('held_navigation', 'in_sample_probe'):
        raise ValueError('explicit training/evaluation role required')
    if (role == 'in_sample_probe') != (case.cohort == 'probe'):
        raise ValueError('probe training status must remain explicit')
    t = case.trace
    poses, responses = replay(case, theta, kind)
    value = dict(**t.identity, cohort=case.cohort, structure=kind, evaluation_role=role,
        independent_validation=False,
        initialization='actual warmup q' if case.cohort == 'probe'
        else 'zero; missing historical warmup q',
        metrics=base.metrics(t, poses, responses, t.ticks),
        prefixes={str(s):base.metrics(t, poses, responses, s*200)
                  for s in (5, 10, 20, 30, 45) if s*200 <= t.ticks})
    return value, poses, responses


def summaries(rows):
    # Descriptive, never a replacement acceptance gate or automatic selection.
    result = {}
    for kind in KINDS:
        result[kind] = {}
        for role in ('held_navigation', 'in_sample_probe'):
            subset = [r for r in rows if r['structure'] == kind and r['evaluation_role'] == role]
            expected = 18 if role == 'held_navigation' else 36
            if len(subset) != expected:
                raise ValueError('all diagnostic rows required')
            groups = {'all':subset}
            if role == 'held_navigation':
                groups.update({level:[r for r in subset if r['level'] == level]
                               for level in base.LEVELS})
            result[kind][role] = {name:dict(rows=len(group),
                position_RMSE_m=float(np.sqrt(sum(r['metrics']['position_squared_error_sum'] for r in group)
                                       /sum(r['metrics']['ticks'] for r in group))),
                terminal_heading_MAE_deg=float(np.degrees(np.mean([
                    abs(r['metrics']['heading_terminal_rad']) for r in group]))))
                for name, group in groups.items()}
    return result


def child_command():
    return [base.PYTHON, '-B', '-c', 'import nav043_matched_support_v5 as m; m.run()']


def run():
    context()
    p = approved()
    if not (HERE/'launch.lock').is_file():
        raise ValueError('requires durable once-only launcher')
    output = Path(p['output_root'])
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    solves, rows = [], []
    def timeout(*_):
        raise TimeoutError('finite CPU allocation exhausted')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(LIMITS['cpu_wall_s'])
    try:
        base.write_new(output/'STARTED.json', dict(pid=os.getpid(), pane=os.environ['TMUX_PANE'],
            proposal_sha256=base.sha(PROPOSAL), approval_sha256=base.sha(APPROVAL)))
        for level, train, held, probes in partitions(diagonal.load_cases()):
            for kind in KINDS:
                folder = output/level/kind
                folder.mkdir(parents=True)
                theta = []
                for axis, name in enumerate(base.AXES):
                    ordinal = len(solves)+1
                    if ordinal > LIMITS['axis_solves']:
                        raise ValueError('no extra solves')
                    x, y, _ = design(train, kind, axis)
                    record = dict(level=level, structure=kind, axis=name, ordinal=ordinal, rows=len(y))
                    base.write_new(folder/(name+'-solve-started.json'), record)
                    solves.append(record)
                    fit = (diagonal.fit_axis(x, y) if kind == 'diagonal_v3'
                           else coupled.solve_axis(x, y))
                    record.update(fit)
                    base.write_new(folder/(name+'-solve-result.json'), record)
                    if not fit['valid']:
                        raise ValueError('invalid fit; retain and stop')
                    theta.append(fit['theta'])
                theta = np.asarray(theta)
                stability = check_model(theta, kind)
                base.write_new(folder/'model.json', dict(theta=theta.tolist(), stability=stability,
                    training=[c.key for c in train], held_navigation=[c.key for c in held],
                    in_sample_probes=[c.key for c in probes], deployment_model=False))
                for case in held+probes:
                    role = 'in_sample_probe' if case.cohort == 'probe' else 'held_navigation'
                    value, poses, responses = row(case, theta, kind, role)
                    stem = case.trace.identity['kind']+'-'+case.trace.identity['method']
                    np.savez(folder/(stem+'.npz'), predicted=poses, responses=responses)
                    base.write_new(folder/(stem+'.json'), value)
                    rows.append(dict(fold=level, **value))
        approved()
        diagonal.load_cases()
        base.write_new(output/'FINISHED.json', dict(status='EXPOSED_MATCHED_DATA_DIAGNOSTIC_ONLY',
            limits=LIMITS, cpu_wall_s=time.monotonic()-began, solves=solves, rows=rows,
            descriptive_summary=summaries(rows), old_gates_unchanged=True,
            new_validation_gate_defined=False, model_selected=False, NAV043_passed=False,
            G5_passed=False, additional_all_data_fit=False))
    except BaseException as error:
        base.write_new(output/'HELD.json', dict(error=repr(error), solves=solves,
            saved_rows=len(rows), cpu_wall_s=time.monotonic()-began))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def launch():
    context()
    p = approved()
    output = Path(p['output_root'])
    if output.exists() or (HERE/'launch.lock').exists():
        raise ValueError('already attempted; no retry or alternate root')
    preflight = base.load(HERE/'preflight.json')
    if (preflight.get('status') != 'APPROVED_MATCHED_SUPPORT_PREFLIGHT_PASS'
            or preflight.get('proposal_sha256') != base.sha(PROPOSAL)
            or preflight.get('approval_sha256') != base.sha(APPROVAL)
            or preflight.get('readiness_sha256') != base.sha(READINESS)):
        raise ValueError('matching approved preflight required before launch')
    with (HERE/'launch.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    began, process, code, error = time.monotonic(), None, -1, None
    env = os.environ.copy()
    env['PYTHONPATH'] = str(base.NAV/'scripts')
    command = child_command()
    try:
        with (HERE/'run.log').open('x') as log:
            process = subprocess.Popen(command, cwd=base.ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            base.write_new(HERE/'launch.json', dict(wrapper_pid=os.getpid(), child_pid=process.pid,
                command=command, pane=os.environ['TMUX_PANE'], output_root=str(output),
                proposal_sha256=base.sha(PROPOSAL), approval_sha256=base.sha(APPROVAL)))
            code = process.wait(timeout=LIMITS['cpu_wall_s'])
        if code == 0 and (not (output/'FINISHED.json').is_file() or (output/'HELD.json').exists()):
            raise ValueError('missing complete child evidence')
    except BaseException as exception:
        error = repr(exception)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        code = -1 if process is None else process.returncode
        if code == 0:
            code = -1
    finally:
        base.write_new(HERE/'exit.json', dict(exit_code=code, error=error,
            wrapper_wall_s=time.monotonic()-began, finished=(output/'FINISHED.json').is_file(),
            held=(output/'HELD.json').is_file(), retry_performed=False))
    return code


def verify_solution(x, y, record, kind):
    n = width(kind)
    theta = np.asarray(record['theta'])
    singular = np.linalg.svd(x, compute_uv=False)
    if (theta.shape != (n,) or record['rows'] != len(y) or record['rank'] != n
            or record['valid'] is not True or not np.isfinite(theta).all()
            or singular[-1] <= 0 or singular[0]/singular[-1] > 1.e8
            or not np.allclose(singular, record['singular_values'], rtol=1.e-10, atol=1.e-10)):
        raise ValueError('bad saved solution / spectrum')
    residual = x@theta-y
    normal = float(np.max(np.abs(x.T@residual)))
    if normal > 1.e-8:
        raise ValueError('saved normal equations fail')
    prior_audit.equivalent([float(residual@residual)], record['residual_sum'])
    return dict(normal_residual_inf=normal, rows=len(y))


def audit():
    context()
    p = approved()
    output = Path(p['output_root'])
    if not (output/'FINISHED.json').is_file() or (output/'HELD.json').exists() or REVIEW.exists():
        raise ValueError('complete unreviewed output required; no overwrite')
    files = {str(f.relative_to(output)):base.sha(f) for f in output.rglob('*') if f.is_file()}
    result = base.load(output/'FINISHED.json')
    start = base.load(output/'STARTED.json')
    launched, exited = base.load(HERE/'launch.json'), base.load(HERE/'exit.json')
    if (exited['exit_code'] != 0 or exited['error'] is not None or exited['retry_performed']
            or not exited['finished'] or exited['held']
            or not 0 <= result['cpu_wall_s'] <= exited['wrapper_wall_s'] <= LIMITS['cpu_wall_s']):
        raise ValueError('execution ledger failed')
    prior_audit.equivalent(launched, dict(wrapper_pid=int((HERE/'launch.lock').read_text()),
        child_pid=start['pid'], command=child_command(), pane=start['pane'], output_root=str(output),
        proposal_sha256=base.sha(PROPOSAL), approval_sha256=base.sha(APPROVAL)))
    prior_audit.equivalent(start, dict(pid=launched['child_pid'], pane=launched['pane'],
        proposal_sha256=base.sha(PROPOSAL), approval_sha256=base.sha(APPROVAL)))
    if (result['limits'] != LIMITS or result['status'] != 'EXPOSED_MATCHED_DATA_DIAGNOSTIC_ONLY'
            or not result['old_gates_unchanged'] or result['new_validation_gate_defined']
            or result['model_selected'] or result['NAV043_passed'] or result['G5_passed']
            or result['additional_all_data_fit']):
        raise ValueError('result scope changed')
    expected, checks, rows = {'STARTED.json', 'FINISHED.json'}, [], []
    for level, train, held, probes in partitions(diagonal.load_cases()):
        for kind in KINDS:
            folder = output/level/kind
            model = base.load(folder/'model.json')
            theta = np.asarray(model['theta'])
            prior_audit.equivalent(model, dict(theta=theta.tolist(), stability=check_model(theta, kind),
                training=[c.key for c in train], held_navigation=[c.key for c in held],
                in_sample_probes=[c.key for c in probes], deployment_model=False))
            expected.add(str((folder/'model.json').relative_to(output)))
            for axis, name in enumerate(base.AXES):
                for suffix in ('started', 'result'):
                    expected.add(str((folder/(name+'-solve-'+suffix+'.json')).relative_to(output)))
                record = base.load(folder/(name+'-solve-result.json'))
                ordinal = len(checks)+1
                prior_audit.equivalent({k:record[k] for k in ('level','structure','axis','ordinal','rows')},
                    base.load(folder/(name+'-solve-started.json')))
                if (record['level'], record['structure'], record['axis'], record['ordinal']) != (level,kind,name,ordinal):
                    raise ValueError('solve order changed')
                prior_audit.equivalent(record, result['solves'][ordinal-1])
                prior_audit.equivalent(record['theta'], theta[axis].tolist())
                x, y, _ = design(train, kind, axis)
                checks.append(verify_solution(x, y, record, kind))
            for case in held+probes:
                role = 'in_sample_probe' if case.cohort == 'probe' else 'held_navigation'
                value, poses, responses = row(case, theta, kind, role)
                stem = case.trace.identity['kind']+'-'+case.trace.identity['method']
                prior_audit.equivalent(value, base.load(folder/(stem+'.json')))
                prior_audit.verify_prediction(folder/(stem+'.npz'), poses, responses)
                for suffix in ('json', 'npz'):
                    expected.add(str((folder/(stem+'.'+suffix)).relative_to(output)))
                rows.append(dict(fold=level, **value))
    if len(checks) != 18 or len(rows) != 108 or set(files) != expected or len(files) != 260:
        raise ValueError('incomplete or extra diagnostic output')
    prior_audit.equivalent(rows, result['rows'])
    prior_audit.equivalent(summaries(rows), result['descriptive_summary'])
    approved()
    diagonal.load_cases()
    if files != {str(f.relative_to(output)):base.sha(f) for f in output.rglob('*') if f.is_file()}:
        raise ValueError('output mutated during audit')
    base.write_new(REVIEW, dict(status='SAVED_MATCHED_SUPPORT_EXECUTION_AUDIT_PASS',
        actual_axis_solves=18, verification_spectra=18, extra_fits=0,
        normal_equation_checks=checks, descriptive_summary=result['descriptive_summary'],
        output_index=[dict(path=str(output/f), sha256=digest, bytes=(output/f).stat().st_size)
                      for f,digest in sorted(files.items())],
        probe_metrics_are_training_only=True, production_consumer=False,
        independent_validation=False, NAV043_passed=False, G5_passed=False))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--audit':
        audit()
    elif len(sys.argv) == 1:
        raise SystemExit(launch())
    else:
        raise SystemExit('use no arguments for once-only launch, or --audit')
