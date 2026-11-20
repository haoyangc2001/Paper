"""Read saved v4 coefficients/outputs: command support and exact error accounting.

No fitting, new trajectories, simulator, policy query or production consumer.
Axis-box coverage is only a necessary support check, never a convex-hull test.
"""
from pathlib import Path
import os
import sys

import numpy as np
import run_nav043_coupled_response as saved
import nav043_motion_probe_recovery_guard as guard

base = saved.base
REPORT = base.NAV/'reports/nav043_v4_support_review_v1.json'
DOMAIN = np.array([1., .4, 1.])


def finite_three(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('nonempty finite Nx3 array required')
    return a


def support(training, held):
    training, held = finite_three(training), finite_three(held)
    low, high = training.min(axis=0), training.max(axis=0)
    outside = (held < low-1.e-7) | (held > high+1.e-7)
    unique = np.unique(training, axis=0)/DOMAIN
    distance = []
    for start in range(0, len(held), 64):
        delta = held[start:start+64, None, :]/DOMAIN - unique[None, :, :]
        distance.extend(np.sqrt(np.min(np.sum(delta*delta, axis=2), axis=1)))
    distance = np.asarray(distance)
    return dict(training_min=low.tolist(), training_max=high.tolist(),
                held_min=held.min(axis=0).tolist(), held_max=held.max(axis=0).tolist(),
                held_intervals=len(held), outside_axis_box_count=int(outside.any(axis=1).sum()),
                outside_per_axis=outside.sum(axis=0).tolist(),
                normalized_nearest_training_distance_mean=float(distance.mean()),
                normalized_nearest_training_distance_max=float(distance.max()),
                full_convex_hull_or_state_support_verified=False)


def accounting(theta, previous, commands, actual, predicted, predicted_previous):
    """e_t = A e_(t-1) + innovation_t; prediction minus actual sign.

    Inputs are already aligned completed intervals, including genuine probe
    warmup. Navigation interval zero is deliberately excluded by the caller.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3, 11) or not np.isfinite(theta).all():
        raise ValueError('finite saved 3x11 coefficients required')
    arrays = [finite_three(v) for v in
              (previous, commands, actual, predicted, predicted_previous)]
    if len({v.shape for v in arrays}) != 1:
        raise ValueError('aligned completed intervals required')
    previous, commands, actual, predicted, predicted_previous = arrays
    memory, forcing = previous@theta[:, :3].T, saved.core.features(commands)@theta[:, 3:].T
    innovation = memory+forcing-actual
    error = predicted-actual
    propagated = (predicted_previous-previous)@theta[:, :3].T
    closure = error-propagated-innovation
    if np.max(np.abs(closure)) > 1.e-11:
        raise ValueError('saved recurrence or interval alignment differs')
    return dict(intervals=len(actual),
                teacher_one_step_RMSE=np.sqrt(np.mean(innovation**2, axis=0)).tolist(),
                free_response_RMSE=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                innovation_mean=innovation.mean(axis=0).tolist(),
                free_error_mean=error.mean(axis=0).tolist(),
                recurrence_closure_max=float(np.max(np.abs(closure))),
                forcing_mean=forcing.mean(axis=0).tolist(),
                actual_memory_mean=memory.mean(axis=0).tolist())


def forbidden_fit(*args, **kwargs):
    raise RuntimeError('saved-output review must not fit or solve parameters')


def main():
    saved.context()
    if REPORT.exists():
        raise ValueError('review exists; no overwrite or automatic repeat')
    # Enforce this script's no-parameter-solve boundary even if imports change.
    np.linalg.lstsq = forbidden_fit
    np.linalg.solve = forbidden_fit
    proposal = saved.approved()
    review_path = base.NAV/'reports/nav043_coupled_response_review_v4.json'
    review = base.load(review_path)
    if (base.sha(review_path) != '4eaf3d0a7a5c16ce8d2556e7efcb2503e11fba2b72fc94a1c1179250ee012ab9'
            or review['status'] != 'SAVED_COUPLED_RESPONSE_EXECUTION_AUDIT_PASS'
            or review['scientific_diagnostic_pass'] is not False):
        raise ValueError('different upstream v4 result')
    identities = guard.protected_sources()
    if identities != dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29):
        raise ValueError('protected identity scope changed')
    for entry in review['output_index']:
        p = Path(entry['path'])
        if p.stat().st_size != entry['bytes'] or base.sha(p) != entry['sha256']:
            raise ValueError('changed saved v4 output')
    output = Path(proposal['output_root'])
    cases = saved.load_cases()
    rows = {r['trial']: r for r in review['rows']}
    fold_rows, case_rows = [], []
    for level, train, held in saved.folds(cases):
        model = base.load(output/level/'model.json')
        if model['training'] != [c.key for c in train] or model['heldout'] != [c.key for c in held]:
            raise ValueError('changed whole-case split')
        theta = np.array(model['theta'])
        saved.check_model(theta)
        commands_train = np.concatenate([c.pairs()[1] for c in train])
        teachers = {}
        for label, subset in [('train', train), ('held', held)]:
            teachers[label] = {}
            for cohort in ('navigation', 'probe'):
                residuals = []
                for c in subset:
                    if c.cohort == cohort:
                        previous, u, target = c.pairs()
                        residuals.append(np.mean((previous@theta[:, :3].T
                            + saved.core.features(u)@theta[:, 3:].T-target)**2, axis=0))
                teachers[label][cohort] = np.sqrt(np.mean(residuals, axis=0)).tolist()
        fold_rows.append(dict(level=level, training_cases=[c.key for c in train],
                              held_cases=[c.key for c in held],
                              training_command_min=commands_train.min(axis=0).tolist(),
                              training_command_max=commands_train.max(axis=0).tolist(),
                              teacher_RMSE_equal_case_weight=teachers))
        for case in held:
            t = case.trace
            stem = t.identity['kind']+'-'+t.identity['method']
            with np.load(output/level/(stem+'.npz'), allow_pickle=False) as z:
                free = z['responses'].copy()
                poses = z['predicted'].copy()
            if free.shape != (len(t.interval_commands), 3) or poses.shape != t.actual.shape:
                raise ValueError('saved array shape differs')
            previous, commands, actual = case.pairs()
            n = len(t.observed_twists)
            first = 0 if case.cohort == 'probe' else 1
            free_previous = np.vstack((case.initial_q, free[:-1]))[first:n]
            detail = accounting(theta, previous, commands, actual, free[first:n], free_previous)
            # The terminal pose can end partway through a command interval.
            partial = t.ticks % 20
            yaw_error_from_responses = .1*np.sum(free[:n, 2]-t.observed_twists[:, 2])
            if partial:
                before = t.actual[n*20-1]
                actual_partial_yaw = float(base.wrap(t.actual[-1, 2]-before[2]))
                yaw_error_from_responses += .005*partial*free[n, 2]-actual_partial_yaw
            saved_terminal = rows[case.key]['candidate']['heading_terminal_rad']
            yaw_closure = float(base.wrap(yaw_error_from_responses-saved_terminal))
            if abs(yaw_closure) > 1.e-9:
                raise ValueError('unaccounted terminal yaw / partial tail')
            case_rows.append(dict(trial=case.key, cohort=case.cohort, fold=level,
                kind=t.identity['kind'], method=t.identity['method'],
                command_support=support(commands_train, commands), response_error=detail,
                initialization=('actual warmup q' if first == 0 else
                    'missing historical warmup; recurrence review excludes first interval'),
                response_axes_units=['m/s', 'm/s', 'rad/s'],
                measured_last_20_q_mean=t.observed_twists[-20:].mean(axis=0).tolist(),
                saved_free_last_20_q_mean=free[:n][-20:].mean(axis=0).tolist(),
                terminal_heading_error_deg=float(np.degrees(saved_terminal)),
                terminal_yaw_accounting_closure_rad=yaw_closure,
                terminal_partial_ticks=partial,
                original_position_RMSE_m=rows[case.key]['candidate']['position_RMSE_m']))
    # Recheck immutable sources before writing only this new report.
    saved.approved()
    saved.load_cases()
    assert guard.protected_sources() == identities
    assert len(case_rows) == 30 and len({r['trial'] for r in case_rows}) == 30
    result = dict(schema='nav043-v4-saved-support-review-v1',
        status='SAVED_SUPPORT_AND_ERROR_ACCOUNTING_PASS', pane=os.environ['TMUX_PANE'],
        source_report=str(review_path), source_report_sha256=base.sha(review_path),
        source_code_sha256=base.sha(__file__), identities=identities,
        folds=fold_rows, cases=case_rows, original_gate=review['diagnostic_gate'],
        scope=dict(new_parameter_solves=0, simulator_applications=0,
                   new_nominal_or_physical_episodes=0, W_queries=0,
                   actor_critic_updates=0, calibration_fits=0, GPU=0),
        limitations=['All thirty cases already exposed; no independent validation.',
            'Axis bounds and nearest command distances do not prove full state support.',
            'Error recurrence is algebraic accounting, not a physical intervention.',
            'Navigation warmup q is missing; no initial-state evidence is fabricated.',
            'Each controlled direction has one case; no repeatability guarantee.'],
        production_consumer=False, NAV043_passed=False, G5_passed=False)
    base.write_new(REPORT, result)
    print(result['status'], 'cases', len(case_rows), 'extra fits 0', flush=True)


if __name__ == '__main__':
    main()
