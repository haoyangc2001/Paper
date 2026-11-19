"""Read only NAV-067 saved pilot evidence; no simulator, rollout or inference.

The output diagnoses transfer of the old margins. It cannot accept NAV-043,
select a budget, fit a margin, or turn exposed wiring data into a heldout test.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parents[1]
sys.path[:0] = [str(NAV / 'scripts'), str(ROOT / 'CaMRA/calibration/src')]
from camra_calibration.nominal import PlanarState, step_exact
from nav067_remaining_audit import audit_episode

PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
ACCEPTANCE = NAV / 'reports/nav067_engineering_acceptance_v1.json'
ACCEPTANCE_SHA = '54da0a72821f31dffe8dd5c6c4f314430ece66fb048a6adcde3a7a570d46a530'
PREFIXES = (5, 10, 20, 30, 45)
DT = .005


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


def q90(values):
    values = sorted(float(v) for v in values)
    return values[math.ceil(.9 * len(values)) - 1] if values else None


def command_summary(commands, duration_s):
    """200 Hz time weighted shares, without pretending to be 10 Hz dwell."""
    arr = np.asarray(commands, dtype=float)
    low = (np.abs(arr[:, 0]) <= .3) & (np.abs(arr[:, 2]) <= .3)
    old_domain = (np.abs(arr[:, 0]) <= .6 + 1.e-12) & (
        np.abs(arr[:, 1]) <= 1.e-12) & (np.abs(arr[:, 2]) <= .33 + 1.e-12)
    calibrated_domain = np.all(np.abs(arr) <= np.array([1., .4, 1.]) + 1.e-12, axis=1)
    # Commands are held at 10 Hz. Use those genuine change opportunities;
    # final partial interval is kept in time weighted shares above.
    nav = arr[::20]
    differences = np.abs(np.diff(nav, axis=0)).max(axis=1)
    flips = {}
    for i, axis in enumerate(('vx', 'vy', 'wz')):
        signs = np.sign(nav[:, i][nav[:, i] != 0.])
        flips[axis] = int(np.sum(signs[1:] != signs[:-1])) / duration_s
    return {'grid': 'actual held command at every active 200 Hz substep',
            'low_speed_active_time_share': float(low.mean()),
            'outside_old_rule_table_domain_time_share': float((~old_domain).mean()),
            'outside_frozen_calibration_command_domain_time_share': float((~calibrated_domain).mean()),
            'max_abs_vx_vy_wz': np.abs(arr).max(axis=0).tolist(),
            '10Hz_change_abs_inf_mean': float(differences.mean()) if len(differences) else 0.,
            '10Hz_change_abs_inf_max': float(differences.max()) if len(differences) else 0.,
            'sign_switches_per_actual_active_second': flips}


def analyze_trial(trial, kind, level, index, alpha, delta, old_table):
    trial = Path(trial)
    sources = {}
    for name in ('result.json', 'substeps.json', 'commands.json', 'navigation.json', 'energy.npz'):
        path = trial / name
        assert str(path) in index and sha(path) == index[str(path)], str(path)
        sources[str(path)] = index[str(path)]
    audited = audit_episode(trial)
    result, rows = load(trial / 'result.json'), load(trial / 'substeps.json')
    with np.load(trial / 'energy.npz', allow_pickle=False) as archive:
        tau = archive['tau_applied'].astype(np.float64)
        velocity = archive['qdot_post'].astype(np.float64)
        valid = archive['valid_mask']
        stored_energy = archive['energy_step_j'].copy()
    assert valid.shape == (len(rows), 1) and valid.all()
    actual_steps = np.maximum(tau * velocity, 0.).sum(axis=-1)[:, 0] * DT
    np.testing.assert_allclose(actual_steps, stored_energy[:, 0], atol=1.e-12, rtol=1.e-12)
    actual_cum = np.cumsum(actual_steps)
    nominal_cum = np.cumsum([r['cost_nom_J'] for r in rows])
    factor = 1. + alpha if result['method'] == 'full' else 1.
    assert abs(nominal_cum[-1] * factor - result['internal_model_cost_J']) < 1.e-6
    initial = result['warmup']['state_at_window_start'][0]
    reference = PlanarState(initial['x'], initial['y'], initial['yaw'])
    deviations = []
    reference_error = 0.
    for row in rows:
        reference = step_exact(reference, row['applied'], DT)
        expected = [reference.px, reference.py, reference.yaw]
        reference_error = max(reference_error, max(abs(a-b) for a, b in zip(expected, row['ref_pose'])))
        deviation = math.hypot(row['state'][0] - reference.px, row['state'][1] - reference.py)
        assert abs(deviation - row['deviation_m']) < 1.e-9
        deviations.append(deviation)
    assert reference_error < 1.e-9
    deviations = np.asarray(deviations)
    def prefix(ticks):
        nom, actual = float(nominal_cum[ticks-1]), float(actual_cum[ticks-1])
        assert nom >= 1., 'small-cost prefix must be explicitly noncomputable'
        # The historical table is on 10 Hz. 200 Hz max is additional evidence.
        grid = deviations[19:ticks:20]
        assert len(grid)
        return {'active_ticks': ticks, 'duration_s': ticks * DT,
                'nominal_command_cost_J': nom, 'actual_joint_work_J': actual,
                'eps_c': actual / nom - 1.,
                'cost_covered_by_original_alpha': actual <= (1. + alpha) * nom,
                'model_ledger_J': factor * nom,
                'max_position_error_10Hz_m': float(grid.max()),
                'max_position_error_200Hz_m': float(deviations[:ticks].max()),
                'position_covered_by_original_delta_on_10Hz': bool(grid.max() <= delta)}
    prefixes = {}
    for seconds in PREFIXES:
        ticks = round(seconds / DT)
        if len(rows) < ticks:
            prefixes[str(seconds)] = {'complete': False, 'available_duration_s': len(rows) * DT,
                                      'reason': 'terminated before prefix; no padding or extrapolation'}
            continue
        item = {'complete': True, **prefix(ticks)}
        if str(seconds) in old_table:
            limit = old_table[str(seconds)]
            item['old_fitted_rule_table_limit_m'] = limit
            item['below_old_table_numeric_value_only'] = item['max_position_error_10Hz_m'] <= limit
            item['old_table_transfer_accepted'] = False
        else:
            item['old_table_comparison'] = 'OUT_OF_SCOPE: old table ends at 30 seconds'
        prefixes[str(seconds)] = item
    terminal = prefix(len(rows))
    terminal['terminal_position_error_m'] = float(deviations[-1])
    return {'level': level, 'method': result['method'], 'trial_kind': kind,
            'trial': str(trial), 'source_hashes': sources, 'saved_trace_audit': audited['status'],
            'budget_mode': result['budget_mode'], 'z0_J': result['z0_J'],
            'label': result['label'], 'safety_y': result['safety_y'],
            'physics_ticks': len(rows), 'terminal': terminal, 'prefixes_s': prefixes,
            'commands': command_summary([r['applied'] for r in rows], len(rows) * DT),
            'reconstructed_reference_max_abs_error': reference_error,
            'independent_performance_evidence': False}


def resource_audit(plan):
    resource = load(NAV / 'config/resource_budget_v1.json')
    registration = load(NAV / 'config/experiment_prereg_v1.json')
    low = load(NAV / 'reports/low_speed_energy_executability.json')
    old = load(NAV / 'reports/calibration_applicability_pilot.json')
    low_count = len(low['attempts'])
    old_count = sum(run['n'] for run in old['runs'])
    records = []
    for task in plan['tasks']:
        if task['id'] in ('NAV-021', 'NAV-059', 'NAV-060', 'NAV-034', 'NAV-042'):
            records.extend({'task': task['id'], 'at': r['timestamp'], 'summary': r['summary']}
                           for r in task['work_records'])
    return {'legacy_total_diagnostic_episode_cap': resource['formal_evaluation_quotas']['diagnostics_total_episodes'],
            'legacy_total_diagnostic_wall_cap_hours': resource['formal_evaluation_quotas']['diagnostics_wall_clock_cap_hours'],
            'legacy_candidate_check_allocation': registration['diagnostics']['nav_034_043_candidate_check'],
            'stale_NAV043_quick_check_entry': resource['quick_check_quotas']['NAV-043_deployed_policy_recheck'],
            'latest_retained_low_speed_report_attempts': low_count,
            'latest_retained_applicability_report_episodes': old_count,
            'latest_retained_report_episode_count_sum': low_count + old_count,
            'low_speed_latest_report_wall_hours': low['wall_clock']['hours_used'],
            'applicability_latest_report_batch_wall_seconds': sum(r['wallclock_s'] for r in old['runs']),
            'historical_task_records': records,
            'accounting_limit': 'Latest retained reports and append-only records are different provenance. Records describe earlier reruns; do not sum these into an exact deduplicated ledger or allege unauthorized use. Startup/failed attempts and all previous wall time are not fully reconstructable here.',
            'available_NAV043_episodes': None, 'available_NAV043_gpu_wall_hours': None,
            'new_execution_authorized_by_this_audit': False,
            'disposition': 'No demonstrable transferable remainder. Need an independent finite NAV-043 registration before any new query/application/episode; NAV-067 recovery allocation is closed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    assert output.parent == NAV / 'reports' and not output.exists()
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    assert sys.executable == PYTHON
    raw = (ROOT / 'doc/plan/20261109_camra_navigation_implementation.json').read_bytes()
    plan = json.loads(raw)
    states = {t['id']: t['status'] for t in plan['tasks']}
    assert next(t['id'] for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])) == 'NAV-043'
    assert sha(ACCEPTANCE) == ACCEPTANCE_SHA
    acceptance = load(ACCEPTANCE)
    original_path = NAV / 'reports/nav067_recovery_v3_execution_review.json'
    assert sha(original_path) == acceptance['original_pilot_report_sha256']
    original = load(original_path)
    index = {r['path']: r['sha256'] for r in acceptance['raw_artifact_index']}
    for entry in original['entries']:
        index.update(entry['artifacts'])
    prereg_path = NAV / 'config/nav066_decision_prereg_v2.json'
    assert sha(prereg_path) == '6e783599f76e68cc483bfc9f4d3e7649e4b312f48335a5c7c578f974ab9d126c'
    prereg = load(prereg_path)
    protected = {str(NAV / k): v for k, v in prereg['implementation_hashes'].items()}
    protected.update(prereg['protected_hashes'])
    protected.update({c['checkpoint']: c['checkpoint_sha256'] for c in prereg['six_candidates']})
    for name, digest in protected.items():
        assert sha(name) == digest, name
    calibration = load(NAV / 'config/calibration_applicability_prereg_v1.json')
    alpha, delta = (calibration['frozen_values'][k] for k in ('alpha', 'delta'))
    geometry = load(NAV / 'config/geometry_margin_closedloop_v1.json')
    assert geometry['validity']['T_range_s'] == [5, 30] and geometry['derivation']['fit_not_validation']
    rows = [analyze_trial(pair[kind]['trial'], kind, pair['level'], index, alpha, delta,
                          geometry['table_q90_by_prefix_s'])
            for pair in acceptance['paired_repetitions'] for kind in ('baseline', 'repeat')]
    groups = []
    for method in ('uncalibrated', 'full'):
        subset = [r for r in rows if r['method'] == method and r['trial_kind'] == 'baseline']
        group = {'method': method, 'distinct_configurations': len(subset), 'repetitions_per_configuration': 2,
                 'q90_is_descriptive_nearest_rank_n3': True, 'prefixes_s': {}}
        for seconds in PREFIXES:
            available = [r['prefixes_s'][str(seconds)] for r in subset if r['prefixes_s'][str(seconds)]['complete']]
            group['prefixes_s'][str(seconds)] = {'n_complete': len(available),
                'q90_eps_c': q90([r['eps_c'] for r in available]),
                'q90_max_position_error_10Hz_m': q90([r['max_position_error_10Hz_m'] for r in available])}
        groups.append(group)
    sources = [ACCEPTANCE, original_path, prereg_path, NAV / 'config/calibration_applicability_prereg_v1.json',
               NAV / 'config/geometry_margin_closedloop_v1.json', NAV / 'config/resource_budget_v1.json',
               NAV / 'config/experiment_prereg_v1.json', NAV / 'reports/low_speed_energy_executability.json',
               NAV / 'reports/calibration_applicability_pilot.json', Path(__file__)]
    report = {'schema': 'camra-nav043-saved-applicability-audit-v1',
        'at': datetime.now().astimezone().isoformat(), 'task': 'NAV-043',
        'status': 'SAVED_EVIDENCE_DIAGNOSTIC_NOT_APPLICABILITY_ACCEPTANCE',
        'plan_before_sha256': hashlib.sha256(raw).hexdigest(), 'all_plan_bytes_read': len(raw),
        'all_work_records_read': sum(len(t['work_records']) for t in plan['tasks']),
        'frozen_alpha': alpha, 'frozen_delta_m': delta, 'protected_identities_checked': len(protected),
        'source_hashes': {str(p): sha(p) for p in sources}, 'rows': rows,
        'descriptive_groups_without_pooled_duplicates': groups, 'resources': resource_audit(plan),
        'new_applications': 0, 'new_physical_episodes': 0, 'new_nominal_episodes': 0,
        'W_queries': 0, 'training_updates': 0, 'final_data_access': False,
        'NAV043_passed': False, 'G5_passed': False,
        'limitations': ['Exposed k=1 wiring traces; no value-selected physical performance claim.',
            'Repetitions are not six independent configurations per method.',
            'Old fitted table is domain limited and stops at 30 seconds; numeric comparisons do not extend validity.',
            'Frozen delta/alpha comparisons diagnose transfer; no coverage probability or unknown-scene guarantee.',
            'Nominal command reference replays saved commands; it is not a fresh closed-loop nominal rollout.',
            'Incomplete prefixes remain unavailable; no padding, terminal relabeling, fitting or threshold changes.']}
    write_new(output, report)
    print(json.dumps({'status': report['status'], 'rows': len(rows), 'groups': groups,
                      'resource_report_episodes': report['resources']['latest_retained_report_episode_count_sum']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
