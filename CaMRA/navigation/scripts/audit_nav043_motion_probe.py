"""Saved-only motion capture integrity and descriptive tracking; never fit.

Full review requires the exact new approval and finished allocation. Partial
captures stay retained failures; this reviewer cannot run a replacement case.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

import nav043_motion_probe_guard as guard
from camra_navigation.envs.physical_v2 import PhysicalState, require_command_evidence
from camra_navigation.motion_state import PoseSample, CompletedInterval


def same_numbers(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(same_numbers(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(same_numbers(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and abs(actual-expected) <= 1.e-9
    return actual == expected


def review_case(trial, case):
    trial = Path(trial)
    rows = [json.loads(line) for line in (trial/'actual.jsonl').read_text().splitlines()]
    reset, q, result = [guard.load(trial/name) for name in
                        ('reset.json','completed_q.json','result.json')]
    proofs = guard.load(trial/'commands.json')
    if len(rows) != 2200 or len(proofs) != 550:
        raise ValueError('incomplete state/command capture; retain failure')
    initial_clock = reset['actual_clock_s']
    command = np.asarray(case['command'], dtype=np.float32)
    samples, states = {}, []
    for tick, row in enumerate(rows, 1):
        s = PhysicalState(**row['state'])
        s.validate()
        expected = [0.,0.,0.] if tick <= 200 else command.tolist()
        if (row['physics_tick'] != tick
                or abs(row['sim_time_s']-initial_clock-tick*.005) > 1.e-7
                or row['phase'] != ('warmup' if tick <= 200 else 'measurement')
                or not np.allclose(row['requested_command'], expected, rtol=0., atol=1.e-7)
                or s.height < .15 or s.tilt_deg > 60.):
            raise ValueError('actual state/clock/schedule/fall integrity failed')
        samples[tick] = PoseSample(tick,(s.x,s.y,s.yaw))
        states.append(s)
    for index, proof in enumerate(proofs):
        begin = index*4
        if proof['tick_before'] != begin or proof['tick_after'] != begin+4 or proof['substeps'] != 4:
            raise ValueError('missing/duplicate physical command proof')
        require_command_evidence(proof['command_written'], proof['command_seen'],
            tick_before=begin, tick_after=begin+4, substeps=4)
        expected = [0.,0.,0.] if begin < 200 else command.tolist()
        if (not np.allclose(proof['command_written'], [expected], rtol=0., atol=1.e-7)
                or abs(proof['sim_time_before']-initial_clock-begin*.005) > 1.e-7
                or abs(proof['sim_time_after']-initial_clock-(begin+4)*.005) > 1.e-7):
            raise ValueError('saved actual command/timing differs from schedule')
    intervals = [CompletedInterval(samples[end-20], samples[end]).as_dict()
                 for end in range(200,2201,20)]
    if (q['ticks_captured'] != 2200 or q['partial_tail_ticks'] != 0 or q['label'] != 'timeout'
            or not same_numbers(q['completed_intervals'], intervals)
            or not same_numbers(q['last_completed_q'], intervals[-1])
            or not same_numbers(q['initial_actual_state'], reset['actual_states'][0])):
        raise ValueError('saved completed q/warmup endpoints differ from actual poses')
    metadata = guard.load(trial/'energy_metadata.json')
    with np.load(trial/'energy.npz', allow_pickle=False) as saved:
        energy = {**metadata, **{name:saved[name] for name in saved.files}}
    from camra_navigation.evaluation.recorder import offline_recompute
    audit = offline_recompute(energy,0)
    if (energy['steps'] != 2200 or energy['problems'] or energy['dt_s'] != .005
            or energy['num_envs'] != 1 or not audit['steps_match'] or audit['rel_diff'] > 1.e-10
            or not np.asarray(energy['valid_mask']).all()):
        raise ValueError('saved actual mechanical-work integrity failed')
    warmup = float(energy['energy_step_j'][:200,0].sum())
    measure = float(energy['energy_step_j'][200:,0].sum())
    for name, value in (('warmup_J',warmup), ('measurement_J',measure),
                        ('total_J',float(energy['total_j'][0]))):
        if abs(result[name]-value) > 1.e-8:
            raise ValueError('saved split mechanical work differs')
    if result['case_id'] != case['id'] or result['requested_command'] != case['command']:
        raise ValueError('case identity differs from preregistered order')
    if not reset['friction']['ok'] or not reset['mass']['ok']:
        raise ValueError('actual clean material/mass evidence failed')
    measured_q = np.asarray([r['q_vx_vy_yaw_rate'] for r in intervals[1:]])
    body = np.asarray([[s.vx,s.vy,s.wz] for s in states[200:]])
    return {'case_id':case['id'], 'requested_command':case['command'],
        'actual_ticks_reviewed':2200, 'actual_policy_proofs_reviewed':550,
        'warmup_q':intervals[0], 'measurement_mean_completed_q':measured_q.mean(axis=0).tolist(),
        'last_two_seconds_mean_completed_q':measured_q[-20:].mean(axis=0).tolist(),
        'measurement_mean_instantaneous_body_velocity':body.mean(axis=0).tolist(),
        'warmup_J':warmup, 'measurement_J':measure, 'total_J':warmup+measure,
        'energy_audit':audit, 'integrity_passed':True,
        'steady_state_established':False, 'independent_repeated_trials':1}


def review_steps(path):
    calls, returned, reserve, finished = 0, 0, 0, []
    current, case_ticks, in_flight = None, 0, False
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        kind = row['kind']
        if kind == 'case_begin':
            if current is not None or in_flight or row['case_id'] != f'M{len(finished)+1:02}':
                raise ValueError('saved physical case order/attempt failed')
            current, case_ticks = row['case_id'], 0
        elif kind == 'step_called':
            if in_flight:
                raise ValueError('overlapping saved explicit steps')
            calls += 1
            case_ticks += int(current is not None)
            reserve += int(current is None)
            in_flight = True
        elif kind == 'step_returned':
            if not in_flight:
                raise ValueError('explicit step returned without call')
            returned += 1
            in_flight = False
        elif kind == 'case_finished':
            if current is None or in_flight or case_ticks != 2200:
                raise ValueError('unfinished saved physical case')
            finished.append(current)
        else:
            raise ValueError('refused/unknown explicit physical step event')
        if (row['calls'] != calls or row['returned'] != returned or row['reserve'] != reserve
                or row['case_id'] != current or row['case_ticks'] != case_ticks):
            raise ValueError('saved explicit step journal accounting differs')
        if kind == 'case_finished':
            current, case_ticks = None, 0
    if in_flight or current is not None or len(finished) != 12 or calls-reserve != 26400 or calls > 32000 or reserve > 5600:
        raise ValueError('incomplete/exceeded saved physical allocation')
    return {'explicit_step_calls':calls, 'returned_calls':returned,
            'reserve_calls':reserve, 'completed_cases':len(finished)}


def main():
    p, _, _ = guard.approved()
    out = Path(p['output_root'])
    if not (out/'FINISHED.json').is_file() or (out/'HELD.json').exists() or (out/'ERROR.json').exists():
        raise ValueError('no finished physical allocation; do not relaunch')
    identities = guard.protected_sources()
    runtime = guard.load(out/'runtime.json')
    if runtime['num_envs'] != 1 or runtime['physics_dt_s'] != .005 or runtime['decimation'] != 4:
        raise ValueError('actual runtime timing differs')
    steps = review_steps(out/'step_calls.jsonl')
    ledger = guard.load(out/'execution_ledger.json')
    if any(ledger[name] != value for name, value in steps.items()):
        raise ValueError('final execution ledger differs from saved journal')
    rows = [review_case(out/case['id'], case) for case in p['schedule']['cases']]
    report = {'status':'SAVED_CONTROLLED_CAPTURE_INTEGRITY_PASS', 'cases':rows,
        'step_accounting':steps, 'identities_checked':identities,
        'proposal_sha256':guard.sha(guard.PROPOSAL), 'output_root':str(out),
        'physical_navigation_episodes':0, 'response_fits':0, 'W_queries':0,
        'actor_critic_updates':0, 'calibration_fits':0,
        'independent_unseen_validation':False, 'NAV043_passed':False, 'G5_passed':False}
    guard.persist(guard.NAV/'reports/nav043_motion_probe_saved_review_v1.json', report)
    print(report['status'], len(rows), flush=True)


if __name__ == '__main__':
    main()
