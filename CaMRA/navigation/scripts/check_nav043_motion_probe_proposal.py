"""Read-only protocol arithmetic and provenance review. No simulator launcher.

This command cannot approve or execute the proposal, fit parameters, query an
upper policy or write into any experiment output directory.
"""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import nav043_affine_response as affine

base = affine.base
PROPOSAL = base.NAV/'config/nav043_motion_probe_proposal_v1.json'


def review():
    if (socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX')
            or Path(sys.executable).resolve() != Path(base.PYTHON).resolve()):
        raise ValueError('requires registered tiger11/tmux/absolute Python')
    p = base.load(PROPOSAL)
    assert base.sha(PROPOSAL) == Path(str(PROPOSAL)+'.sha256').read_text().strip()
    assert p['schema'] == 'nav043-motion-probe-proposal-v1' and p['task_id'] == 'NAV-043'
    assert p['status'] == 'PROPOSED' and p['approval_required']
    assert p['execution_readiness'] == 'PROTOCOL_ONLY_COLLECTOR_NOT_IMPLEMENTED_OR_SEALED'
    assert not p['production_enabled'] and not p['previous_allocation_reused']
    assert not p['NAV043_passed'] and not p['G5_passed']
    assert not (base.ROOT/p['approval_path']).exists()
    assert not Path(p['output_root']).exists()
    for path, digest in p['source_hashes'].items():
        assert base.sha(base.ROOT/path) == digest, path
    env, schedule, limits = p['environment'], p['schedule'], p['limits']
    assert env['controller'] == 'camra_go2_velocity_it2999_v2'
    assert (env['physics_dt_s'], env['low_policy_dt_s'], env['command_dt_s']) == (.005, .02, .1)
    assert env['host'] == 'tiger11' and env['physical_gpu_index'] == 6
    assert env['python'] == base.PYTHON and env['tmux_required']
    assert env['command_limits'] == [[-1., 1.], [-.4, .4], [-1., 1.]]
    cases = schedule['cases']
    assert [row['id'] for row in cases] == [f'M{i:02}' for i in range(1, 13)]
    commands = [tuple(row['command']) for row in cases]
    assert len(set(commands)) == len(cases) == limits['controlled_physical_cases'] == 12
    for command in commands:
        assert len(command) == 3
        assert all(math.isfinite(x) and lo <= x <= hi
                   for x, (lo, hi) in zip(command, env['command_limits']))
    assert commands[0] == (0., 0., 0.)
    assert set(commands[1:6]) == {(.1, 0., 0.), (.2, 0., 0.), (.4, 0., 0.),
                                (1., 0., 0.), (-1., 0., 0.)}
    assert set(commands[6:10]) == {(0., .4, 0.), (0., -.4, 0.),
                                  (0., 0., 1.), (0., 0., -1.)}
    assert set(commands[10:]) == {(.4, .1, .4), (.4, -.1, -.4)}
    warmup = schedule['warmup_ticks_per_case']
    measure = schedule['measurement_ticks_per_case']
    assert warmup == schedule['warmup_s']/env['physics_dt_s'] == 200
    assert measure == schedule['measurement_s']/env['physics_dt_s'] == 2000
    assert schedule['warmup_q_endpoint_ticks'] == [warmup-20, warmup]
    assert schedule['warmup_command'] == [0., 0., 0.]
    assert len(cases)*(warmup+measure) == limits['warmup_and_measurement_environment_ticks'] == 26400
    assert (limits['setup_reset_reserve_step_calls'] + limits['warmup_and_measurement_environment_ticks']
            == limits['all_explicit_simulation_step_calls'] == 32000)
    assert (limits['applications'], limits['simultaneous_environments'],
            limits['gpus'], limits['gpu_wall_s']) == (1, 1, 1, 7200)
    for field in ('automatic_retries', 'physical_navigation_episodes', 'nominal_episodes',
                  'W_queries', 'response_parameter_fits', 'actor_critic_updates', 'calibration_fits'):
        assert limits[field] == 0
    plan_path = base.ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
    plan = base.load(plan_path)
    before = json.loads(subprocess.check_output(
        ['git', 'show', '63a5bf6:'+str(plan_path.relative_to(base.ROOT))], cwd=base.ROOT))
    assert {k:v for k,v in before.items() if k != 'tasks'} == {
        k:v for k,v in plan.items() if k != 'tasks'}
    additions = 0
    for old, now in zip(before['tasks'], plan['tasks']):
        assert {k:v for k,v in old.items() if k != 'work_records'} == {
            k:v for k,v in now.items() if k != 'work_records'}
        n = len(old['work_records'])
        assert old['work_records'] == now['work_records'][:n]
        if now['id'] == 'NAV-043':
            additions = len(now['work_records'])-n
            assert additions >= 1 and not now['status']
        else:
            assert now['work_records'] == old['work_records']
    states = {t['id']:t['status'] for t in plan['tasks']}
    assert len(plan['tasks']) == len(before['tasks']) == 67 and sum(states.values()) == 55
    assert plan['document']['version'] == p['plan_version'] == '1.5.0'
    assert [t['id'] for t in plan['tasks'] if not t['status']
            and all(states[d] for d in t['depends_on'])] == ['NAV-043']
    # Existing approval is used solely to verify the already exposed protection
    # and raw-input hash lists; it grants no new response collection or fitting.
    affine.approval()
    base.verify_protocol()
    return {'status':'PROPOSED_PROTOCOL_AND_SOURCES_REVIEWED',
        'at':datetime.now().astimezone().isoformat(), 'host':socket.gethostname(),
        'pane':os.environ['TMUX_PANE'], 'python':sys.executable,
        'proposal_sha256':base.sha(PROPOSAL), 'checker_sha256':base.sha(Path(__file__)),
        'case_count':len(cases), 'planned_case_ticks':26400, 'planned_step_call_cap':32000,
        'planned_wall_cap_s':7200, 'proposed_sources_checked':len(p['source_hashes']),
        'protected_identities_checked':295, 'existing_raw_input_hashes_checked':90,
        'plan_baseline':'63a5bf6', 'NAV043_appended_records':additions,
        'historical_definitions_statuses_records_preserved':True,
        'approval_exists':False, 'physical_collector_implemented':False,
        'experiment_output_exists':False, 'actual_new_applications':0,
        'actual_new_physics_steps':0, 'actual_new_fits':0, 'actual_W_queries':0,
        'actual_network_updates':0, 'GPU_use':False,
        'NAV043_passed':False, 'G5_passed':False}


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False, indent=2, allow_nan=False))
