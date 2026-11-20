"""Retain audited response-v4 output and describe errors; no parameter solves."""
import json
import math
import os
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/coupled-execution-v4'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import run_nav043_coupled_response as c
import nav043_motion_probe_recovery_guard as guard
import audit_nav043_coupled_response as audit

c.context()
p = c.approved()
report = c.base.load(audit.REPORT)
assert report['status'] == 'SAVED_COUPLED_RESPONSE_EXECUTION_AUDIT_PASS'
assert report['scientific_diagnostic_pass'] is False
assert (EV/'audit.exit').read_text().strip() == '0'
output = Path(p['output_root'])
finished = c.base.load(output/'FINISHED.json')
exited = c.base.load(c.HERE/'exit.json')
launch = c.base.load(c.HERE/'launch.json')
assert exited['exit_code'] == 0 and exited['finished'] and not exited['held']
assert exited['retry_performed'] is False
for name in ('wrapper_pid', 'child_pid'):
    assert not Path('/proc', str(launch[name])).exists(), name
identities = guard.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
cases = c.load_cases()
details, models = [], []
for level, train, held in c.folds(cases):
    saved = c.base.load(output/level/'model.json')
    theta = np.asarray(saved['theta'])
    models.append(dict(level=level, stability=saved['stability']))
    for case in held:
        previous, commands, truth = case.pairs()
        prediction = previous @ theta[:, :3].T + c.core.features(commands) @ theta[:, 3:].T
        error = prediction-truth
        row = next(r for r in report['rows'] if r['trial'] == case.key)
        keys = ('candidate', 'affine', 'ideal', 'previous_v3')
        item = dict(trial=case.key, level=level, cohort=case.cohort,
            kind=case.trace.identity['kind'], method=case.trace.identity['method'],
            one_step_q_bias=error.mean(axis=0).tolist(),
            one_step_q_RMSE=np.sqrt((error**2).mean(axis=0)).tolist(),
            commands_min=commands.min(axis=0).tolist(), commands_max=commands.max(axis=0).tolist(),
            position_RMSE_m={k: row[k]['position_RMSE_m'] for k in keys},
            terminal_heading_error_deg={k: math.degrees(row[k]['heading_terminal_rad']) for k in keys})
        if case.cohort == 'probe':
            item['command'] = case.trace.interval_commands[0].tolist()
            item['actual_last_20_completed_q_mean'] = case.trace.observed_twists[-20:].mean(axis=0).tolist()
        details.append(item)
small = []
assert len(report['output_index']) == 83
for item in report['output_index']:
    path = Path(item['path'])
    assert path.stat().st_size == item['bytes'] and c.base.sha(path) == item['sha256']
    if path.suffix == '.json':
        target = EV/'saved-small-output'/path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
        assert c.base.sha(target) == item['sha256']
        small.append(str(target.relative_to(ROOT)))
assert len(small) == 53
for name in ('launch.lock', 'launch.json', 'exit.json', 'run.log'):
    target = EV/'saved-launcher'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write((c.HERE/name).read_bytes())
heading = {cohort: {key: float(np.mean([abs(math.degrees(r[key]['heading_terminal_rad']))
                   for r in report['rows'] if r['cohort'] == cohort]))
                   for key in ('candidate', 'affine', 'ideal', 'previous_v3')}
           for cohort in ('navigation', 'probe')}
aggregate = {cohort: {key: math.sqrt(float(np.mean([r[key]['position_RMSE_m']**2
                     for r in report['rows'] if r['cohort'] == cohort])))
                     for key in ('candidate', 'affine', 'ideal', 'previous_v3')}
             for cohort in ('navigation', 'probe')}
summary = dict(status='RETAINED_ANALYSIS_PASS_DIAGNOSTIC_FAIL', host=socket.gethostname(),
    pane=os.environ['TMUX_PANE'], python=sys.executable, execution_audit=report['status'],
    actual_axis_solves=9, extra_parameter_solves=0, source_cases=30,
    measurement_ticks=182797, controlled_warmup_ticks=2400,
    candidate_wall_s=finished['cpu_wall_s'], wrapper_wall_s=exited['wrapper_wall_s'],
    original_processes_absent=True, identities=identities, gate=report['diagnostic_gate'],
    terminal_heading_MAE_deg=heading, cohort_position_RMSE_m=aggregate,
    model_stability=models, details=details,
    original_files=83, original_bytes=sum(i['bytes'] for i in report['output_index']),
    saved_small_files=53, external_npz_files=30, output_index=report['output_index'],
    saved_small_paths=small, new_applications=0, new_physical_episodes=0,
    new_nominal_episodes=0, W_queries=0, actor_critic_updates=0, calibration_fits=0,
    original_physical_HELD_preserved=True, production_enabled=False,
    NAV043=False, G5_passed=False, plan_true=55,
    limitations=['One-step errors use observed previous q; free prediction receives no truth correction.',
      'Observed prediction errors do not identify one unique physical cause.',
      'Overall navigation improves but L1 and several controlled checks fail; no diagnostic pass.',
      'All data exposed and correlated: no independent unknown/final/90s validation or calibration.',
      'No deployment model, production consumer, actor training, automatic retries or additional fit.'])
c.base.write_new(EV/'close.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k not in ('details', 'output_index', 'saved_small_paths')}, indent=2), flush=True)
