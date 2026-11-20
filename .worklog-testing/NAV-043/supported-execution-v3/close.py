"""Read-only diagnostic analysis and retained evidence; no parameter fitting."""
import json
import math
import os
from pathlib import Path
import socket
import sys
import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/supported-execution-v3'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import nav043_motion_probe_recovery_guard as g
import audit_nav043_supported_response as audit

c.context()
p = c.approved()
report = c.base.load(audit.REPORT)
assert report['status'] == 'SAVED_SUPPORTED_RESPONSE_EXECUTION_AUDIT_PASS'
assert not report['scientific_diagnostic_pass']
assert len(report['output_index']) == 83
assert (EV/'audit.exit').read_text().strip() == '0'
output = Path(p['output_root'])
finished = c.base.load(output/'FINISHED.json')
exited = c.base.load(c.HERE/'exit.json')
launch = c.base.load(c.HERE/'launch.json')
assert exited['exit_code'] == 0 and exited['finished'] and not exited['held']
assert not exited['retry_performed']
for name in ('wrapper_pid','child_pid'):
    assert not Path('/proc',str(launch[name])).exists(), name
identities = g.protected_sources()
assert identities == dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
cases = c.load_cases()
details = []
for level,train,held in c.folds(cases):
    theta = np.asarray(c.base.load(output/level/'model.json')['theta'])
    for case in held:
        previous, commands, truth = case.pairs()
        predicted = previous*theta[:,0] + c.features(commands) @ theta[:,1:].T
        error = predicted-truth
        row = next(r for r in report['rows'] if r['trial'] == case.key)
        item = dict(trial=case.key,level=level,cohort=case.cohort,kind=case.trace.identity['kind'],
                    method=case.trace.identity['method'],
                    one_step_q_bias=error.mean(axis=0).tolist(),
                    one_step_q_RMSE=np.sqrt((error**2).mean(axis=0)).tolist(),
                    commands_min=commands.min(axis=0).tolist(),commands_max=commands.max(axis=0).tolist(),
                    position_RMSE_m={k:row[k]['position_RMSE_m'] for k in ('candidate','affine','ideal')},
                    terminal_heading_error_deg={k:math.degrees(row[k]['heading_terminal_rad']) for k in ('candidate','affine','ideal')})
        if case.cohort == 'probe':
            command=case.trace.interval_commands[0]
            item['command']=command.tolist()
            item['algebraic_constant_command_limit_q']=(theta[:,1:] @ c.features(command[None])[0]/(1-theta[:,0])).tolist()
            item['constant_limit_is_not_measured_steady_state']=True
            item['actual_last_20_completed_q_mean']=case.trace.observed_twists[-20:].mean(axis=0).tolist()
        details.append(item)
small=[]
for item in report['output_index']:
    path=Path(item['path'])
    assert path.stat().st_size==item['bytes'] and c.base.sha(path)==item['sha256']
    if path.suffix=='.json':
        target=EV/'saved-small-output'/path.relative_to(output)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:
            f.write(path.read_bytes())
        assert c.base.sha(target)==item['sha256']
        small.append(str(target.relative_to(ROOT)))
assert len(small)==53
for name in ('launch.lock','launch.json','exit.json','run.log'):
    target=EV/'saved-launcher'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:
        f.write((c.HERE/name).read_bytes())
all_paths=[Path(item['path']) for item in report['output_index']]
heading={cohort:{key:np.mean([abs(math.degrees(r[key]['heading_terminal_rad']))
                    for r in report['rows'] if r['cohort']==cohort]).item()
                for key in ('candidate','affine','ideal')}
         for cohort in ('navigation','probe')}
summary=dict(status='RETAINED_ANALYSIS_PASS_DIAGNOSTIC_FAIL',host=socket.gethostname(),
    pane=os.environ['TMUX_PANE'],python=sys.executable,execution_audit=report['status'],
    actual_axis_solves=9,extra_parameter_solves=0,source_cases=30,
    measurement_ticks=182797,controlled_warmup_ticks=2400,
    candidate_wall_s=finished['cpu_wall_s'],wrapper_wall_s=exited['wrapper_wall_s'],
    original_processes_absent=True,identities=identities,gate=report['diagnostic_gate'],
    terminal_heading_MAE_deg=heading,details=details,
    original_files=83,original_bytes=sum(p.stat().st_size for p in all_paths),
    saved_small_files=53,external_npz_files=30,
    output_index=report['output_index'],saved_small_paths=small,
    new_applications=0,new_physical_episodes=0,new_nominal_episodes=0,
    W_queries=0,actor_critic_updates=0,calibration_fits=0,
    original_physical_HELD_preserved=True,production_enabled=False,
    NAV043=False,G5_passed=False,plan_true=55,
    limitations=['One-step q errors use observed previous q; free-running arrays use no corrections.',
                 'Prediction bias association is descriptive, not a unique physical cause.',
                 'M01 misses a strict no-worse-than-ideal check by millimetres; navigation L1 degradation is separate.',
                 'No new unknown/final or 90-second validation, calibration, deployment selection, or actor training.'])
c.base.write_new(EV/'close.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ('details','output_index','saved_small_paths')},ensure_ascii=False,indent=2),flush=True)
