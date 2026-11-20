"""Read-only diagnosis of attempt1; no app, fit, policy, calibration or retry."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
import nav043_motion_probe_guard as G
from camra_navigation.motion_state import PoseSample, CompletedInterval
from camra_navigation.evaluation.recorder import offline_recompute
from camra_navigation.envs.physical_v2 import PhysicalState, require_command_evidence

p,a,r=G.approved()
assert os.environ.get('TMUX') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
out=Path(p['output_root']);trial=out/'M01'
owner=G.load(out/'OWNER.json');process=G.load(out/'PROCESS.json')
assert owner==G.load(G.EVIDENCE/'launch.lock')
assert owner['git_head'].startswith('05c52f4')
assert all(not (Path('/proc')/str(pid)).exists() for pid in (owner['pid'],process['pid']))
assert not (out/'FINISHED.json').exists() and not (out/'HELD.json').exists()
assert not (out/'child_step_accounting.json').exists()
assert [d.name for d in out.glob('M*') if d.is_dir()]==['M01']
error=G.load(out/'ERROR.json')
assert error['error']=="ValueError('actual simulator clock differs from consecutive physics ticks')"
reset=G.load(trial/'reset.json');origin=reset['actual_clock_s']
rows=[json.loads(line) for line in (trial/'actual.jsonl').read_text().splitlines()]
ticks=np.asarray([row['physics_tick'] for row in rows])
times=np.asarray([row['sim_time_s'] for row in rows])
assert np.array_equal(ticks,np.arange(1,896))
for row in rows:
    state=PhysicalState(**row['state']);state.validate()
    assert state.height>=.15 and state.tilt_deg<=60.
    assert row['requested_command']==[0.,0.,0.]
nominal_error=times-origin-ticks*.005
callback_dt=float(np.float32(.005))
representation_error=times-origin-ticks*callback_dt
first_rejected=int(np.flatnonzero(np.abs(nominal_error)>1.e-7)[0])+1
assert first_rejected==895 and np.max(np.abs(representation_error))<=1.e-12
assert np.max(np.abs(np.diff(np.r_[origin,times])-callback_dt))<=1.e-12
q=G.load(trial/'completed_q.json')
assert q['ticks_captured']==894 and q['measurement_ticks_captured']==694
assert q['partial_tail_ticks']==14 and q['last_completed_q']['end_tick']==880
intervals=[]
for end in range(200,881,20):
    before,after=rows[end-21],rows[end-1]
    def pose(row):
        s=row['state'];return PoseSample(row['physics_tick'],(s['x'],s['y'],s['yaw']))
    intervals.append(CompletedInterval(pose(before),pose(after)).as_dict())
assert q['completed_intervals']==intervals
meta=G.load(trial/'energy_metadata.json')
with np.load(trial/'energy.npz',allow_pickle=False) as z:
    energy={**meta,**{k:z[k] for k in z.files}}
audit=offline_recompute(energy,0)
assert energy['steps']==895 and audit['steps_match'] and audit['rel_diff']<=1.e-10
proofs=G.load(trial/'commands.json');end=0
for proof in proofs:
    assert proof['tick_before']==end and proof['substeps']==4
    require_command_evidence(proof['command_written'],proof['command_seen'],
        tick_before=end,tick_after=proof['tick_after'],substeps=4)
    assert proof['command_written']==[[0.,0.,0.]]
    end=proof['tick_after']
assert len(proofs)==223 and end==892
journal=[json.loads(line) for line in (out/'step_calls.jsonl').read_text().splitlines()]
called=[r for r in journal if r['kind']=='step_called']
returned=[r for r in journal if r['kind']=='step_returned']
assert len(called)==len(returned)==895
assert all(r['case_id']=='M01' and r['reserve']==0 for r in called)
assert not any(r['kind']=='case_finished' for r in journal)
assert error['step_accounting']==G.load(out/'pre_cleanup_step_accounting.json')
identities=G.protected_sources()
assert identities=={'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
paths=sorted(f for f in out.rglob('*') if f.is_file())
index=[{'path':str(f),'bytes':f.stat().st_size,'sha256':G.sha(f)} for f in paths]
exit_record=G.load(G.EVIDENCE/'exit.json')
assert 'TimeoutExpired' in exit_record['error'] and not exit_record['retry_performed']
timer=Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py')
assert 'self._current_time += step_size' in timer.read_text()
report={'schema':'nav043-motion-probe-first-failure-review-v1','status':'FAILED_CAPTURE_RETAINED',
    'at':datetime.now().astimezone().isoformat(),'pane':os.environ.get('TMUX_PANE'),
    'output_root':str(out),'applications_started':1,'cases_started':1,'cases_completed':0,
    'unstarted_cases':[f'M{i:02}' for i in range(2,13)],
    'explicit_calls_and_returns':895,'setup_reserve_explicit_calls':0,
    'native_internal_initialization_steps':'unobservable; not declared zero',
    'raw_state_rows':895,'accepted_capture_ticks':894,'warmup_ticks':200,
    'raw_measurement_rows':695,'accepted_measurement_ticks':694,
    'completed_command_proofs':223,'command_proofs_cover_through_tick':892,
    'terminal_ticks_without_completed_policy_proof':[893,894,895],
    'all_saved_states_upright':True,'q_intervals_recomputed':len(intervals),
    'last_completed_q_tick':880,'accepted_partial_tail_ticks':14,
    'clock':{'origin_s':origin,'last_s':float(times[-1]),'nominal_dt_s':.005,
        'float32_callback_dt_s':callback_dt,'first_rejected_tick':first_rejected,
        'last_nominal_error_s':float(nominal_error[-1]),
        'max_error_against_float32_dt_s':float(np.max(np.abs(representation_error))),
        'installed_timer_source':str(timer),'installed_timer_sha256':G.sha(timer),
        'interpretation':'The 1e-7 absolute cumulative check rejected float32 physics-callback dt representation. This is not evidence of a missing physics step.'},
    'energy':{'audit':audit,'warmup_J':float(energy['energy_step_j'][:200,0].sum()),
        'partial_measurement_J':float(energy['energy_step_j'][200:,0].sum()),
        'total_J':float(energy['total_j'][0]),'original_problems':meta['problems']},
    'launcher_exit':exit_record,'both_registered_processes_absent':True,
    'terminal_marker_gap':'cleanup wait TimeoutExpired escaped before HELD and child final accounting; original files unchanged',
    'startup_message_correction':'libXt/omni.UsdMdl errors also existed on historical successful startup; APP_READY and runtime are present. No evidence these caused the capture clock exception.',
    'manual_stop_attempt':'stop-startup.py identity assertion failed before any os.kill; no signal sent by that helper',
    'saved_files':index,'identities_checked':identities,
    'extra_applications':0,'new_fits':0,'new_W_queries':0,'new_network_updates':0,
    'new_calibration':0,'no_automatic_retry':True,'NAV043_passed':False,'G5_passed':False}
assert not (G.EVIDENCE/'startup-stop-request.json').exists()
G.persist(NAV/'reports/nav043_motion_probe_first_failure_v1.json',report)
print(json.dumps({k:report[k] for k in ('status','applications_started','cases_started','cases_completed','explicit_calls_and_returns','clock','launcher_exit')},ensure_ascii=False,indent=2))
