"""Recompute the completed first saved pair only; no inference/new experiment."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import numpy as np
from nav067_sources import E, load, sha
from nav067_remaining_audit import audit_episode
from nav043_budget import actual_start_env, outcome
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from camra_navigation.envs.physical_v2 import PhysicalState, SubstepNavigation

assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
trial=Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1/L1_empty/uncalibrated')
saved=load(trial/'decision.json')
assert sha(trial/'decision.json')==load(trial/'decision.pin.json')['sha256']
nom=load(trial/'nominal.json')
result=load(trial/'result.json')
raw_hashes={str(p):sha(p) for p in trial.iterdir() if p.is_file()}
prereg,_=verify_prereg(REGISTRATION)
ctx=E.build_rollout_ctx('uncalibrated')
manifest=load('/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1/manifest.json')
row=next(r for r in manifest['maps'] if r['level']=='L1_empty')
spec=E.M.build(row['level'],row['seed'],ctx['design'])
assert spec.to_dict()==row['spec']
z=saved['decision']['selected_budget_J']
initial=PhysicalState(**saved['initial_state'])
traced=audit_episode(trial)
mon=SubstepNavigation(spec,ctx['geom'],ctx['cm'],[initial],[z],prereg)
rows=load(trial/'substeps.json')
for r in rows:
    mon.set_commands([r['applied']])
    state=r['state']
    mon.advance([PhysicalState(*state[:7],tuple(state[7:]))],physics_tick=r['physics_tick'])
    assert mon.rows[-1]==r,r['physics_tick']
metrics=outcome(result['label'],result['safety_y'],result['t_s'],
                traced['actual_work_J'],result['internal_model_cost_J'],z)
assert all(result[k]==v for k,v in metrics.items())
replay=actual_start_env(spec,ctx,z,prereg,initial)
for i,r in enumerate(nom['trace']):
    assert r['step']==i
    np.testing.assert_array_equal(replay.observation[0],r['observation'])
    transition=replay.step(np.asarray([r['action']],dtype=np.float32))
    np.testing.assert_array_equal(replay.pose[0],r['pose'])
    assert float(transition.z_J[0])==r['z_J']
    assert float(transition.safety_y[0])==r['safety_y']
    assert bool(transition.danger_violation[0])==r['danger_violation']
    assert float(transition.goal_distance_m[0])==r['goal_distance_m']
assert replay.done[0] and replay._label[0]==nom['label']
assert float(replay.result(0).t_s)==nom['t_s']
assert float(replay.result(0).cost_model_J)==nom['cost_model_J']
assert all(sha(p)==v for p,v in raw_hashes.items())
report={
    'at':datetime.now().astimezone().isoformat(),
    'status':'ONE_SAVED_PAIR_REVIEW_PASS_NOT_SCIENTIFIC_PASS',
    'host':socket.getfqdn(),'tmux':os.environ['TMUX'],
    'tmux_pane':os.environ.get('TMUX_PANE'),
    'script_sha256':sha(__file__),'raw_sources':raw_hashes,
    'level':'L1_empty','method':'uncalibrated',
    'physical':{'label':result['label'],'safety_y':result['safety_y'],
                't_s':result['t_s'],'budget_J':z,**traced,**metrics},
    'nominal':{k:v for k,v in nom.items() if k not in ('trace','initial_state')},
    'physical_substeps_replayed':len(rows),'nominal_saved_actions_replayed':len(nom['trace']),
    'NAV_043_pass':False,'G5_pass':False,
    'new_applications':0,'new_rollouts':0,'new_W_queries':0,'training_updates':0,
}
out=ROOT/'.worklog-testing/NAV-043/exploration-v1/first-saved-review.json'
with out.open('x') as stream:
    json.dump(report,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='raw_sources'},ensure_ascii=False,indent=2),flush=True)
