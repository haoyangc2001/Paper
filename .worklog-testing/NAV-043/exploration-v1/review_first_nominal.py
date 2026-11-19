"""Review saved nominal and energy data after preserving first-review failure."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
from nav067_sources import E, NAV, load, sha
from nav067_remaining_audit import audit_episode
from nav043_budget import outcome
from audit_nav043_exploration import audit_nominal_trace
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from camra_navigation.envs.physical_v2 import PhysicalState

assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
root=Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1/L1_empty')
prereg,_=verify_prereg(REGISTRATION)
manifest=load('/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1/manifest.json')
row=next(r for r in manifest['maps'] if r['level']=='L1_empty')
reports=[];sources={}
for method in ('uncalibrated','full'):
    trial=root/method
    hashes={str(p):sha(p) for p in trial.iterdir() if p.is_file()}
    saved=load(trial/'decision.json');nom=load(trial/'nominal.json');r=load(trial/'result.json')
    assert sha(trial/'decision.json')==load(trial/'decision.pin.json')['sha256']
    ctx=E.build_rollout_ctx(method)
    spec=E.M.build(row['level'],row['seed'],ctx['design'])
    assert spec.to_dict()==row['spec']
    initial=PhysicalState(**saved['initial_state'])
    z=saved['decision']['selected_budget_J']
    traced=audit_episode(trial)
    metrics=outcome(r['label'],r['safety_y'],r['t_s'],r['measured_joint_work_J'],r['internal_model_cost_J'],z)
    assert all(r[k]==v for k,v in metrics.items())
    n=audit_nominal_trace(nom,spec,ctx,z,prereg,initial)
    assert all(sha(p)==v for p,v in hashes.items())
    sources.update(hashes)
    reports.append({
        'method':method,'physics_ticks_accounted':traced['physics_ticks'],
        'saved_nominal_actions_replayed':n,
        'online_work_J':r['measured_joint_work_J'],'offline_work_J':traced['actual_work_J'],
        'offline_minus_online_work_J':traced['actual_work_J']-r['measured_joint_work_J'],
        'saved_business_metrics_exactly_reproduced':True,
        'physical_label':r['label'],'physical_t_s':r['t_s'],'budget_J':z,
        **metrics,
        'nominal':{k:v for k,v in nom.items() if k not in ('trace','initial_state')},
    })
report={
    'at':datetime.now().astimezone().isoformat(),
    'status':'L1_NOMINAL_AND_ACCOUNTING_PASS_FULL_PHYSICAL_EVENT_REVIEW_PENDING',
    'host':socket.getfqdn(),'tmux':os.environ['TMUX'],'tmux_pane':os.environ.get('TMUX_PANE'),
    'reviewer_sha256':sha(NAV/'scripts/audit_nav043_exploration.py'),
    'script_sha256':sha(__file__),'raw_sources':sources,'pairs':reports,
    'preserved_first_review_error':sha(ROOT/'.worklog-testing/NAV-043/exploration-v1/first-saved-review.log'),
    'new_applications':0,'new_rollouts':0,'new_W_queries':0,'training_updates':0,
    'NAV_043_pass':False,'G5_pass':False,
}
out=ROOT/'.worklog-testing/NAV-043/exploration-v1/first-nominal-review.json'
with out.open('x') as stream:
    json.dump(report,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='raw_sources'},ensure_ascii=False,indent=2),flush=True)
