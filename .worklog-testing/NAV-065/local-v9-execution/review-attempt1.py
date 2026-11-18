"""Main-agent acceptance: reconstruct targets and re-evaluate, never re-optimize."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
HERE=Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert sys.executable=='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'src')]
import probe_nav065_local as L

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
context=read(HERE/'context.json')
protocol,pin=L.approved_registration(NAV/'config/nav065_local_probe_prereg_v9.json')
assert pin==context['approval_pin']
process=read(HERE/'process.json')
assert process['returncode']==0 and process['finished_exists']
out=L.R.WORK/protocol['output_name']
assert {p.name for p in out.iterdir()}=={'selection.json','local.jsonl','diagnostic-probe.pt','PROBE_FINISHED.json'}
finished=read(out/'PROBE_FINISHED.json')
assert finished['pin']==pin and finished['new_rollouts']==finished['development_scoring_rounds']==0
assert not any(finished[k] for k in ('scientific_nav065_pass','deployment_allowed','automatic_expansion_allowed'))
for name,digest in protocol['implementation_hashes'].items():
    assert sha(NAV/name)==sha(HERE/'source'/name)==digest,name
print('V9_REVIEW_RECONSTRUCT_SAVED_TRAIN',flush=True)
upstream,reg,inputs,prepared=L.sources()
digests={c['run_id']:L.R.digests(net) for c,net,_,_ in prepared}
diagnostic,keys=L.diagnose(reg,inputs,prepared)
assert diagnostic==read(out/'selection.json')
preflight=read(NAV/'reports/nav065_local_probe_preflight_v9.json')
assert diagnostic['local_keys']==preflight['local_keys']
assert diagnostic['local_targets']==preflight['local_targets']
assert diagnostic['full_train_gate']==preflight['full_train_gate']==finished['full_train_before']
c,source,old,new=next(r for r in prepared if r[0]['run_id']==L.P.TARGET)
scale=inputs.value['quantities']['Z_REF_J']['value']
batch=L.local_batch(old+new,keys)
checkpoint=out/'diagnostic-probe.pt'
assert sha(checkpoint)==finished['diagnostic_checkpoint_sha256']
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert payload['diagnostic_only'] is True and payload['pin']==pin
model=L.R.load_network(inputs,c,checkpoint)
after=L.R.digests(model)
assert after['policy']==digests[L.P.TARGET]['policy']
assert any(after[r]!=digests[L.P.TARGET][r] for r in L.R.SCORE_ROLES)
local=L.evaluate_local(model,batch,reg,scale)
assert local==finished['local']
gate=L.P.A.train_gate(model,old,new,reg,L.R.load(L.P.PROPOSAL)['fit'],scale)
assert gate==finished['full_train_after']
steps=finished['optimizer_trials']
assert 0<=steps<=400
rows=[json.loads(line) for line in (out/'local.jsonl').read_text().splitlines()]
evaluations=[r for r in rows if r['kind']=='local_evaluation']
attempts=[r for r in rows if r['kind']=='optimizer_attempt']
assert len(rows)==2*steps+1
assert [r['trial'] for r in evaluations]==list(range(steps+1))
assert [r['trial'] for r in attempts]==list(range(1,steps+1))
assert not any(r['local_pass'] for r in evaluations[:-1])
assert local['local_pass'] or steps==400
assert {k:v for k,v in evaluations[-1].items() if k not in ('kind','trial','loss_before')}==local
obs,budget,tc,th=batch
true=np.maximum(th,tc-budget/scale)<=0
for r in evaluations:
    vc=np.asarray(r['predictions']['vc_norm']);vh=np.asarray(r['predictions']['vh'])
    assert np.isfinite(vc).all() and np.isfinite(vh).all()
    predicted=np.maximum(vh,vc-budget/scale)<=0
    assert np.array_equal(np.maximum(vh,vc-budget/scale),r['predictions']['W'])
    counts={'false_feasible':int((~true&predicted).sum()),'false_infeasible':int((true&~predicted).sum()),
        'joint_sign_errors':int((true!=predicted).sum()),
        'cost_sign_errors':int(((vc-budget/scale<=0)!=(tc-budget/scale<=0)).sum()),
        'reach_sign_errors':int(((vh<=0)!=(th<=0)).sum())}
    assert all(r[k]==v for k,v in counts.items())
    metrics={name:L.error_metrics(pred,target,reg['criteria']['variance_ratio_min'])
        for name,pred,target in (('vc_norm',vc,tc),('vh',vh,th))}
    assert r['metrics']==metrics
    expected_pass=not L.P.A.numeric_failures(metrics,reg) and not any(counts.values())
    assert r['local_pass']==expected_pass
starts=L.P.S.start_snapshot(model,old,new,scale,reg['primary_k'])
before={ (r['map_id'],r['k']):r for r in diagnostic['starts'] }
transitions=[]
for r in starts:
    previous=before[(r['map_id'],r['k'])]
    if r['classification']!=previous['classification']:
        transitions.append({'map_id':r['map_id'],'k':r['k'],'before':previous,'after':r})
assert digests=={c['run_id']:L.R.digests(net) for c,net,_,_ in prepared}
v2=L.R.load(NAV/'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name,digest in v2.items():
    assert sha(Path(name))==digest,name
report={'schema':'camra-nav065-local-probe-review-v9','timestamp':datetime.now().astimezone().isoformat(),
    'host':socket.getfqdn(),'tmux':os.environ['TMUX'],'python':sys.executable,'base_git':context['base_git'],
    'approval_pin':pin,'candidate':L.P.TARGET,'integrity_pass':True,
    'available_train_episodes':len(old)+len(new),'available_train_steps':sum(len(t.observations) for _,t in old+new),
    'fitted_train_starts':len(keys),'optimizer_trials':steps,'source_models_unchanged':True,
    'all_six_actors_unchanged':True,'source_state_dict_hashes':digests,'diagnostic_state_dict_hashes':after,
    'v2_files_verified':len(v2),'local_before':diagnostic['local_source_evaluation'],'local_after':local,
    'local_gate_pass':local['local_pass'],'full_train_before':diagnostic['full_train_gate'],'full_train_after':gate,
    'full_train_gate_pass':gate['pass'],'start_confusion_before':diagnostic['start_confusion'],
    'start_confusion_after':dict(Counter(r['classification'] for r in starts)),
    'start_transitions':transitions,'full_starts_after':starts,
    'new_rollouts':0,'development_scoring_rounds':0,'scientific_pass':False,'deployment_allowed':False,
    'NAV065':False,'NAV066_started':False,'completed_tasks':52,'original_G4':'FAIL',
    'evidence':{'output':str(out),'output_hashes':{p.name:sha(p) for p in sorted(out.iterdir())},
        'context_sha256':sha(HERE/'context.json'),'run_log_sha256':sha(HERE/'run.log')},
    'limitations':['A selected set of known train mistakes is a local fitting diagnosis, not independent generalization.',
        'One bounded optimizer/subset choice cannot establish architecture impossibility or a unique root cause.',
        'No full-task acceptance, development score, budget search, physical run or deployment was performed.']}
with (NAV/'reports/nav065_local_probe_v9_review.json').open('x') as f:
    json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:report[k] for k in ('integrity_pass','optimizer_trials','local_gate_pass','full_train_gate_pass','start_confusion_before','start_confusion_after')},ensure_ascii=False),flush=True)
for stage in ('local_before','local_after'):
    print(stage,json.dumps({k:v for k,v in report[stage].items() if k!='predictions'}),flush=True)
