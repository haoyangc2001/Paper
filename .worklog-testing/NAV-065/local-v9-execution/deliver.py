"""Validate append-only closure, preserve evidence, and commit the authorized result."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper');NAV=ROOT/'CaMRA/navigation';HERE=Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert sys.executable=='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
r=read(NAV/'reports/nav065_local_probe_v9_review.json');base=r['base_git']
assert r['integrity_pass'] and git('rev-parse','HEAD')==base
plan='doc/plan/20261109_camra_navigation_implementation.json'
before=json.loads(subprocess.check_output(['git','show',base+':'+plan],cwd=ROOT));current=read(ROOT/plan)
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in current.items() if k!='tasks'}
for old,new in zip(before['tasks'],current['tasks'],strict=True):
    if old['id']=='NAV-065':
        assert {k:v for k,v in old.items() if k!='work_records'}=={k:v for k,v in new.items() if k!='work_records'}
        assert new['work_records'][:-2]==old['work_records']
        assert len(new['work_records'])==len(old['work_records'])+2
    else:assert old==new
states={t['id']:t['status'] for t in current['tasks']}
assert len(states)==67 and sum(states.values())==52
assert next(t['id'] for t in current['tasks'] if not t['status'] and all(states[x] for x in t['depends_on']))=='NAV-065'
protocol=read(NAV/'config/nav065_local_probe_prereg_v9.json')
assert sha(NAV/'config/nav065_local_probe_prereg_v9.json')==r['approval_pin']
for name,digest in protocol['implementation_hashes'].items():assert sha(NAV/name)==digest
out=Path(r['evidence']['output'])
for name,digest in r['evidence']['output_hashes'].items():assert sha(out/name)==digest
assert sha(HERE/'context.json')==r['evidence']['context_sha256']
assert sha(HERE/'run.log')==r['evidence']['run_log_sha256']
v2=read(NAV/'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name,digest in v2.items():assert sha(Path(name))==digest
with (HERE/'validation-close.log').open('x') as f:
    result=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(ROOT/plan),'--mode','update'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
assert result.returncode==0
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--check'],cwd=ROOT,check=True)
protected=['RC-PPO','HiRA','manuscripts','CaMRA/locomotion','CaMRA/calibration']
assert not git('diff','--name-only','--',*protected)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'HiRA',text=True).strip()
closure={'timestamp':datetime.now().astimezone().isoformat(),'base_git':base,
    'legacy_validation_pass':True,'plan_old_fields_and_history_preserved':True,'new_records':2,
    'integrity_pass':True,'local_pass':False,'full_train_pass':False,'scientific_pass':False,
    'completed_tasks':52,'next_task':'NAV-065','NAV066_started':False,'optimizer_updates':400,
    'new_rollouts':0,'development_scoring_rounds':0,'all_six_original_models_unchanged':True,
    'all_six_actors_unchanged':True,'v2_files_verified':len(v2),
    'evidence':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/plan,NAV/'reports/nav065_local_probe_v9_review.json',NAV/'docs/nav065_local_probe_v9_review_20261118.md',HERE/'validation-close.log')}}
with (HERE/'closure.json').open('x') as f:json.dump(closure,f,ensure_ascii=False,indent=2);f.write('\n')
files=['AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',plan,
    'CaMRA/navigation/config/nav065_local_probe_prereg_v9.json','CaMRA/navigation/config/nav065_local_probe_prereg_v9.json.sha256',
    'CaMRA/navigation/docs/nav065_local_probe_v9_review_20261118.md','CaMRA/navigation/reports/nav065_local_probe_v9_review.json']
files += [str((HERE/name).relative_to(ROOT)) for name in ('prepare.py','execute.py','review.py','review-attempt1.py','review-attempt1.log','close.py','deliver.py',
    'context.json','started.json','process.json','closure.json','validation-before.log','validation-open.log','validation-close.log','run.log')]
assert not git('diff','--cached','--name-only'),'preserve existing index'
subprocess.run(['git','add','-f','--',*files],cwd=ROOT,check=True)
assert set(git('diff','--cached','--name-only').splitlines())==set(files)
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Record NAV-065 v9 local probe failure and full-train regression'],cwd=ROOT,check=True)
receipt={'commit':git('rev-parse','HEAD'),'parent':base,'files':files,'pushed':False}
with (HERE/'commit.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'commit':receipt['commit'],'files':len(files),'integrity_pass':True,'local_pass':False,'full_train_pass':False,'completed_tasks':52,'pushed':False}),flush=True)
