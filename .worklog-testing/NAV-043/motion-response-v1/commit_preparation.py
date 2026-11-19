"""Final append-only/whitelist review and authorized local commit; no push."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
HERE=Path(__file__).resolve().parent
BASE='77cd7376c200bf8a18bf4672e75d040f01b7e7a4'
assert Path.cwd()==ROOT and os.environ.get('TMUX')
assert socket.gethostname().split('.')[0]=='tiger11'
assert sys.executable=='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
def command(args):return subprocess.check_output(args,text=True).strip()
def load(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert command(['git','rev-parse','HEAD'])==BASE
assert not command(['git','diff','--cached','--name-only'])
assert (HERE/'preparation-close.exit').read_text().strip()=='0'
ready=load(HERE/'preparation-ready.json')
assert ready['status']=='PASS_PREPARATION_ONLY_REAL_FIT_NOT_STARTED'
for path,digest in ready['evidence_hashes'].items():assert sha(path)==digest
plan_path='doc/plan/20261109_camra_navigation_implementation.json'
plan=load(ROOT/plan_path)
old=json.loads(command(['git','show',BASE+':'+plan_path]))
task=next(t for t in plan['tasks'] if t['id']=='NAV-043')
original=next(t for t in old['tasks'] if t['id']=='NAV-043')
assert len(task['work_records'])==27 and task['work_records'][:25]==original['work_records']
clone=json.loads(json.dumps(plan))
next(t for t in clone['tasks'] if t['id']=='NAV-043')['work_records']=original['work_records']
assert clone==old and sum(t['status'] for t in plan['tasks'])==55 and task['status'] is False
protocol=ROOT/'CaMRA/navigation/config/nav043_motion_response_proposal_v1.json'
proposal=load(protocol)
assert sha(protocol)==ready['proposal_sha256']
assert not Path(proposal['approval_required']).exists()
assert not Path(proposal['output_root']).exists()
allow={plan_path,'AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
       'CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md',
       'CaMRA/navigation/docs/nav043_motion_response_preparation_20261119.md',
       'CaMRA/navigation/config/nav043_motion_response_proposal_v1.json',
       'CaMRA/navigation/config/nav043_motion_response_proposal_v1.json.sha256',
       'CaMRA/navigation/scripts/nav043_motion_response.py','CaMRA/navigation/tests/test_nav043_motion_response.py'}
folder=str(HERE.relative_to(ROOT))+'/'
changed=command(['git','diff','--name-only','--ignore-submodules=all',BASE]).splitlines()
untracked=command(['git','ls-files','--others','--exclude-standard']).splitlines()
assert all(p in allow or p.startswith(folder) for p in changed+untracked),(changed,untracked)
subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',plan_path,'--mode','update'],check=True)
subprocess.run(['git','diff','--check','--ignore-submodules=all'],check=True)
final={'status':'PASS_PREPARATION_APPEND_ONLY_AND_LOCAL_COMMIT_READY_NOT_G5',
       'base_commit':BASE,'NAV043_records_before':25,'NAV043_records_after':27,'true_tasks':55,
       'protocol_sha256':sha(protocol),'preparation_ready_sha256':sha(HERE/'preparation-ready.json'),
       'real_model_solves':0,'approval_exists':False,'formal_output_exists':False,
       'NAV043_passed':False,'G5_passed':False}
with (HERE/'final-basic.json').open('x') as stream:json.dump(final,stream,ensure_ascii=False,indent=2);stream.write('\n')
evidence=[str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file() and p.suffix not in ('.log',)]
subprocess.run(['git','add','--',*sorted(allow),*sorted(evidence)],check=True)
logs=[str(HERE/name) for name in ('basic.log','proposal.log','preflight.log','preparation-close.log','commit-before-format-fix.log')]
subprocess.run(['git','add','-f','--',*logs],check=True)
staged=command(['git','diff','--cached','--name-only']).splitlines()
assert all(p in allow or p.startswith(folder) for p in staged)
subprocess.run(['git','diff','--cached','--check'],check=True)
subprocess.run(['git','commit','-m','feat(nav): prepare guarded saved-motion response diagnostic'],check=True)
print(json.dumps({'commit':command(['git','rev-parse','HEAD']),'push':False,'real_model_solves':0}),flush=True)
