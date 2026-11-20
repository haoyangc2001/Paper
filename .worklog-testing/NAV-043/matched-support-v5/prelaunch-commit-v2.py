"""Retain raw failure logs while checking source whitespace and sealing approval."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m

m.context()
p=m.approved()
assert (m.HERE/'preflight.exit').read_text().strip()=='0'
preflight=m.base.load(m.HERE/'preflight.json')
assert preflight['status']=='APPROVED_MATCHED_SUPPORT_PREFLIGHT_PASS'
assert preflight['approval_sha256']==m.base.sha(m.APPROVAL)
assert not Path(p['output_root']).exists() and not (m.HERE/'launch.lock').exists() and not m.REVIEW.exists()
plan_path='doc/plan/20261109_camra_navigation_implementation.json'
prior=json.loads(subprocess.check_output(['git','show','e776675:'+plan_path],cwd=ROOT,text=True))
current=m.base.load(ROOT/plan_path)
assert {k:v for k,v in prior.items() if k!='tasks'}=={k:v for k,v in current.items() if k!='tasks'}
for a,b in zip(prior['tasks'],current['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']=='NAV-043':
        assert len(a['work_records'])==69 and len(b['work_records'])==70
    else:
        assert a==b
validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',plan_path,'--mode','update'],cwd=ROOT,text=True,capture_output=True)
assert validation.returncode==0,validation.stdout+validation.stderr
paths=['AGENTS.md','README.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',plan_path,
    str(m.APPROVAL.relative_to(ROOT)),str(m.APPROVAL.relative_to(ROOT))+'.sha256']
prefix=str(m.HERE.relative_to(ROOT))+'/'
staged_before=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert all(x in paths or x.startswith(prefix) for x in staged_before),staged_before
prior_log=Path('/tmp/nav043-v5-prelaunch-commit.log').read_bytes()
assert b'new blank line at EOF' in prior_log and b'CalledProcessError' in prior_log
with (m.HERE/'prelaunch-commit-v1-failed.log').open('xb') as stream:
    stream.write(prior_log)
m.base.write_new(m.HERE/'prelaunch-commit-check-v2.json',dict(status='APPROVED_PREFLIGHT_COMMIT_CHECK_PASS',
    pane=os.environ['TMUX_PANE'],proposal_sha256=m.base.sha(m.PROPOSAL),readiness_sha256=m.base.sha(m.READINESS),
    approval_sha256=m.base.sha(m.APPROVAL),plan_records=70,plan_true=55,NAV043=False,
    output_present=False,launch_present=False,new_fits=0,validation=validation.stdout,
    original_failure_log_retained=True,raw_log_whitespace_not_modified=True,
    whitespace_scope='strict for every staged non-log file; original raw log bytes preserved'))
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','add','-f','--',str(m.HERE.relative_to(ROOT))],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(paths).issubset(staged)
assert all(x in paths or x.startswith(prefix) for x in staged),staged
assert not any('__pycache__' in x or x.endswith('.pyc') for x in staged)
source_paths=[x for x in staged if not x.endswith('.log')]
subprocess.run(['git','diff','--cached','--check','--',*source_paths],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','NAV-043: approve matched-support v5 diagnostic and seal preflight'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
assert not status,status
print('PRELAUNCH_LOCAL_COMMIT',commit,'FILES',len(staged),'WORKTREE_CLEAN',flush=True)
