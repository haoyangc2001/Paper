"""Verify final append-only state and create the authorized local commit. No push."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m

m.context()
ev=ROOT/'.worklog-testing/NAV-043/v4-support-review'
assert (ev/'final-verify-v2.exit').read_text().strip()=='0'
verified=m.base.load(ev/'final-verify.json')
assert verified['status']=='SAVED_REVIEW_AND_MATCHED_PREPARATION_PASS'
for path,digest in verified['source_hashes'].items():
    assert m.base.sha(ROOT/path)==digest
p,ready=[m.diagonal.pin(path) for path in (m.PROPOSAL,m.READINESS)]
for mapping in (p['source_hashes'],ready['source_hashes']):
    for path,digest in mapping.items():
        assert m.base.sha(path)==digest,path
assert not m.APPROVAL.exists() and not Path(p['output_root']).exists() and not (m.HERE/'launch.lock').exists()
plan_path='doc/plan/20261109_camra_navigation_implementation.json'
prior=json.loads(subprocess.check_output(['git','show','f187b6f:'+plan_path],cwd=ROOT,text=True))
now=m.base.load(ROOT/plan_path)
assert {k:v for k,v in prior.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
for a,b in zip(prior['tasks'],now['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']=='NAV-043':
        assert len(a['work_records'])==66 and len(b['work_records'])==69
    else:
        assert a==b
validation=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
    plan_path,'--mode','update'],cwd=ROOT,text=True,capture_output=True)
assert validation.returncode==0,validation.stdout+validation.stderr
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
m.base.write_new(ev/'post-record.json',dict(status='FINAL_APPEND_ONLY_69_RECORDS_PASS',pane=os.environ['TMUX_PANE'],
    original_records=66,current_records=69,all_task_definitions_and_states_preserved=True,
    validation=validation.stdout,final_verifier_failure_and_source_retained=True,
    proposal_sha256=m.base.sha(m.PROPOSAL),readiness_sha256=m.base.sha(m.READINESS),
    source_hashes_rechecked=True,new_fits=0,NAV043=False,G5_passed=False))
paths=['AGENTS.md','README.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
    plan_path,'CaMRA/navigation/docs/nav043_v4_support_review_20261120.md',
    'CaMRA/navigation/scripts/analyze_nav043_v4_support.py','CaMRA/navigation/scripts/nav043_matched_support_v5.py',
    'CaMRA/navigation/reports/nav043_v4_support_review_v1.json',
    'CaMRA/navigation/config/nav043_matched_support_proposal_v5.json',
    'CaMRA/navigation/config/nav043_matched_support_proposal_v5.json.sha256',
    'CaMRA/navigation/config/nav043_matched_support_readiness_v5.json',
    'CaMRA/navigation/config/nav043_matched_support_readiness_v5.json.sha256']
directories=['.worklog-testing/NAV-043/v4-support-review','.worklog-testing/NAV-043/matched-support-v5']
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','add','-f','--',*directories],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(paths).issubset(staged)
assert all(path in paths or any(path.startswith(prefix+'/') for prefix in directories) for path in staged)
assert not any('__pycache__' in path or path.endswith('.pyc') for path in staged)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','NAV-043: review v4 support gaps and prepare matched-data diagnostic'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
status=subprocess.check_output(['git','status','--short','--ignore-submodules=all'],cwd=ROOT,text=True)
assert not status,status
print('LOCAL_COMMIT',commit,'FILES',len(staged),'WORKTREE_CLEAN',flush=True)
