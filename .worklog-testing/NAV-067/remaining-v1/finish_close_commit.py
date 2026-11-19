"""Finish the saved review without rewriting raw simulator log whitespace."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
from nav067_sources import load, sha, write_new
import run_nav067_remaining_validation as V1
import run_nav067_remaining_recovery_v2 as V2

assert os.environ.get('TMUX_PANE')=='%599'
assert not V2.APPROVAL.exists()
assert not (V1.EXPECTED_ROOT/'C01/attempt2').exists()
assert not Path('/proc/'+str(load(V1.EXPECTED_ROOT/'C01/PROCESS.json')['pid'])).exists()
proposal=load(V2.PROPOSAL)
for name,digest in load(V1.EXECUTION)['source_hashes'].items():
    assert sha(ROOT/name)==digest
for group in ('source_hashes','evidence_hashes','installed_source_hashes'):
    for name,digest in proposal[group].items():
        assert sha(ROOT/name)==digest
report=load(ROOT/'CaMRA/navigation/reports/nav067_remaining_validation_v1_execution_review.json')
for name,digest in report['failure_evidence_hashes'].items():
    source=Path(name)
    assert sha(source)==digest
    assert sha(HERE/'execution-attempt1'/source.relative_to(V1.EXPECTED_ROOT))==digest
plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
before=json.loads(subprocess.check_output(['git','show','a70214e:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
after=load(plan_path)
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in after.items() if k!='tasks'}
for a,b in zip(before['tasks'],after['tasks'],strict=True):
    if a['id']=='NAV-067':
        assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
        assert b['work_records'][:-1]==a['work_records']
    else:
        assert a==b
assert sum(t['status'] for t in after['tasks'])==54
assert not next(t for t in after['tasks'] if t['id']=='NAV-067')['status']
validator=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update']
subprocess.run(validator,cwd=ROOT,check=True)
raw_log=str((HERE/'execution-attempt1/C01/stdout.log').relative_to(ROOT))
check_log=str((HERE/'close-review.log').relative_to(ROOT))
policy=HERE/'closure-whitespace-policy.json'
write_new(policy,{'at':datetime.now().astimezone().isoformat(),
    'reason':'The first commit check found six trailing-space lines in the byte-preserved raw simulator log. Its captured diagnostic repeats those lines. Preserve both logs unchanged.',
    'excluded_from_whitespace_check_only':[raw_log,check_log],
    'raw_log_sha256':sha(ROOT/raw_log),
    'original_log_sha256':sha(V1.EXPECTED_ROOT/'C01/stdout.log'),
    'all_original_failure_evidence_and_snapshot_hashes_match':True,
    'plan_history_appended_once_only':True,'new_applications':0,'push':False})
allowed={'AGENTS.md','doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/docs/implementation_handoff.md',
    'CaMRA/navigation/docs/nav067_remaining_recovery_v2_20261119.md',
    'CaMRA/navigation/reports/nav067_remaining_validation_v1_execution_review.json',
    'CaMRA/navigation/config/nav067_remaining_validation_recovery_proposal_v2.json',
    *proposal['source_hashes']}
prefix=str(HERE.relative_to(ROOT))+'/'
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert all(p in allowed or p.startswith(prefix) for p in staged),staged
subprocess.run(['git','add','-f','--',str(Path(__file__).relative_to(ROOT)),str(policy.relative_to(ROOT)),str((HERE/'close-review.log').relative_to(ROOT))],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check','--','.',':(exclude)'+raw_log,':(exclude)'+check_log],cwd=ROOT,check=True)
assert not subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True).strip()
subprocess.run(['git','commit','-m','fix(nav): prepare installed-API recovery and preserve NAV-067 failure'],cwd=ROOT,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
write_new(HERE/'closure-commit.json',{'at':datetime.now().astimezone().isoformat(),'head':head,
    'status':'FAILURE_PRESERVED_RECOVERY_PREPARED_NOT_APPROVED','NAV_067_status':False,'completed_tasks':54,
    'proposal_sha256':sha(V2.PROPOSAL),'push':False,'new_recovery_applications':0,
    'raw_log_kept_byte_for_byte':True})
print(json.dumps({'commit':head,'status':'REVIEW_AND_LOCAL_COMMIT_COMPLETE',
    'NAV_067_status':False,'pending':'Explicit one-time C01 recovery and 10 cumulative applications.'},ensure_ascii=False),flush=True)
