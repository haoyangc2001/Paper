#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/execution-finished-plan.log" 2>&1
"$PY" -B - <<'PYTHON' > "$LOG/execution-finished-ready.log" 2>&1
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_bytes())
base='268c2efe906bb89c54c2846a87002719f728919d'
plan='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show',f'{base}:{plan}']))
new=load(plan)
for a,b in zip(old['tasks'],new['tasks'],strict=True):
 if a['id']=='NAV-043':
  n=len(a['work_records']);assert b['work_records'][:n]==a['work_records'] and len(b['work_records'])==n+2
  a['work_records']=b['work_records']
 assert a==b,a['id']
assert old==new
report=load('.worklog-testing/NAV-043/exploration-v1/execution-finished-progress.json')
assert report['status']=='EXECUTION_FINISHED_FULL_EVENT_REVIEW_PENDING'
assert len(report['rows'])==6 and report['explicit_physical_steps']==54581
assert report['measurement_steps']==53381 and report['saved_nominal_steps']==2257
assert report['initial_W_queries']==30 and not report['NAV043_passed'] and not report['G5_passed']
for p,d in report['source_hashes'].items():assert sha(p)==d,p
assert sha('.worklog-testing/NAV-043/exploration-v1/save_completed_allocation.py')==report['analysis_sha256']
protocol=load('CaMRA/navigation/config/nav043_exploration_v1.json')
for p,d in protocol['sealed_hashes'].items():assert sha(p)==d,p
for name in ('save_completed_allocation.py','verify_exploration_close.py'):
 p=Path('.worklog-testing/NAV-043/exploration-v1')/name;compile(p.read_text(),str(p),'exec')
proposal=Path('CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md').read_text()
assert 'PROPOSED' in proposal and '尚未执行' in proposal
assert not subprocess.check_output(['git','diff',base,'--name-only','--ignore-submodules=all','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA','CaMRA/locomotion','CaMRA/calibration'],text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).strip()
ready={'at':datetime.now().astimezone().isoformat(),'status':'PASS_EXECUTION_SNAPSHOT_AND_PROPOSAL_READY_NOT_FULL_REVIEW','plan_records_appended':2,'tasks_true':55,'raw_source_hashes_checked':len(report['source_hashes']),'sealed_sources_checked':len(protocol['sealed_hashes']),'full_event_review_completed':False,'new_applications':0,'new_episodes':0,'new_W_queries':0,'new_model_fits':0,'NAV043_passed':False,'G5_passed':False}
with Path('.worklog-testing/NAV-043/exploration-v1/execution-finished-ready.json').open('x') as f:json.dump(ready,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(ready,ensure_ascii=False),flush=True)
PYTHON
git diff --check --ignore-submodules=all > "$LOG/execution-finished-diff-check.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_exploration_execution_review_20261119.md CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md doc/plan/20261109_camra_navigation_implementation.json "$LOG/save_completed_allocation.py" "$LOG/verify_exploration_close.py" "$LOG/execution-finished-progress.json" "$LOG/execution-finished-ready.json" "$LOG/commit_execution_finished.sh"
git add -f -- "$LOG/execution-finished-progress.log" "$LOG/execution-finished-ready.log" "$LOG/execution-finished-plan.log" "$LOG/execution-finished-diff-check.log"
"$PY" -B - <<'PYTHON'
import subprocess
allowed={'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/docs/nav043_exploration_execution_review_20261119.md','CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md','doc/plan/20261109_camra_navigation_implementation.json'}
prefix='.worklog-testing/NAV-043/exploration-v1/'
allowed.update(prefix+p for p in ('save_completed_allocation.py','verify_exploration_close.py','execution-finished-progress.json','execution-finished-ready.json','commit_execution_finished.sh','execution-finished-progress.log','execution-finished-ready.log','execution-finished-plan.log','execution-finished-diff-check.log'))
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert staged and set(staged)<=allowed,staged
print('STAGED',staged,flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): save completed physical exploration and next route"
git rev-parse HEAD
