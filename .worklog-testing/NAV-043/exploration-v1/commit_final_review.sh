#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
NAV043_PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
NAV043_LOG=.worklog-testing/NAV-043/exploration-v1
"$NAV043_PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$NAV043_LOG/final-review-plan.log" 2>&1
"$NAV043_PY" -B - <<'PYTHON' > "$NAV043_LOG/final-review-ready.log" 2>&1
import hashlib,json,os,socket,subprocess,sys
from pathlib import Path
from datetime import datetime
base='852683008b270829982d2e4be24cd084212094fa'
prefix=Path('.worklog-testing/NAV-043/exploration-v1')
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert sys.executable=='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==base
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show',f'{base}:{plan}']))
new=load(plan)
for a,b in zip(old['tasks'],new['tasks'],strict=True):
 if a['id']=='NAV-043':
  n=len(a['work_records'])
  assert a['status'] is False and b['status'] is False
  assert b['work_records'][:n]==a['work_records'] and len(b['work_records'])==n+2
  a['work_records']=b['work_records']
 assert a==b,a['id']
assert old==new and sum(t['status'] for t in new['tasks'])==55
review_path=Path('CaMRA/navigation/reports/nav043_exploration_review_v1.json')
margin_path=Path('CaMRA/navigation/reports/nav043_exploration_applicability_v1.json')
assert sha(review_path)=='37ae5020796809ba6c3692904c8d5c999b93f6679c8c302dc50b6814699b99b2'
assert sha(margin_path)=='1f58cfe1608343792cac02a5a9f227bd9d5cb2badc10c26113273363a8f5395c'
close=load(prefix/'exploration-close-basic.json')
assert close['status']=='PASS_SAVED_EXPLORATION_CLOSURE_NOT_G5'
assert close['review_sha256']==sha(review_path) and close['applicability_sha256']==sha(margin_path)
assert close['verification_sha256']==sha(prefix/'verify_exploration_close.py')
assert close['protected_hashes_checked']==295 and close['sealed_hashes_checked']==16 and close['source_hash_entries_checked']==105
assert all(close[k]==0 for k in ('new_applications','new_episodes','new_W_queries','new_fits'))
review,margin=load(review_path),load(margin_path)
assert len(review['episodes'])==len(margin['rows'])==6
assert all(e['complete_saved_physics_event_replay'] for e in review['episodes'])
assert not review['incomplete'] and not margin['incomplete'] and not margin['refusals']
assert review['methods']['full']['physical_safe_timely_budget_success']==1
assert review['methods']['uncalibrated']['physical_safe_timely_budget_success']==0
assert all(m['nominal_feasible']==3 for m in review['methods'].values())
for index in (review['raw_sources'],margin['source_hashes']):
 for p,d in index.items():assert sha(p)==d,p
followup=load(prefix/'applicability-followup-finished.json')
assert followup['review_sha256']==sha(review_path) and followup['output_sha256']==sha(margin_path) and followup['returncode']==0
assert not (prefix/'applicability-followup-error.json').exists()
assert not review['NAV_043_pass'] and not review['G5_pass'] and not margin['NAV043_passed'] and not margin['G5_passed']
for name in ('AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/README.md'):
 previous=subprocess.check_output(['git','show',f'{base}:{name}'],text=True)
 assert Path(name).read_text().endswith(previous.split('\n',1)[1]),name
name='CaMRA/navigation/docs/nav043_exploration_execution_review_20261119.md'
assert Path(name).read_text().endswith(subprocess.check_output(['git','show',f'{base}:{name}'],text=True))
assert not subprocess.check_output(['git','diff',base,'--name-only','--ignore-submodules=all','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA','CaMRA/locomotion','CaMRA/calibration'],text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).strip()
ready={'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),'pane':os.environ.get('TMUX_PANE'),'status':'PASS_FINAL_SAVED_REVIEW_DELIVERY_NOT_G5','baseline_commit':base,'plan_records_appended':2,'tasks_true':55,'closure_sha256':sha(prefix/'exploration-close-basic.json'),'review_sha256':sha(review_path),'applicability_sha256':sha(margin_path),'review_source_hashes_checked':105,'protected_hashes_checked_by_closure':295,'new_applications':0,'new_episodes':0,'new_W_queries':0,'new_model_fits':0,'NAV043_passed':False,'G5_passed':False}
with (prefix/'final-review-ready.json').open('x') as f:json.dump(ready,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(ready,ensure_ascii=False),flush=True)
PYTHON
git diff --check --ignore-submodules=all > "$NAV043_LOG/final-review-diff-check.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_exploration_execution_review_20261119.md CaMRA/navigation/docs/nav043_exploration_final_review_20261119.md CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/reports/nav043_exploration_review_v1.json CaMRA/navigation/reports/nav043_exploration_applicability_v1.json "$NAV043_LOG/applicability-followup-finished.json" "$NAV043_LOG/exploration-close-basic.json" "$NAV043_LOG/final-review-ready.json" "$NAV043_LOG/commit_final_review.sh"
git add -f -- "$NAV043_LOG/saved-review.log" "$NAV043_LOG/applicability-followup.log" "$NAV043_LOG/exploration-close-basic.log" "$NAV043_LOG/final-review-plan.log" "$NAV043_LOG/final-review-ready.log" "$NAV043_LOG/final-review-diff-check.log"
"$NAV043_PY" -B - <<'PYTHON'
import subprocess
allowed={'AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/docs/nav043_exploration_execution_review_20261119.md','CaMRA/navigation/docs/nav043_exploration_final_review_20261119.md','CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md','doc/plan/20261109_camra_navigation_implementation.json','CaMRA/navigation/reports/nav043_exploration_review_v1.json','CaMRA/navigation/reports/nav043_exploration_applicability_v1.json'}
prefix='.worklog-testing/NAV-043/exploration-v1/'
allowed.update(prefix+n for n in ('applicability-followup-finished.json','exploration-close-basic.json','final-review-ready.json','commit_final_review.sh','saved-review.log','applicability-followup.log','exploration-close-basic.log','final-review-plan.log','final-review-ready.log','final-review-diff-check.log'))
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert set(staged)==allowed,staged
print('STAGED_WHITELIST_OK',len(staged),flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): close physical exploration evidence and applicability review"
git rev-parse HEAD
