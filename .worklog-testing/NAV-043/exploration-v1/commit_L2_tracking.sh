#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/L2-tracking-plan.log" 2>&1
"$PY" -B - <<'PYTHON' > "$LOG/L2-tracking-close-check.log" 2>&1
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='9d206a3bcb4afea3d2d9c4607272aea1192c5313'
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','9d206a3:'+p]));new=load(p)
assert len(old['tasks'])==len(new['tasks'])==67
assert sum(t['status'] for t in new['tasks'])==55
for a,b in zip(old['tasks'],new['tasks']):
    if a['id']=='NAV-043':
        assert len(b['work_records'])==len(a['work_records'])+2
        assert b['work_records'][:-2]==a['work_records']
        b['work_records']=a['work_records']
    assert a==b,a['id']
assert old==new
report=load('.worklog-testing/NAV-043/exploration-v1/L2-tracking-diagnostic.json')
assert report['status']=='DESCRIPTIVE_COMPLETED_L2_TRACKING_NOT_FINAL_REVIEW'
assert len(report['rows'])==2 and len(report['source_hashes'])==15
assert report['analysis_sha256']==sha('.worklog-testing/NAV-043/exploration-v1/analyze_L2_tracking.py')
for path,digest in report['source_hashes'].items():assert sha(path)==digest,path
assert all(report[k]==0 for k in ('new_applications','new_episodes','W_queries','training_updates','model_fits'))
assert report['NAV043_passed'] is False and report['G5_passed'] is False
for row in report['rows']:
    assert row['physical_label']=='timeout' and row['nominal_label']=='goal_reached'
    for path,digest in row['historical_k1_descriptive_comparison']['source_hashes'].items():assert sha(path)==digest,path
full=next(r for r in report['rows'] if r['method']=='full')
assert full['measured_budget_utilization']<1 and 2.39<full['nominal_slack_to_45s']<2.41
assert not full['prefixes_s']['45']['nominal_complete_to_this_time']
protocol=load('CaMRA/navigation/config/nav043_exploration_v1.json')
assert sha('CaMRA/navigation/config/nav043_exploration_v1.json')==Path('CaMRA/navigation/config/nav043_exploration_v1.json.sha256').read_text().strip()
for path,digest in protocol['sealed_hashes'].items():assert sha(path)==digest,path
assert not subprocess.check_output(['git','diff','9d206a3','--name-only','--ignore-submodules=all','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA','CaMRA/locomotion','CaMRA/calibration'],text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).strip()
print('PASS: only two NAV-043 records appended; all definitions/status/dependencies/history preserved; 15 direct inputs plus old comparison hashes and sealed sources verified; no new trial or model change',flush=True)
PYTHON
git diff --check --ignore-submodules=all > "$LOG/L2-tracking-diff-check.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/docs/nav043_L2_budget_tracking_review_20261119.md doc/plan/20261109_camra_navigation_implementation.json "$LOG/analyze_L2_tracking.py" "$LOG/L2-tracking-diagnostic.json" "$LOG/commit_L2_tracking.sh"
git add -f -- "$LOG/L2-tracking-diagnostic.log" "$LOG/L2-tracking-plan.log" "$LOG/L2-tracking-close-check.log" "$LOG/L2-tracking-diff-check.log"
"$PY" -B - <<'PYTHON'
import subprocess
allowed={'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/docs/nav043_L2_budget_tracking_review_20261119.md','doc/plan/20261109_camra_navigation_implementation.json'}
prefix='.worklog-testing/NAV-043/exploration-v1/'
allowed.update(prefix+p for p in ('analyze_L2_tracking.py','L2-tracking-diagnostic.json','commit_L2_tracking.sh','L2-tracking-diagnostic.log','L2-tracking-plan.log','L2-tracking-close-check.log','L2-tracking-diff-check.log'))
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert staged and set(staged)<=allowed,staged
print('STAGED',staged,flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): quantify saved L2 tracking and deadline gap"
git rev-parse HEAD
