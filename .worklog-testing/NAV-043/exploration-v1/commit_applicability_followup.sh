#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/applicability-followup-close-plan.log" 2>&1
"$PY" -B - <<'PYTHON' > "$LOG/applicability-followup-close-check.log" 2>&1
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
root=Path.cwd();nav=root/'CaMRA/navigation'
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','6fa0cd3:'+p]));new=load(p)
for a,b in zip(old['tasks'],new['tasks']):
    if a['id']=='NAV-043':
        assert b['work_records'][:-2]==a['work_records']
        assert len(b['work_records'])==len(a['work_records'])+2
        b['work_records']=a['work_records']
    assert a==b,a['id']
assert old==new
start=load('.worklog-testing/NAV-043/exploration-v1/applicability-followup-start.json')
assert start['original_reviewer_pid']==59048 and start['original_reviewer_start_ticks']=='167163226'
assert start['script_sha256']==sha(nav/'scripts/audit_nav043_exploration_applicability.py')
assert start['waiter_sha256']==sha('.worklog-testing/NAV-043/exploration-v1/applicability_when_reviewed.py')
assert all(start[k]==0 for k in ('new_applications','new_rollouts','W_queries','training_updates'))
compile(Path('.worklog-testing/NAV-043/exploration-v1/applicability_when_reviewed.py').read_text(),'waiter','exec')
protocol=load(nav/'config/nav043_exploration_v1.json')
assert sha(nav/'config/nav043_exploration_v1.json')==Path(str(nav/'config/nav043_exploration_v1.json')+'.sha256').read_text().strip()
for path,digest in protocol['sealed_hashes'].items():assert sha(path)==digest,path
assert sha(nav/'scripts/audit_nav043_exploration.py')=='d56a8c86d293eda3fe7bfc7965d7bb37925f22d14deaca43cec8ef9eca4df42b'
assert not subprocess.check_output(['git','diff','6fa0cd3','--name-only','--ignore-submodules=all','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA','CaMRA/locomotion','CaMRA/calibration'],text=True).strip()
print('PASS: only two NAV-043 records appended; definitions/status/dependencies/history preserved; original sealed sources and reviewers unchanged; actual followup guard/start evidence matches',flush=True)
PYTHON
git diff --check --ignore-submodules=all > "$LOG/applicability-followup-diff-check.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json "$LOG/applicability_when_reviewed.py" "$LOG/applicability-followup-start.json" "$LOG/commit_applicability_followup.sh"
git add -f -- "$LOG/applicability-followup-check.log" "$LOG/applicability-followup-plan.log" "$LOG/applicability-followup-close-plan.log" "$LOG/applicability-followup-close-check.log" "$LOG/applicability-followup-diff-check.log"
"$PY" -B - <<'PYTHON'
import subprocess
allowed={'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','doc/plan/20261109_camra_navigation_implementation.json'}
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert staged and all(p in allowed or p.startswith('.worklog-testing/NAV-043/exploration-v1/') for p in staged),staged
print('STAGED',staged,flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): chain saved applicability review after physical replay"
git rev-parse HEAD
