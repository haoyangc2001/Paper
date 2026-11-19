#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/review-update-plan-validation-v2.log" 2>&1
"$PY" -B - <<'PYTHON' > "$LOG/review-append-check-v2.log" 2>&1
import json
from pathlib import Path
import subprocess
p='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','73b482c:'+p],text=True))
new=json.loads(Path(p).read_text())
for a,b in zip(old['tasks'],new['tasks']):
    assert a['id']==b['id'] and a['status']==b['status']
    if a['id']=='NAV-043':
        assert b['work_records'][:-1]==a['work_records']
        assert len(b['work_records'])==len(a['work_records'])+1
        b['work_records']=a['work_records']
    assert a==b,a['id']
assert old==new
compile(Path('CaMRA/navigation/scripts/audit_nav043_exploration.py').read_text(),'audit','exec')
print('PASS: only one NAV-043 append; all task definitions, dependencies, statuses and history retained; syntax checked.')
PYTHON
git diff --check -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/scripts/audit_nav043_exploration.py doc/plan/20261109_camra_navigation_implementation.json > "$LOG/review-diff-check-v2.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/scripts/audit_nav043_exploration.py doc/plan/20261109_camra_navigation_implementation.json "$LOG/check_reviewer.py" "$LOG/review-ready.json" "$LOG/commit_review.sh"
git add -f -- "$LOG/review-ready.log" "$LOG/review-ready-v2.log" "$LOG/review-ready-plan-validation.log" "$LOG/review-update-plan-validation.log" "$LOG/review-append-check.log" "$LOG/review-diff-check.log" "$LOG/review-update-plan-validation-v2.log" "$LOG/review-append-check-v2.log" "$LOG/review-diff-check-v2.log" "$LOG/running-commit.log"
"$PY" -B - <<'PYTHON'
import subprocess
allowed={
 'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md',
 'CaMRA/navigation/scripts/audit_nav043_exploration.py',
 'doc/plan/20261109_camra_navigation_implementation.json',
}
prefix='.worklog-testing/NAV-043/exploration-v1/'
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert staged and all(p in allowed or p.startswith(prefix) for p in staged),staged
assert not any(p.endswith(('.tex','.bib','.pdf')) or p.startswith(('manuscripts/','RC-PPO/','HiRA/')) for p in staged)
print('STAGED',staged,flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): replay saved nominal actions and preserve partial accounting"
git rev-parse HEAD
