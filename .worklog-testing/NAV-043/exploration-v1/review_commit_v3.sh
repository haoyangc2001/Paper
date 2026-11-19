#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/review-plan-validation-v3.log" 2>&1
"$PY" -B "$LOG/verify_review_v3.py" > "$LOG/review-verification-v3.log" 2>&1
git diff --check -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/scripts/audit_nav043_exploration.py doc/plan/20261109_camra_navigation_implementation.json > "$LOG/review-diff-check-v3.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md CaMRA/navigation/scripts/audit_nav043_exploration.py doc/plan/20261109_camra_navigation_implementation.json "$LOG/review_first_saved.py" "$LOG/review_first_nominal.py" "$LOG/first-nominal-review.json" "$LOG/review_commit_v3.sh" "$LOG/verify_review_v3.py" "$LOG/review-verification-v3.json"
git add -f -- "$LOG/first-saved-review.log" "$LOG/first-nominal-review.log" "$LOG/review-plan-validation-v3.log" "$LOG/review-verification-v3.log" "$LOG/review-diff-check-v3.log"
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
git -c core.fsmonitor=false commit -m "fix(nav): preserve online scoring during saved-trace audits"
git rev-parse HEAD
