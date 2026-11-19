#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/applicability-plan-validation-v1.log" 2>&1
"$PY" -B - <<'PYTHON' > "$LOG/applicability-close-check-v1.log" 2>&1
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
old=json.loads(subprocess.check_output(['git','show','56b4163:'+p]))
new=load(p)
for a,b in zip(old['tasks'],new['tasks']):
    if a['id']=='NAV-043':
        assert b['work_records'][:-2]==a['work_records']
        assert len(b['work_records'])==len(a['work_records'])+2
        b['work_records']=a['work_records']
    assert a==b,a['id']
assert old==new
report=load('.worklog-testing/NAV-043/exploration-v1/applicability-ready-v1.json')
assert report['script_sha256']==sha(nav/'scripts/audit_nav043_exploration_applicability.py')
assert report['verifier_sha256']==sha('.worklog-testing/NAV-043/exploration-v1/verify_applicability_ready.py')
assert report['premature_production_call_refused_without_output'] and len(report['L1_rows'])==2
pair=load(nav/'config/nav066_decision_prereg_v2.json')
protected={str(nav/k):v for k,v in pair['implementation_hashes'].items()}
protected.update(pair['protected_hashes'])
protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
for path,digest in protected.items():assert sha(path)==digest,path
assert not subprocess.check_output(['git','diff','56b4163','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA','CaMRA/locomotion','CaMRA/calibration'],text=True).strip()
print('PASS: two NAV-043 appends only; task definitions/status/order/history preserved; real saved checks match source; protected identities',len(protected),flush=True)
PYTHON
git diff --check > "$LOG/applicability-diff-check-v1.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/scripts/audit_nav043_exploration_applicability.py "$LOG/verify_applicability_ready.py" "$LOG/applicability-ready-v1.json" "$LOG/commit_applicability_ready.sh"
git add -f -- "$LOG/applicability-ready-v1.log" "$LOG/applicability-premature-refusal.log" "$LOG/applicability-plan-validation-v1.log" "$LOG/applicability-close-check-v1.log" "$LOG/applicability-diff-check-v1.log"
"$PY" -B - <<'PYTHON'
import subprocess
allowed={'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','doc/plan/20261109_camra_navigation_implementation.json','CaMRA/navigation/scripts/audit_nav043_exploration_applicability.py'}
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert staged and all(p in allowed or p.startswith('.worklog-testing/NAV-043/exploration-v1/') for p in staged),staged
print('STAGED',staged,flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): prepare selected-budget margin applicability review"
git rev-parse HEAD
