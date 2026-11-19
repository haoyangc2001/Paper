#!/usr/bin/env bash
set -eu
cd /home/caohy/repositories/Paper
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
LOG=.worklog-testing/NAV-043/exploration-v1
"$PY" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$LOG/L2-close-plan.log" 2>&1
"$PY" -B - <<'PYTHON' > "$LOG/L2-close-check.log" 2>&1
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == '94f0a2de2f2f6926bdf714c0b52f4f696a65b31f'
def load(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','94f0a2d:'+p]))
new=load(p); original=copy.deepcopy(new)
assert len(old['tasks']) == len(new['tasks']) == 67
for a,b in zip(old['tasks'],new['tasks']):
    if a['id'] == 'NAV-043':
        assert len(b['work_records']) == len(a['work_records'])+1
        assert b['work_records'][:-1] == a['work_records']
        assert b['work_records'][-1]['timestamp'] == '2026-11-19T19:58:40+08:00'
        b['work_records']=a['work_records']
    assert a == b, a['id']
assert old == new
assert sum(t['status'] for t in original['tasks']) == 55
snapshot=load('.worklog-testing/NAV-043/exploration-v1/L2-completed-progress.json')
assert len(snapshot['rows']) == 4 and len(snapshot['source_hashes']) == 22
for path,digest in snapshot['source_hashes'].items(): assert sha(path)==digest,path
assert all(snapshot[k]==0 for k in ('new_applications','new_episodes','W_queries','training_updates'))
assert snapshot['NAV043_passed'] is False and snapshot['G5_passed'] is False
row=next(r for r in snapshot['rows'] if r['level']=='L2_single_detour' and r['method']=='full')
assert row['label']=='timeout' and row['nominal']['feasible']
assert row['measured_joint_work_J'] < row['z0_J'] and row['internal_model_cost_J'] < row['z0_J']
protocol=load('CaMRA/navigation/config/nav043_exploration_v1.json')
assert sha('CaMRA/navigation/config/nav043_exploration_v1.json') == Path('CaMRA/navigation/config/nav043_exploration_v1.json.sha256').read_text().strip()
for path,digest in protocol['sealed_hashes'].items(): assert sha(path)==digest,path
assert sha('CaMRA/navigation/scripts/audit_nav043_exploration.py')=='d56a8c86d293eda3fe7bfc7965d7bb37925f22d14deaca43cec8ef9eca4df42b'
assert sha('CaMRA/navigation/scripts/audit_nav043_exploration_applicability.py')=='fc37986d5a212c2b6b064c2ba1edf1f34f36fac49549f6abf57942292e148fe0'
assert not subprocess.check_output(['git','diff','94f0a2d','--name-only','--ignore-submodules=all','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA','CaMRA/locomotion','CaMRA/calibration'],text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).strip()
print('PASS: one NAV-043 record appended; all definitions/status/history preserved; 22 saved sources and sealed runtime/reviewers unchanged; L2 timeout with both costs below selected budget',flush=True)
PYTHON
git diff --check --ignore-submodules=all > "$LOG/L2-diff-check.log" 2>&1
git add -- AGENTS.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json "$LOG/save_L2_progress.py" "$LOG/L2-completed-progress.json" "$LOG/commit_L2_progress.sh"
git add -f -- "$LOG/L2-completed-progress.log" "$LOG/L2-close-plan.log" "$LOG/L2-close-check.log" "$LOG/L2-diff-check.log"
"$PY" -B - <<'PYTHON'
import subprocess
allowed={'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','doc/plan/20261109_camra_navigation_implementation.json'}
prefix='.worklog-testing/NAV-043/exploration-v1/'
allowed.update(prefix+p for p in ('save_L2_progress.py','L2-completed-progress.json','commit_L2_progress.sh','L2-completed-progress.log','L2-close-plan.log','L2-close-check.log','L2-diff-check.log'))
staged=subprocess.check_output(['git','diff','--cached','--name-only','--ignore-submodules=all'],text=True).splitlines()
assert staged and set(staged) <= allowed,staged
print('STAGED',staged,flush=True)
PYTHON
git -c core.fsmonitor=false commit -m "audit(nav): preserve completed L2 budget exploration outcome"
git rev-parse HEAD
