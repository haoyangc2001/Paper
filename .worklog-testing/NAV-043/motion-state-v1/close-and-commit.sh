#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-state-v1
export PYTHONPATH=/home/caohy/repositories/Paper/CaMRA/navigation/src:/home/caohy/repositories/Paper/CaMRA/calibration/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
test ! -e "$task_evidence/final-check.exit"
if "$task_python" -B - <<'PY' > "$task_evidence/final-check.log" 2>&1
import json, os, socket, subprocess, sys
from pathlib import Path
root=Path.cwd()
evidence=root/'.worklog-testing/NAV-043/motion-state-v1'
sys.path.insert(0,str(root/'CaMRA/navigation/scripts'))
import nav043_affine_response as affine
base=affine.base
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve()==Path(base.PYTHON).resolve()
affine.approval()
base.verify_protocol()
audit=base.load(evidence/'preparation-review.json')
assert audit['status']=='ISOLATED_PREPARATION_CHECKS_PASS'
for path,digest in audit['source_hashes'].items():
    assert base.sha(root/path)==digest,path
before=json.loads(subprocess.check_output(['git','show','10d9259:doc/plan/20261109_camra_navigation_implementation.json']))
now=base.load(root/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
assert len(now['tasks'])==67 and sum(t['status'] for t in now['tasks'])==55
for a,b in zip(before['tasks'],now['tasks']):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    n=len(a['work_records'])
    assert a['work_records']==b['work_records'][:n]
    assert len(b['work_records'])-n==(2 if a['id']=='NAV-043' else 0)
    for record in b['work_records'][n:]:
        assert set(record)=={'timestamp','summary','actions','files_changed','commands','results','evidence','remaining_work','blockers'}
assert not subprocess.check_output(['git','diff','--cached','--name-only']).strip()
assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=D']).strip()
subprocess.run(['git','diff','--check'],check=True)
print('PASS: final sources match audited 22-check artifacts; 295/90 identities preserved; 67 definitions and all old records retained, exactly two NAV-043 records appended; false/G5 and 55/67 unchanged.')
print('host',socket.gethostname(),'pane',os.environ['TMUX_PANE'],'python',sys.executable)
PY
then
  printf '0\n' > "$task_evidence/final-check.exit"
else
  task_status=$?
  printf '%s\n' "$task_status" > "$task_evidence/final-check.exit"
  exit "$task_status"
fi
"$task_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_evidence/plan-final.log" 2>&1
git add AGENTS.md CaMRA/navigation/README.md CaMRA/navigation/docs/implementation_handoff.md doc/plan/20261109_camra_navigation_implementation.json CaMRA/navigation/src/camra_navigation/motion_state.py CaMRA/navigation/tests/test_motion_state.py CaMRA/navigation/config/motion_state_candidate_v1.json CaMRA/navigation/reports/nav043_affine_support_review_v1.json CaMRA/navigation/docs/nav043_motion_state_preparation_20261120.md
git add -f .worklog-testing/NAV-043/motion-state-v1
git diff --cached --check
git diff --cached --stat
git commit -m 'NAV-043: prepare causal motion state and record affine support limits'
git status --short
git log -1 --format='%h %s'
