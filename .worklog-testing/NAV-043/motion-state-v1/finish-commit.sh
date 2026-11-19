#!/usr/bin/env bash
set -euo pipefail
cd /home/caohy/repositories/Paper
task_python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
task_evidence=.worklog-testing/NAV-043/motion-state-v1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
test ! -e "$task_evidence/commit-preflight.log"
"$task_python" -B - <<'PY' > "$task_evidence/commit-preflight.log" 2>&1
import json,os,socket,subprocess
from pathlib import Path
root=Path.cwd()
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
evidence=root/'.worklog-testing/NAV-043/motion-state-v1'
audit=json.loads((evidence/'preparation-review.json').read_bytes())
import hashlib
for path,digest in audit['source_hashes'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
before=json.loads(subprocess.check_output(['git','show','10d9259:doc/plan/20261109_camra_navigation_implementation.json']))
now=json.loads((root/'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
for a,b in zip(before['tasks'],now['tasks']):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    n=len(a['work_records'])
    assert a['work_records']==b['work_records'][:n]
    assert len(b['work_records'])-n==(2 if a['id']=='NAV-043' else 0)
assert len(now['tasks'])==67 and sum(t['status'] for t in now['tasks'])==55
print('PASS: unchanged verified sources, 67 definitions/statuses and all history; 2 NAV-043 records only. Raw pytest logs/XML retained byte-for-byte; whitespace lint excludes only raw log/XML evidence in this directory.')
PY
"$task_python" -B /home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update > "$task_evidence/plan-at-commit.log" 2>&1
git add -f .worklog-testing/NAV-043/motion-state-v1
git diff --cached --check -- . ':(exclude).worklog-testing/NAV-043/motion-state-v1/*.log' ':(exclude).worklog-testing/NAV-043/motion-state-v1/*.xml'
git diff --cached --stat
git commit -m 'NAV-043: prepare causal motion state and record affine support limits'
git status --short
git log -1 --format='%h %s'
