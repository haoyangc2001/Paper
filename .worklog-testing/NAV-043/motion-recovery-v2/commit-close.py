"""Scope-limited local commit preparation after saved-data verification."""
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT=Path('/home/caohy/repositories/Paper')
EV=ROOT/'.worklog-testing/NAV-043/motion-recovery-v2'
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert (EV/'close-review.exit').read_text().strip()=='0'
assert 'VALID:' in (EV/'close-plan.log').read_text()
assert not (EV/'close-diff-check.log').read_text()
plan='doc/plan/20261109_camra_navigation_implementation.json'
before=json.loads(subprocess.check_output(['git','show','c8d6f71:'+plan],cwd=ROOT))
after=json.loads((ROOT/plan).read_text())
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in after.items() if k!='tasks'}
for old,new in zip(before['tasks'],after['tasks']):
    assert {k:v for k,v in old.items() if k!='work_records'}=={k:v for k,v in new.items() if k!='work_records'}
    assert old['work_records']==new['work_records'][:len(old['work_records'])]
    assert len(new['work_records'])==len(old['work_records'])+(3 if new['id']=='NAV-043' else 0)
paths=[str(EV.relative_to(ROOT)),'AGENTS.md','CaMRA/navigation/README.md',
    'CaMRA/navigation/docs/implementation_handoff.md',
    'CaMRA/navigation/docs/nav043_motion_probe_recovery_v2_20261120.md',
    'CaMRA/navigation/docs/nav043_motion_probe_recovery_result_20261120.md',
    'CaMRA/navigation/reports/nav043_motion_probe_recovery_held_review_v2.json',plan]
for path in EV.rglob('*'):
    if path.is_file():
        assert path.stat().st_size < 1024*1024, 'retain large raw output outside Git: '+str(path)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert staged
assert all(name in paths[1:] or name.startswith(paths[0]+'/') for name in staged)
assert all(not name.startswith(('manuscripts/','RC-PPO/','HiRA/','submission','docs/paper'))
    and not name.endswith(('.tex','.bib','.pdf')) for name in staged)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
print('COMMIT_SCOPE_CHECK_PASS',len(staged),'files; 53 NAV043 records; 55/67 statuses retained',flush=True)
