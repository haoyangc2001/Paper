"""Verify append-only closeout and commit only the authorized v5 evidence."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_adaptive_refit_launch_20261118_v5')
WORKLOG = ROOT / '.worklog-testing/NAV-065/adaptive-v5-execution'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
review = json.loads((LAUNCH / 'review.json').read_text())
assert review['execution_integrity_pass'] and not review['scientific_pass']
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads((LAUNCH / 'before-plan.json').read_text())
new = json.loads(plan_path.read_text())
assert old.keys() == new.keys()
for key in old.keys() - {'tasks'}:
    assert old[key] == new[key]
assert [t['id'] for t in old['tasks']] == [t['id'] for t in new['tasks']]
for a,b in zip(old['tasks'],new['tasks']):
    if a['id'] != 'NAV-065':
        assert a == b
    else:
        assert not b['status']
        assert {k:v for k,v in a.items() if k!='work_records'} == {k:v for k,v in b.items() if k!='work_records'}
        assert b['work_records'][:-1] == a['work_records']
assert sum(t['status'] for t in new['tasks']) == 52
command = [sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(plan_path), '--mode', 'update']
with (LAUNCH/'validate-final.log').open('x') as stream:
    subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--check'],cwd=ROOT,check=True)
report = {'time':datetime.now().astimezone().isoformat(),'host':socket.getfqdn(),'tmux':os.environ['TMUX'],
    'plan_history_preserved':True,'legacy_plan_validation_pass':True,'scientific_pass':False,
    'NAV_065_status':False,'completed_tasks':52,'NAV_066_started':False,
    'review':'CaMRA/navigation/reports/nav065_adaptive_refit_v5_review.json',
    'run_status':review['run_status'],'new_train_episodes':review['new_train_episodes'],
    'optimizer_trials':review['optimizer_trials'],'accepted_updates':review['accepted_updates'],
    'development_scoring_rounds':review['development_scoring_rounds']}
for path in (WORKLOG/'closeout.json',LAUNCH/'finalize.json'):
    with path.open('x') as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False);stream.write('\n')
owned = ['AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
    'CaMRA/navigation/docs/nav065_adaptive_refit_v5_review_20261118.md',
    'CaMRA/navigation/reports/nav065_adaptive_refit_v5_review.json',
    'doc/plan/20261109_camra_navigation_implementation.json',
    '.worklog-testing/NAV-065/adaptive-v5-execution']
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
changed = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert all(not p.startswith(('manuscripts/','docs/paper','submission')) and not p.endswith(('.tex','.bib')) for p in changed)
subprocess.run(['git','add','--',*owned],cwd=ROOT,check=True)
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert all(any(p==x or p.startswith(x+'/') for x in owned) for p in staged)
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check'],cwd=ROOT,check=True)
with (LAUNCH/'result-commit.log').open('x') as stream:
    subprocess.run(['git','commit','-m','nav: preserve adaptive boundary refit v5 result and gate evidence'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
print('COMMIT',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),flush=True)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
