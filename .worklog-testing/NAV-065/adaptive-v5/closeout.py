"""Validate append-only records and archive only this authorized preparation."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
OUT = ROOT / '.worklog-testing/NAV-065/adaptive-v5'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads((OUT / 'before-plan.json').read_text())
new = json.loads(plan_path.read_text())
assert [t['id'] for t in old['tasks']] == [t['id'] for t in new['tasks']]
for a, b in zip(old['tasks'], new['tasks']):
    assert {k:v for k,v in a.items() if k != 'work_records'} == {k:v for k,v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-065':
        assert a == b
    else:
        assert len(b['work_records']) == len(a['work_records']) + 2
for k in old.keys() - {'tasks'}:
    assert old[k] == new[k]
assert sum(t['status'] for t in new['tasks']) == 52
assert (ROOT / 'CaMRA/navigation/docs/implementation_handoff.md').read_text().splitlines()[2].startswith('## 2026-11-18 NAV-065 v5')
validation = [sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(plan_path), '--mode', 'update']
with (OUT / 'plan-validation-final.log').open('x') as log:
    subprocess.run(validation, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--check'], cwd=ROOT, check=True)
report = {'timestamp': datetime.now().astimezone().isoformat(), 'host':socket.getfqdn(),
          'tmux':os.environ['TMUX'], 'plan_history_preserved':True, 'validation_pass':True,
          'completed_tasks':52, 'NAV_065_status':False, 'new_formal_sampling':0, 'new_formal_updates':0,
          'new_development_scoring':0, 'stage':'preparation only; numeric quotas unapproved'}
with (OUT / 'closeout.json').open('x') as stream:
    json.dump(report, stream, indent=2)
paths = ['AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
         'doc/plan/20261109_camra_navigation_implementation.json',
         'CaMRA/navigation/scripts/refit_nav065_adaptive.py',
         'CaMRA/navigation/tests/test_nav065_adaptive_refit.py',
         'CaMRA/navigation/config/nav065_adaptive_refit_proposal_v5.json',
         'CaMRA/navigation/docs/nav065_adaptive_refit_v5_20261118.md',
         'CaMRA/navigation/reports/nav065_train_coverage_v5_attempt3.json',
         'CaMRA/navigation/reports/nav065_adaptive_refit_v5_preparation_review.json',
         '.worklog-testing/NAV-065/adaptive-v5']
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert all(any(p == owned or p.startswith(owned+'/') for owned in paths) for p in staged)
subprocess.run(['git','-c','core.whitespace=cr-at-eol','diff','--cached','--check'], cwd=ROOT, check=True)
subprocess.run(['git','commit','-m','nav: prepare bounded train boundary sampling and critic refit v5'],cwd=ROOT,check=True)
print('COMMIT',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),flush=True)
