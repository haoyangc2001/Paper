"""Read source identities and historical invariants; never fit or run physics."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = ROOT/'.worklog-testing/NAV-043/response-residual-v1'
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
plan = json.loads((ROOT/plan_path).read_bytes())
old = json.loads(subprocess.check_output(['git','show','fb4b574:'+plan_path],cwd=ROOT))
assert len(plan['tasks'])==67 and sum(t['status'] for t in plan['tasks'])==55
by_id = {t['id']:t for t in plan['tasks']}
first = next(t for t in plan['tasks'] if not t['status'] and all(by_id[d]['status'] for d in t['depends_on']))
assert first['id']=='NAV-043'
for before,after in zip(old['tasks'],plan['tasks']):
    assert before['id']==after['id']
    assert {k:v for k,v in before.items() if k!='work_records'}=={k:v for k,v in after.items() if k!='work_records'}
    assert after['work_records'][:len(before['work_records'])]==before['work_records']
paths = ['CaMRA/navigation/scripts/nav043_affine_response.py',
         'CaMRA/navigation/scripts/nav043_motion_response.py',
         'CaMRA/navigation/scripts/analyze_nav043_response_residuals.py',
         'CaMRA/navigation/reports/nav043_motion_response_review_v1.json',
         'CaMRA/navigation/reports/nav043_response_residuals_v1.json',
         'CaMRA/navigation/config/navigation_horizon_v2.json']
metadata = {'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),
    'pane':os.environ['TMUX_PANE'],'python':sys.executable,'plan_version':plan['document']['version'],
    'first_eligible':first['id'],'old_task_definitions_and_records_preserved':True,
    'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
with (HERE/'source-identities.json').open('x') as stream:
    json.dump(metadata,stream,indent=2); stream.write('\n')
