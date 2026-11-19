"""Final scoped consistency checks; no fits, new episodes or protected writes."""
from pathlib import Path
import json
import os
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = ROOT/'.worklog-testing/NAV-043/response-residual-v1'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_response as m

assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
protocol = m.verify_protocol()
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
plan = m.load(ROOT/plan_path)
old = json.loads(subprocess.check_output(['git','show','fb4b574:'+plan_path],cwd=ROOT))
for a,b in zip(old['tasks'],plan['tasks']):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
assert sum(t['status'] for t in plan['tasks'])==55 and len(plan['tasks'])==67
proposed = ROOT/'CaMRA/navigation/config/nav043_affine_response_proposal_v2.json'
assert m.sha(proposed)==Path(str(proposed)+'.sha256').read_text().strip()
p = m.load(proposed)
for path,digest in p['source_hashes'].items(): assert m.sha(ROOT/path)==digest,path
assert not (ROOT/'CaMRA/navigation/config/nav043_affine_response_approval_v2.json').exists()
assert not Path(p['output_root']).exists()
r = m.load(ROOT/'CaMRA/navigation/reports/nav043_response_residuals_v1.json')
assert len(r['rows'])==18 and r['saved_ticks']==158797
assert all(x['actual_minus_one_step_response']['mean'][2]<0. for x in r['rows'])
assert r['maximum_accounting_closure_m']<1.e-9
command = [sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',plan_path,'--mode','update']
subprocess.run(command,cwd=ROOT,check=True)
m.write_new(HERE/'close.json',{'host':socket.gethostname(),'pane':os.environ['TMUX_PANE'],
    'python':sys.executable,'protected_checked':295,'input_hashes_checked':90,
    'historical_task_definitions_records_and_statuses_unchanged':True,
    'real_solves':0,'new_episodes':0,'new_W_queries':0,'new_network_updates':0,
    'proposal_pin':m.sha(proposed),'approval_exists':False,'formal_run_exists':False,
    'NAV043_passed':False,'G5_passed':False})
