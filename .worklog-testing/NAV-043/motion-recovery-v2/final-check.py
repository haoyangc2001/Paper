"""Final append-only plan and saved-preparation check, no simulator or data rerun."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path('/home/caohy/repositories/Paper');NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
import nav043_motion_probe_guard as old
import nav043_motion_probe_recovery_guard as G
assert os.environ.get('TMUX') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
p,r=G.pinned(G.PROPOSAL),G.pinned(G.READINESS)
old.approved()
for name,digest in r['source_hashes'].items():
    assert G.sha(ROOT/name)==digest,name
    if name.endswith('.py'): ast.parse((ROOT/name).read_text(),filename=name)
assert not G.APPROVAL.exists() and not (G.EVIDENCE/'launch.lock').exists()
assert not Path(p['output_root']).exists()
report=G.load(NAV/'reports/nav043_motion_probe_recovery_preparation_v2.json')
assert report['status']=='PREPARED_NOT_APPROVED'
assert report['proposal_sha256']==G.sha(G.PROPOSAL)
assert report['readiness_sha256']==G.sha(G.READINESS)
assert report['main_agent_CPU_checks_passed']==20 and not report['new_applications']
assert (G.EVIDENCE/'basic.exit').read_text().strip()=='0'
assert (G.EVIDENCE/'verify.exit').read_text().strip()=='0'
base=json.loads(subprocess.check_output(['git','show','05c52f4:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
now=G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
assert {k:v for k,v in base.items() if k!='tasks'}=={k:v for k,v in now.items() if k!='tasks'}
for b,n in zip(base['tasks'],now['tasks']):
    assert {k:v for k,v in b.items() if k!='work_records'}=={k:v for k,v in n.items() if k!='work_records'}
    assert b['work_records']==n['work_records'][:len(b['work_records'])]
    assert n['id']=='NAV-043' or b['work_records']==n['work_records']
assert sum(t['status'] for t in now['tasks'])==55 and len(now['tasks'])==67
nav043=next(t for t in now['tasks'] if t['id']=='NAV-043')
print('FINAL_PREPARATION_CHECK_PASS',len(nav043['work_records']),'NAV043 records; 55/67 historical; recovery not approved')
