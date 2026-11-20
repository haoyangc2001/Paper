"""Read current task and preparation identity; no fits or applications."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import nav043_motion_probe_recovery_guard as g

c.context()
p=c.base.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
tasks={t['id']:t for t in p['tasks']}
eligible=[t['id'] for t in p['tasks'] if not t['status'] and all(tasks[d]['status'] for d in t['depends_on'])]
assert eligible[0]=='NAV-043' and sum(t['status'] for t in p['tasks'])==55
assert len(tasks['NAV-043']['work_records'])==55 and not tasks['NAV-043']['status']
for key in ('document','global_constraints','uncertainty_policy','execution_alignment_policy','upstream_correction_policy'):
    print(key,json.dumps(p[key],ensure_ascii=False),flush=True)
print('TASK',json.dumps({k:v for k,v in tasks['NAV-043'].items() if k!='work_records'},ensure_ascii=False),flush=True)
for i,record in enumerate(tasks['NAV-043']['work_records']):
    print('RECORD',i,record['summary'],flush=True)
for item in (c.pin(c.PROPOSAL),c.pin(c.READINESS)):
    for path,digest in item['source_hashes'].items():assert c.base.sha(path)==digest,path
assert not c.APPROVAL.exists() and not (c.HERE/'launch.lock').exists()
assert not Path(c.pin(c.PROPOSAL)['output_root']).exists()
assert g.protected_sources()==dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),flush=True)
print('STATUS',subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),flush=True)
print('SUBMODULE',subprocess.check_output(['git','submodule','status'],cwd=ROOT,text=True),flush=True)
c.base.write_new(ROOT/'.worklog-testing/NAV-043/supported-audit-v3/preflight.json',dict(
    status='READONLY_AUDIT_PREPARATION_PREFLIGHT_PASS',pane=os.environ['TMUX_PANE'],
    host=socket.gethostname(),eligible=eligible,task_records=55,plan_true=55,
    current_commit='8fbaad40fa54d407fc10325a4164f41b9cd9587b',
    proposal_sha256=c.base.sha(c.PROPOSAL),readiness_sha256=c.base.sha(c.READINESS),
    pending_same_CPU_fit_approval=True,actual_fits=0,new_applications=0,new_episodes=0,W_queries=0,updates=0))
