"""Read-only NAV-043 continuation context and saved source checks; no fits."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/motion-candidate-v3'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_probe_recovery_guard as G
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
p=G.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
tasks={t['id']:t for t in p['tasks']}
eligible=[t['id'] for t in p['tasks'] if not t['status'] and all(tasks[d]['status'] for d in t['depends_on'])]
assert eligible[0]=='NAV-043' and sum(t['status'] for t in p['tasks'])==55
assert not tasks['NAV-043']['status'] and len(tasks['NAV-043']['work_records'])==53
print('PLAN_FULL_JSON', json.dumps(p,ensure_ascii=False),flush=True)
assert G.protected_sources()=={'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
G.approved()
review=G.load(G.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json')
assert review['case_integrity_passed'] and not review['strict_original_completion_gate_passed']
index=G.load(ROOT/review['raw_file_index'])
for item in index['files']:
    assert Path(item['path']).stat().st_size==item['bytes'] and G.sha(item['path'])==item['sha256']
assert len(index['files'])==108
assert all(not (Path('/proc')/str(pid)).exists() for pid in review['registered_processes_absent'])
print('SOURCE_STATUS',json.dumps({'host':socket.gethostname(),'python':sys.executable,
    'pane':os.environ['TMUX_PANE'],'eligible':eligible,'plan_true':55,
    'protected':G.protected_sources(),'raw_files':108,'old_HELD_preserved':True,
    'new_applications':0,'fits':0,'W_queries':0,'updates':0},ensure_ascii=False),flush=True)
print('HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),flush=True)
print('GIT_STATUS',subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),flush=True)
print('SUBMODULE_STATUS',subprocess.check_output(['git','submodule','status'],cwd=ROOT,text=True),flush=True)
G.persist(EV/'preflight.json',{'status':'READONLY_CONTINUATION_PREFLIGHT_PASS',
    'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'eligible':'NAV-043',
    'plan_true':55,'NAV043_records':53,'source_commit':'3d4db993f87ef5a388a1c318c62b208ae731cf6f',
    'identities':G.protected_sources(),'raw_files_rechecked':108,
    'physical_processes_absent':True,'old_HELD_preserved':True,
    'new_applications':0,'fits':0,'W_queries':0,'updates':0})
