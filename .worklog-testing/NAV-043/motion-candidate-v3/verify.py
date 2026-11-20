"""Preparation closeout, no coefficient fitting and no application."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import nav043_motion_probe_recovery_guard as g

c.context()
p = c.pin(c.PROPOSAL)
ready = c.pin(c.READINESS)
assert p['status']=='PROPOSED' and p['limits']==c.LIMITS
assert ready['status']=='PREPARED_ONLY' and ready['proposal_sha256']==c.base.sha(c.PROPOSAL)
for group in (p['source_hashes'], ready['source_hashes']):
    for path, digest in group.items():
        assert c.base.sha(path)==digest, path
assert g.protected_sources()=={'protected_identities':295,'existing_raw_input_hashes':90,'bundle_files':29}
assert not c.APPROVAL.exists() and not (c.HERE/'launch.lock').exists()
assert not Path(p['output_root']).exists()
before = set(c.HERE.iterdir())
command = [c.base.PYTHON, '-B', str(c.base.NAV/'scripts/launch_nav043_supported_response_once.py')]
result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True)
assert result.returncode==1 and 'require separate APPROVED' in result.stderr
assert set(c.HERE.iterdir())==before
assert not Path(p['output_root']).exists()
old = json.loads(subprocess.check_output(['git','show','3d4db99:doc/plan/20261109_camra_navigation_implementation.json'],text=True,cwd=ROOT))
current = c.base.load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
for a,b in zip(old['tasks'], current['tasks'], strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-043': assert a==b
assert {k:v for k,v in old.items() if k!='tasks'}=={k:v for k,v in current.items() if k!='tasks'}
assert sum(t['status'] for t in current['tasks'])==55
assert not next(t for t in current['tasks'] if t['id']=='NAV-043')['status']
assert not c.base.load(c.base.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json')['strict_original_completion_gate_passed']
consumers = subprocess.run(['rg','-l','nav043_supported_response',str(c.base.NAV/'src')],
                          capture_output=True,text=True)
assert consumers.returncode==1 and not consumers.stdout
for path in p['source_hashes']:
    assert not path.startswith(('manuscripts/','RC-PPO/','HiRA/'))
c.base.write_new(c.HERE/'verify.json', dict(status='PREPARATION_SOURCE_HISTORY_GUARD_PASS',
    host=socket.gethostname(),pane=os.environ['TMUX_PANE'],python=sys.executable,
    proposal_sha256=c.base.sha(c.PROPOSAL), readiness_sha256=c.base.sha(c.READINESS),
    identities=g.protected_sources(), basic_checks=13, synthetic_solves=3,
    actual_model_solves=0, applications=0, new_episodes=0, W_queries=0, updates=0,
    actual_launcher_missing_approval_exit=result.returncode,
    actual_launcher_created_no_marker=True, original_HELD_preserved=True,
    production_consumers=0, plan_true=55, plan_NAV043=False,
    history_unchanged=True, approved_registration_exists=False,
    formal_output_exists=False))
print('PREPARATION_SOURCE_HISTORY_GUARD_PASS', flush=True)
