"""Seal the explicit approval and inspect inputs; no fit or trajectory replay."""
from datetime import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import launch_nav043_affine_response_once as launcher

base, here = launcher.base, launcher.HERE
pin = Path(str(launcher.affine.APPROVAL)+'.sha256')
with pin.open('x') as stream:
    stream.write(base.sha(launcher.affine.APPROVAL)+'\n')
proposal, approval = launcher.prepare()
protocol = base.verify_protocol()
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
before = json.loads(subprocess.check_output(['git', 'show', '965db56:'+plan_path], cwd=ROOT))
after = base.load(ROOT/plan_path)
assert {k: v for k, v in before.items() if k != 'tasks'} == {k: v for k, v in after.items() if k != 'tasks'}
assert len(before['tasks']) == len(after['tasks']) == 67
for old, new in zip(before['tasks'], after['tasks']):
    assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
    assert new['work_records'][:len(old['work_records'])] == old['work_records']
    assert len(new['work_records'])-len(old['work_records']) == (1 if old['id'] == 'NAV-043' else 0)
assert sum(t['status'] for t in after['tasks']) == 55
assert not (here/'launch.lock').exists() and not Path(proposal['output_root']).exists()
base.write_new(here/'approval-preflight.json', {'at': datetime.now().astimezone().isoformat(),
    'host': socket.gethostname(), 'pane': __import__('os').environ['TMUX_PANE'],
    'python': sys.executable, 'proposal_sha256': base.sha(launcher.affine.PROPOSAL),
    'approval_sha256': base.sha(launcher.affine.APPROVAL),
    'readiness_sha256': base.sha(launcher.READINESS),
    'protected_identities_checked': 295, 'raw_input_hashes_checked': 90,
    'trace_entries': len(protocol['traces']), 'source_and_history_check': 'PASS',
    'actual_new_solves_before_launch': 0, 'new_business_episodes': 0,
    'W_queries': 0, 'network_updates': 0, 'calibration_fits': 0,
    'NAV043_passed': False, 'G5_passed': False})
print('approved exact proposal; registered host/tmux/python, 295 protected, 90 inputs and history PASS; no fit')
