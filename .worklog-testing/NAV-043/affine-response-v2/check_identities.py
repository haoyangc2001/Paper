"""Read-only preparation checks. Never call affine.fit_axis or run."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_affine_response as affine

base = affine.base
HERE = ROOT/'.worklog-testing/NAV-043/affine-response-v2'
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
before = json.loads(subprocess.check_output(['git', 'show', '0bd8fae:'+plan_path], cwd=ROOT))
after = base.load(ROOT/plan_path)
assert {k: v for k, v in before.items() if k != 'tasks'} == {k: v for k, v in after.items() if k != 'tasks'}
assert len(before['tasks']) == len(after['tasks']) == 67
for old, new in zip(before['tasks'], after['tasks']):
    assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
    assert new['work_records'][:len(old['work_records'])] == old['work_records']
    if old['id'] != 'NAV-043':
        assert old == new
first = next(t for t in after['tasks'] if t['status'] is False and all(
    next(x for x in after['tasks'] if x['id'] == dependency)['status'] is True for dependency in t['depends_on']))
assert first['id'] == 'NAV-043' and sum(t['status'] for t in after['tasks']) == 55
proposal = base.load(affine.PROPOSAL)
assert base.sha(affine.PROPOSAL) == Path(str(affine.PROPOSAL)+'.sha256').read_text().strip()
for path, digest in proposal['source_hashes'].items():
    assert base.sha(path) == digest
original = base.verify_protocol()
assert not affine.APPROVAL.exists() and not Path(str(affine.APPROVAL)+'.sha256').exists()
assert not Path(proposal['output_root']).exists()
assert not (HERE/'launch.lock').exists() and not (base.NAV/'reports/nav043_affine_response_review_v2.json').exists()
base.write_new(HERE/'identities.json', {'at': datetime.now().astimezone().isoformat(),
    'status': 'PREPARATION_IDENTITY_CHECK_PASS', 'base_commit': '0bd8fae',
    'protected_identities_checked': 295, 'raw_input_hashes_checked': 90,
    'saved_trace_entries': len(original['traces']), 'historical_true': 55,
    'current_task': first['id'], 'new_real_solves': 0, 'new_synthetic_solves': 0,
    'new_physical_episodes': 0, 'new_nominal_episodes': 0, 'W_queries': 0,
    'actor_critic_updates': 0, 'calibration_fits': 0,
    'approval_present': False, 'formal_output_present': False, 'NAV043_passed': False})
print('old history/proposal/source/295 protected/90 raw input identities preserved; no new solve')
