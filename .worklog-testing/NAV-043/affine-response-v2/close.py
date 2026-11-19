"""Final preparation-only history and scope review; no fits or data replay."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import launch_nav043_affine_response_once as launcher
base, here = launcher.base, launcher.HERE
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
before = json.loads(subprocess.check_output(['git', 'show', '0bd8fae:'+plan_path], cwd=ROOT))
after = base.load(ROOT/plan_path)
assert {k: v for k, v in before.items() if k != 'tasks'} == {k: v for k, v in after.items() if k != 'tasks'}
for old, new in zip(before['tasks'], after['tasks']):
    assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
    assert new['work_records'][:len(old['work_records'])] == old['work_records']
    assert len(new['work_records'])-len(old['work_records']) == (2 if old['id'] == 'NAV-043' else 0)
identities = base.load(here/'identities.json')
assert identities['protected_identities_checked'] == 295 and identities['raw_input_hashes_checked'] == 90
ready = base.load(launcher.READINESS)
assert base.sha(launcher.READINESS) == Path(str(launcher.READINESS)+'.sha256').read_text().strip()
assert ready['status'] == 'PREPARED_ONLY'
assert base.sha(launcher.affine.PROPOSAL) == ready['proposal_sha256']
for path, digest in ready['source_hashes'].items():
    assert base.sha(path) == digest
for path, digest in base.load(launcher.affine.PROPOSAL)['source_hashes'].items():
    assert base.sha(path) == digest
assert not launcher.affine.APPROVAL.exists() and not Path(ready['formal_output']).exists()
assert not (here/'launch.lock').exists() and not (base.NAV/'reports/nav043_affine_response_review_v2.json').exists()
assert (here/'refusal.exit').read_text().strip() == 'launcher=1 auditor=1 no marker/output/report'
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert not any(p.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/'))
               or p.endswith(('.tex', '.bib', '.pdf')) for p in changed)
assert all(p.stat().st_size < 1_000_000 for p in here.rglob('*') if p.is_file())
base.write_new(here/'close.json', {'at': datetime.now().astimezone().isoformat(),
    'status': 'PREPARATION_CLOSE_PASS', 'task': 'NAV-043', 'plan_version': after['document']['version'],
    'historical_true': sum(t['status'] for t in after['tasks']),
    'task_definitions_and_history_preserved': True, 'source_pins_preserved': True,
    'basic_checks': 20, 'new_real_solves': 0, 'new_synthetic_solves': 0,
    'new_business_episodes': 0, 'W_queries': 0, 'network_updates': 0, 'calibration_fits': 0,
    'approval_present': False, 'full_saved_output_audit_performed': False,
    'changed_tracked_files': changed, 'NAV043_passed': False, 'G5_passed': False})
print('preparation close PASS; all old task definitions/status/history and source pins preserved')
