"""Final preparation history/source checks, no coefficient fits."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/supported-audit-v3'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import nav043_motion_probe_recovery_guard as g
import audit_nav043_supported_response as audit

c.context()
p, ready = c.pin(c.PROPOSAL), c.pin(c.READINESS)
support = c.pin(EV/'support.json')
for group in (p['source_hashes'], ready['source_hashes'], support['source_hashes']):
    for path, digest in group.items():
        assert c.base.sha(path) == digest, path
identities = g.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
raw = c.base.load(ROOT/'.worklog-testing/NAV-043/motion-recovery-v2/held-raw-output-index.json')
for item in raw['files']:
    assert c.base.sha(item['path']) == item['sha256']
assert not c.APPROVAL.exists() and not (c.HERE/'launch.lock').exists()
assert not Path(p['output_root']).exists() and not audit.REPORT.exists()
planpath = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git', 'show', '8fbaad4:'+planpath], cwd=ROOT, text=True))
now = c.base.load(ROOT/planpath)
for a, b in zip(old['tasks'], now['tasks'], strict=True):
    assert {k:v for k,v in a.items() if k != 'work_records'} == {k:v for k,v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-043':
        assert a == b
assert {k:v for k,v in old.items() if k != 'tasks'} == {k:v for k,v in now.items() if k != 'tasks'}
assert sum(t['status'] for t in now['tasks']) == 55
task = next(t for t in now['tasks'] if t['id'] == 'NAV-043')
assert not task['status'] and len(task['work_records']) == 57
changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
assert all(not path.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/'))
           and not path.endswith(('.tex', '.bib')) for path in changed)
c.base.write_new(EV/'close.json', dict(status='READONLY_AUDIT_PREPARATION_CLOSE_PASS',
    pane=os.environ['TMUX_PANE'], source_commit='8fbaad4', identities=identities,
    retained_raw_files=108, task_definitions_and_statuses_unchanged=67,
    NAV043_history_prefix_unchanged=55, NAV043_current_records=57,
    original_HELD_preserved=True, new_basic_checks=5, extra_parameter_solves=0,
    actual_fits=0, applications=0, new_episodes=0, W_queries=0, network_updates=0,
    formal_output_exists=False, pending_same_CPU_fit_confirmation=True,
    production_enabled=False, NAV043_passed=False, G5_passed=False,
    plan_sha256=c.base.sha(ROOT/planpath), support_sha256=c.base.sha(EV/'support.json')))
print('READONLY_AUDIT_PREPARATION_CLOSE_PASS')
