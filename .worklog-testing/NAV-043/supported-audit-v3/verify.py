"""Seal audit-only preparation without data fitting or simulator creation."""
import json
import os
from pathlib import Path
import socket
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
assert c.base.sha(c.PROPOSAL) == 'a3b67ab3943c1b675f5fada4e933bcc8d5906326166a52511785877eaca2d746'
assert c.base.sha(c.READINESS) == '965fbf8e24110be1e1b6603db127b3b86a41d119718361654820d913479f11da'
for group in (p['source_hashes'], ready['source_hashes']):
    for path, digest in group.items():
        assert c.base.sha(path) == digest, path
identities = g.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
raw = c.base.load(ROOT/'.worklog-testing/NAV-043/motion-recovery-v2/held-raw-output-index.json')
for item in raw['files']:
    assert c.base.sha(item['path']) == item['sha256']
assert len(raw['files']) == 108
assert not c.APPROVAL.exists() and not (c.HERE/'launch.lock').exists()
assert not Path(p['output_root']).exists() and not audit.REPORT.exists()
before = {str(path): c.base.sha(path) for path in c.HERE.iterdir() if path.is_file()}
result = subprocess.run([c.base.PYTHON, '-B', str(Path(audit.__file__))], cwd=ROOT,
                        env=os.environ.copy(), capture_output=True, text=True)
assert result.returncode == 1 and 'require separate APPROVED' in result.stderr
c.base.write_new(EV/'actual-refusal.json', dict(exit_code=result.returncode,
    stdout=result.stdout, stderr=result.stderr, no_data_fit=True))
assert before == {str(path): c.base.sha(path) for path in c.HERE.iterdir() if path.is_file()}
assert not audit.REPORT.exists() and not Path(p['output_root']).exists()
assert (EV/'basic.exit').read_text().strip() == '0'
assert 'Ran 5 tests' in (EV/'basic.log').read_text() and '\nOK\n' in (EV/'basic.log').read_text()
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
assert not task['status'] and len(task['work_records']) == 56
review = c.base.load(c.base.NAV/'reports/nav043_motion_probe_recovery_held_review_v2.json')
assert review['case_integrity_passed'] and not review['strict_original_completion_gate_passed']
consumers = subprocess.run(['rg', '-l', 'audit_nav043_supported_response', str(c.base.NAV/'src')],
                          capture_output=True, text=True)
assert consumers.returncode == 1 and not consumers.stdout
sources = [Path(audit.__file__), c.base.NAV/'tests/test_supported_response_audit_v3.py',
           EV/'basic.sh', EV/'basic.log', EV/'basic.exit', EV/'verify.py', EV/'verify.sh']
support = dict(status='PREPARED_ONLY', schema='nav043-supported-response-audit-v3',
    pane=os.environ['TMUX_PANE'], host=socket.gethostname(), python=sys.executable,
    proposal_sha256=c.base.sha(c.PROPOSAL), readiness_sha256=c.base.sha(c.READINESS),
    original_13_checks_sources_unchanged=True, new_main_agent_checks=5, synthetic_parameter_solves=0,
    actual_parameter_solves=0, new_applications=0, new_episodes=0, W_queries=0, network_updates=0,
    identities=identities, retained_raw_files=108, original_HELD_preserved=True,
    actual_missing_approval_exit=1, actual_missing_approval_created_no_marker_or_report=True,
    NAV043=False, G5_passed=False, plan_true=55, source_history_unchanged=True,
    production_enabled=False, pending_same_CPU_fit_confirmation=True,
    source_hashes={str(path.relative_to(ROOT)): c.base.sha(path) for path in sources})
c.base.write_new(EV/'support.json', support)
(EV/'support.json.sha256').write_text(c.base.sha(EV/'support.json')+'\n')
print('READONLY_AUDIT_PREPARATION_PASS', flush=True)
