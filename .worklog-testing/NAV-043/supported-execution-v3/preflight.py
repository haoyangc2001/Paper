"""One authorized saved-data attempt: identity/absence checks, no fitting."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/supported-execution-v3'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import nav043_motion_probe_recovery_guard as g
import audit_nav043_supported_response as audit

c.context()
p = c.approved()
assert c.base.sha(c.PROPOSAL) == 'a3b67ab3943c1b675f5fada4e933bcc8d5906326166a52511785877eaca2d746'
assert c.base.sha(c.READINESS) == '965fbf8e24110be1e1b6603db127b3b86a41d119718361654820d913479f11da'
support = ROOT/'.worklog-testing/NAV-043/supported-audit-v3/support.json'
assert c.base.sha(support) == 'fb66bed401c00420bdd692c9f6f9386cf9260082e4957ef9b37d026df0f3f0d6'
for path, digest in c.pin(support)['source_hashes'].items():
    assert c.base.sha(path) == digest, path
identities = g.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
cases = c.load_cases()
groups = c.folds(cases)
assert len(groups) == 3 and len(cases) == 30
assert sum(x.trace.ticks for x in cases) == 182797
assert not Path(p['output_root']).exists() and not audit.REPORT.exists()
for name in ('launch.lock', 'launch.json', 'exit.json', 'run.log'):
    assert not (c.HERE/name).exists(), name
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
active = [line for line in processes.splitlines()
          if ('python' in line and ('launch_nav043_supported_response_once.py' in line
                                   or 'c.run(c.approved()' in line))]
assert not active, active
path = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git', 'show', '8b29243:'+path], text=True))
now = c.base.load(ROOT/path)
assert {k:v for k,v in old.items() if k != 'tasks'} == {k:v for k,v in now.items() if k != 'tasks'}
for a,b in zip(old['tasks'], now['tasks'], strict=True):
    assert {k:v for k,v in a.items() if k != 'work_records'} == {k:v for k,v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-043':
        assert a == b
task = next(t for t in now['tasks'] if t['id'] == 'NAV-043')
assert len(task['work_records']) == 59 and not task['status']
assert sum(t['status'] for t in now['tasks']) == 55
byid = {t['id']:t for t in now['tasks']}
assert next(t['id'] for t in now['tasks'] if not t['status']
            and all(byid[d]['status'] for d in t['depends_on'])) == 'NAV-043'
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', path, '--mode', 'update'], text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout+validation.stderr
c.base.write_new(EV/'preflight.json', dict(status='PASS_BEFORE_ONE_AUTHORIZED_ATTEMPT',
    host=socket.gethostname(), pane=os.environ['TMUX_PANE'], python=sys.executable,
    proposal_sha256=c.base.sha(c.PROPOSAL), readiness_sha256=c.base.sha(c.READINESS),
    approval_sha256=c.base.sha(c.APPROVAL), identities=identities,
    cases=30, measurement_ticks=182797, folds=3,
    fold_sizes=[dict(level=level,train=len(train),heldout=len(held)) for level,train,held in groups],
    original_tests_source_valid=True, audit_support_sources_valid=True,
    original_HELD_preserved=True, old_58_records_unchanged=True,
    no_existing_output_or_process_or_lock=True, actual_parameter_solves=0,
    applications=0, episodes=0, W_queries=0, network_updates=0,
    NAV043=False, G5_passed=False, plan_true=55, validation=validation.stdout,
    source_hashes={str(path.relative_to(ROOT)):c.base.sha(path) for path in (EV/'preflight.py',EV/'preflight.sh',c.APPROVAL)}))
print('PREFLIGHT_PASS; no coefficient solve; launch not yet performed', flush=True)
