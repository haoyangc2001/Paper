"""Verify the pending-approval preflight without fitting or invoking its shell."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/coupled-response-v4'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import run_nav043_coupled_response as c
import nav043_motion_probe_recovery_guard as g

c.context()
p = c.core.old.pin(c.PROPOSAL)
ready = c.core.old.pin(c.READINESS)
assert c.base.sha(c.PROPOSAL) == 'f7b5ba4a39adc655f4cc06f2d30a7412e76ae749dc0efd8cc6aa92c813f9b7e7'
assert c.base.sha(c.READINESS) == '8dc215b35cd71b04a4d67bdebb05ac48db150cd842097f14cc0dc533720ee4c1'
assert not c.APPROVAL.exists() and not Path(p['output_root']).exists()
for source in (p['source_hashes'], ready['source_hashes']):
    for path, digest in source.items():
        assert c.base.sha(path) == digest, path
names = ('launch.lock', 'preflight.json', 'preflight.log', 'preflight.exit')
assert all(not (EV/name).exists() for name in names)
assert not (c.base.NAV/'reports/nav043_coupled_response_review_v4.json').exists()
before = {str(path): c.base.sha(path) for path in EV.iterdir() if path.is_file()}
refusal = subprocess.run([sys.executable, '-B', str(EV/'preflight.py')], cwd=ROOT,
                         env=os.environ.copy(), text=True, capture_output=True)
assert refusal.returncode == 1 and 'require separate APPROVED' in refusal.stderr
assert before == {str(path): c.base.sha(path) for path in EV.iterdir() if path.is_file()}
assert all(not (EV/name).exists() for name in names)
identities = g.protected_sources()
assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
cases = c.load_cases()
assert len(cases) == len(c.reference_rows()) == 30
path = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git', 'show', '79bd77a:'+path], cwd=ROOT, text=True))
now = c.base.load(ROOT/path)
assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in now.items() if k != 'tasks'}
for a, b in zip(old['tasks'], now['tasks'], strict=True):
    assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-043':
        assert a == b
assert sum(task['status'] for task in now['tasks']) == 55
assert not next(task for task in now['tasks'] if task['id'] == 'NAV-043')['status']
validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', path, '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert validation.returncode == 0, validation.stdout + validation.stderr
raw = (EV/'preflight.py').read_text()
assert raw.index('proposal = candidate.approved()') < raw.index('processes = subprocess.check_output')
assert raw.index('proposal = candidate.approved()') < raw.index('candidate.load_cases()')
assert raw.index('proposal = candidate.approved()') < raw.index("candidate.base.write_new(EV/'preflight.json'")
c.base.write_new(EV/'preflight-preparation.json', dict(status='PREPARED_ONLY_PREFLIGHT_REFUSAL_VERIFIED',
    host=socket.gethostname(), pane=os.environ['TMUX_PANE'], python=sys.executable,
    proposal_sha256=c.base.sha(c.PROPOSAL), readiness_sha256=c.base.sha(c.READINESS),
    identities=identities, saved_cases=30, original_62_records_and_all_task_definitions_preserved=True,
    actual_preflight_not_run=True, actual_approved_branch_not_verified=True,
    refusal=dict(exit_code=refusal.returncode, stdout=refusal.stdout, stderr=refusal.stderr),
    no_marker_or_report_created_by_refusal=True, legacy_validation=validation.stdout,
    real_coefficient_solves=0, synthetic_coefficient_solves=0, applications=0, episodes=0,
    W_queries=0, network_updates=0, calibration_fits=0, NAV043_passed=False, G5_passed=False,
    source_hashes={str(path.relative_to(ROOT)):c.base.sha(path)
                   for path in (EV/'preflight.py', EV/'preflight.sh', EV/'prepare-preflight.py')}))
print('PREPARED_ONLY_PREFLIGHT_REFUSAL_VERIFIED; zero solves; APPROVED absent', flush=True)
