"""Read-only checks before the separately approved response-v4 attempt.

No coefficients are solved and no once-only execution markers are created.
The exact APPROVED registration is checked before data, reports or processes.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/coupled-response-v4'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import run_nav043_coupled_response as candidate
import nav043_motion_probe_recovery_guard as guard


def main():
    candidate.context()
    proposal = candidate.approved()
    assert candidate.base.sha(candidate.PROPOSAL) == 'f7b5ba4a39adc655f4cc06f2d30a7412e76ae749dc0efd8cc6aa92c813f9b7e7'
    assert candidate.base.sha(candidate.READINESS) == '8dc215b35cd71b04a4d67bdebb05ac48db150cd842097f14cc0dc533720ee4c1'
    for name in ('launch.lock', 'launch.json', 'exit.json', 'run.log', 'preflight.json'):
        assert not (EV/name).exists(), name
    assert not Path(proposal['output_root']).exists()
    assert not (candidate.base.NAV/'reports/nav043_coupled_response_review_v4.json').exists()
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    active = [line for line in processes.splitlines() if 'python' in line and (
        'launch_nav043_coupled_response_once.py' in line or
        'import run_nav043_coupled_response as r; r.run' in line)]
    assert not active, active
    identities = guard.protected_sources()
    assert identities == dict(protected_identities=295, existing_raw_input_hashes=90, bundle_files=29)
    cases = candidate.load_cases()
    reference = candidate.reference_rows()
    folds = candidate.folds(cases)
    assert len(cases) == len(reference) == 30 and len(folds) == 3
    assert sum(case.trace.ticks for case in cases) == 182797
    evidence = candidate.base.load(EV/'verify.json')
    assert evidence['status'] == 'PREPARED_ONLY_VERIFIED'
    assert evidence['proposal_sha256'] == candidate.base.sha(candidate.PROPOSAL)
    assert evidence['readiness_sha256'] == candidate.base.sha(candidate.READINESS)
    assert (EV/'final-basic.exit').read_text().strip() == '0'
    assert 'Ran 11 tests' in (EV/'final-basic.log').read_text()
    path = 'doc/plan/20261109_camra_navigation_implementation.json'
    old = json.loads(subprocess.check_output(['git', 'show', '79bd77a:'+path], cwd=ROOT, text=True))
    now = candidate.base.load(ROOT/path)
    assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in now.items() if k != 'tasks'}
    for a, b in zip(old['tasks'], now['tasks'], strict=True):
        assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
        assert b['work_records'][:len(a['work_records'])] == a['work_records']
        if a['id'] != 'NAV-043':
            assert a == b
    states = {task['id']: task['status'] for task in now['tasks']}
    eligible = [task['id'] for task in now['tasks'] if not task['status']
                and all(states[d] for d in task['depends_on'])]
    assert eligible == ['NAV-043'] and sum(states.values()) == 55
    validation = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', path, '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
    assert validation.returncode == 0, validation.stdout + validation.stderr
    candidate.base.write_new(EV/'preflight.json', dict(status='PASS_BEFORE_ONE_APPROVED_V4_ATTEMPT',
        host=socket.gethostname(), pane=os.environ['TMUX_PANE'], python=sys.executable,
        proposal_sha256=candidate.base.sha(candidate.PROPOSAL),
        readiness_sha256=candidate.base.sha(candidate.READINESS),
        approval_sha256=candidate.base.sha(candidate.APPROVAL), identities=identities,
        cases=30, measurement_ticks=182797, folds=3,
        old_62_records_and_all_task_definitions_preserved=True,
        no_existing_output_or_process_or_lock=True, original_basic_sources_valid=True,
        actual_coefficient_solves=0, synthetic_coefficient_solves=0, applications=0,
        episodes=0, W_queries=0, network_updates=0, calibration_fits=0,
        NAV043_passed=False, G5_passed=False, plan_true=55, validation=validation.stdout,
        source_hashes={str(p.relative_to(ROOT)):candidate.base.sha(p)
                       for p in (EV/'preflight.py', EV/'preflight.sh', candidate.APPROVAL)}))
    print('PREFLIGHT_PASS; no coefficient solve or execution performed', flush=True)


if __name__ == '__main__':
    main()
