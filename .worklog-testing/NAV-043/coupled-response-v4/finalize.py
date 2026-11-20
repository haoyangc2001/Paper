"""Verify completed preparation without parameter solves or actual execution."""
from datetime import datetime
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

candidate.context()
proposal = candidate.core.old.pin(candidate.PROPOSAL)
ready = candidate.core.old.pin(candidate.READINESS)
assert proposal['status'] == 'PROPOSED' and ready['status'] == 'PREPARED_ONLY'
assert ready['proposal_sha256'] == candidate.base.sha(candidate.PROPOSAL)
for group in (proposal['source_hashes'], ready['source_hashes']):
    for path, digest in group.items():
        assert candidate.base.sha(path) == digest, path
assert not candidate.APPROVAL.exists()
assert not (EV/'launch.lock').exists()
assert not Path(proposal['output_root']).exists()
assert not (candidate.base.NAV/'reports/nav043_coupled_response_review_v4.json').exists()
assert (EV/'verify.exit').read_text().strip() == '0'
assert (EV/'final-basic.exit').read_text().strip() == '0'
verified = json.loads((EV/'verify.json').read_text())
assert verified['status'] == 'PREPARED_ONLY_VERIFIED'
assert verified['proposal_sha256'] == candidate.base.sha(candidate.PROPOSAL)
assert verified['readiness_sha256'] == candidate.base.sha(candidate.READINESS)
identities = guard.protected_sources()
assert identities == verified['identities']
cases = candidate.load_cases()
references = candidate.reference_rows()
assert len(cases) == len(references) == 30
assert sum(case.trace.ticks for case in cases) == 182797
path = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git', 'show', 'a59b0e2:'+path], cwd=ROOT, text=True))
now = json.loads((ROOT/path).read_text())
assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in now.items() if k != 'tasks'}
for a, b in zip(old['tasks'], now['tasks'], strict=True):
    assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-043':
        assert a == b
task = next(task for task in now['tasks'] if task['id'] == 'NAV-043')
assert len(task['work_records']) == 62 and not task['status']
assert sum(task['status'] for task in now['tasks']) == 55
states = {task['id']: task['status'] for task in now['tasks']}
eligible = [task['id'] for task in now['tasks'] if not task['status']
            and all(states[d] for d in task['depends_on'])]
assert eligible == ['NAV-043']
result = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', path, '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert result.returncode == 0, result.stdout + result.stderr
consumers = subprocess.run(['rg', '-l', 'nav043_coupled_response', 'CaMRA/navigation/src'], cwd=ROOT, text=True, capture_output=True)
assert consumers.returncode == 1 and not consumers.stdout
diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, text=True, capture_output=True)
assert diff.returncode == 0, diff.stdout + diff.stderr
changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
assert set(changed) == {'AGENTS.md', 'README.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', path}
assert all(p.startswith('.worklog-testing/NAV-043/coupled-response-v4/')
           or (p.startswith('CaMRA/navigation/') and ('coupled_response' in p or 'coupled_response' in Path(p).name))
           for p in untracked), untracked
for filename, title in [('AGENTS.md', '# Agent 接手说明'), ('README.md', '# 四足机器人导航研发工作区'),
                        ('CaMRA/navigation/README.md', '# 上层最小成本到达避障导航器'),
                        ('CaMRA/navigation/docs/implementation_handoff.md', '# 上层导航实施交接')]:
    previous = subprocess.check_output(['git', 'show', 'HEAD:'+filename], cwd=ROOT, text=True)
    text = (ROOT/filename).read_text()
    assert previous.startswith(title) and text.startswith(title)
    assert text.endswith(previous[len(title):]), filename
docs = ROOT/'CaMRA/navigation/docs/nav043_coupled_response_preparation_20261120.md'
assert candidate.base.sha(candidate.PROPOSAL) in docs.read_text()
assert candidate.base.sha(candidate.READINESS) in docs.read_text()
files = sorted(set(changed + untracked))
candidate.base.write_new(EV/'finalize.json', dict(status='PREPARATION_FINAL_REVIEW_PASS',
    at=datetime.now().astimezone().isoformat(), host=socket.gethostname(), pane=os.environ['TMUX_PANE'],
    python=sys.executable, proposal_sha256=candidate.base.sha(candidate.PROPOSAL),
    readiness_sha256=candidate.base.sha(candidate.READINESS), identities=identities,
    cases=30, measurement_ticks=182797, all_old_60_records_and_67_definitions_preserved=True,
    work_records=62, true_tasks=55, eligible=eligible, original_docs_suffixes_preserved=True,
    basic_checks_reused_not_rerun=11, real_coefficient_solves=0, new_applications=0,
    new_episodes=0, W_queries=0, network_updates=0, calibration_fits=0,
    new_approval_present=False, candidate_production_consumer=False,
    NAV043_passed=False, G5_passed=False, legacy_validation=result.stdout,
    changed_file_index=[dict(path=p, bytes=(ROOT/p).stat().st_size, sha256=candidate.base.sha(ROOT/p)) for p in files]))
print('PREPARATION_FINAL_REVIEW_PASS; no actual solves or new approval; legacy VALID', flush=True)
