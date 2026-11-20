"""Commit the finished, approved local preparation; no real diagnostic execution."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/coupled-response-v4'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import run_nav043_coupled_response as c

c.context()
assert (EV/'finalize.exit').read_text().strip() == '0'
report = c.base.load(EV/'finalize.json')
assert report['status'] == 'PREPARATION_FINAL_REVIEW_PASS'
assert not c.APPROVAL.exists() and not (EV/'launch.lock').exists()
proposal = c.core.old.pin(c.PROPOSAL)
assert not Path(proposal['output_root']).exists()
assert report['proposal_sha256'] == c.base.sha(c.PROPOSAL)
assert report['readiness_sha256'] == c.base.sha(c.READINESS)
for source in (proposal['source_hashes'], c.core.old.pin(c.READINESS)['source_hashes']):
    for path, digest in source.items():
        assert c.base.sha(path) == digest, path
changed_index = report['changed_file_index']
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
for item in changed_index:
    if item['path'] != plan_path:
        assert (ROOT/item['path']).stat().st_size == item['bytes'], item['path']
        assert c.base.sha(ROOT/item['path']) == item['sha256'], item['path']
# Correct only the current uncommitted closing record's mistyped timestamp.
plan = c.base.load(ROOT/plan_path)
record = next(t for t in plan['tasks'] if t['id'] == 'NAV-043')['work_records'][-1]
assert record['timestamp'] == '2026-11-20T12:50:21+08:00'
before_time_correction = json.loads(json.dumps(plan))
next(t for t in before_time_correction['tasks'] if t['id'] == 'NAV-043')['work_records'][-1]['timestamp'] = '2026-11-20T13:10:00+08:00'
import hashlib
encoded = (json.dumps(before_time_correction, ensure_ascii=False, indent=2)+'\n').encode()
original = next(i for i in changed_index if i['path'] == plan_path)
assert hashlib.sha256(encoded).hexdigest() == original['sha256']
assert len(encoded) == original['bytes']
result = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', plan_path, '--mode', 'update'], cwd=ROOT, text=True, capture_output=True)
assert result.returncode == 0, result.stdout + result.stderr
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
evidence_files = sorted(p for p in EV.rglob('*') if p.is_file()
                        and p.name not in ('commit.log', 'commit.exit', 'commit-result.json'))
c.base.write_new(EV/'precommit.json', dict(status='LOCAL_PREPARATION_READY_TO_COMMIT',
    at=datetime.now().astimezone().isoformat(), pane=os.environ['TMUX_PANE'],
    corrected_only_current_draft_closing_timestamp=True, old_committed_history_unchanged=True,
    finalization_other_index_entries_match=True, legacy_validation=result.stdout,
    actual_solves=0, new_approval_present=False, NAV043_passed=False, G5_passed=False,
    evidence_index=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=c.base.sha(p))
                    for p in evidence_files],
    current_plan_sha256=c.base.sha(ROOT/plan_path)))
paths = ['AGENTS.md', 'README.md', 'CaMRA/navigation/README.md',
         'CaMRA/navigation/docs/implementation_handoff.md', plan_path,
         'CaMRA/navigation/docs/nav043_coupled_response_preparation_20261120.md',
         'CaMRA/navigation/config/nav043_coupled_response_proposal_v4.json',
         'CaMRA/navigation/config/nav043_coupled_response_proposal_v4.json.sha256',
         'CaMRA/navigation/config/nav043_coupled_response_readiness_v4.json',
         'CaMRA/navigation/config/nav043_coupled_response_readiness_v4.json.sha256',
         'CaMRA/navigation/scripts/nav043_coupled_response.py',
         'CaMRA/navigation/scripts/run_nav043_coupled_response.py',
         'CaMRA/navigation/scripts/launch_nav043_coupled_response_once.py',
         'CaMRA/navigation/scripts/audit_nav043_coupled_response.py',
         'CaMRA/navigation/tests/test_coupled_response_v4.py']
subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
subprocess.run(['git', 'add', '-f', '--', str(EV.relative_to(ROOT))], cwd=ROOT, check=True)
staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
assert set(staged) == set(paths) | {str(p.relative_to(ROOT)) for p in EV.rglob('*') if p.is_file()}
assert not any(p.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/')) for p in staged)
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
print('LOCAL_COMMIT_FILES', len(staged), flush=True)
subprocess.run(['git', 'commit', '-m', 'NAV-043: prepare isolated coupled response v4 and retain pending fit scope'], cwd=ROOT, check=True)
print('COMMIT', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), flush=True)
print('STATUS', subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True), flush=True)
print('NO_PUSH; NO_REAL_FIT; NO_NEW_APPROVAL', flush=True)
