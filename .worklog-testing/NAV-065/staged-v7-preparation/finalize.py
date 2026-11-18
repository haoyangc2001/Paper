"""Finalize reviewed preparation in local Git; never push or start training."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from worklog import ROOT, PLAN, VALIDATOR

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
read = lambda path: json.loads(path.read_bytes())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
review = read(NAV / 'reports/nav065_staged_refit_v7_preparation_review.json')
assert review['preparation_pass'] and not review['scientific_pass']
for name, digest in review['evidence'].items():
    assert sha(Path(name)) == digest
for key, name in (('basic_log_sha256', 'basic-attempt1.log'), ('diagnosis_log_sha256', 'diagnose-attempt1.log'), ('resources_sha256', 'resources.json')):
    assert sha(HERE / name) == review[key]
assert '"preparation_pass": true' in (HERE / 'review-attempt2.log').read_text()
assert not (Path('/pub/data/caohy/work/camra_navigation') / 'nav065_staged_refit_v7').exists()
current = read(PLAN)
baseline = json.loads(subprocess.check_output(['git', 'show', 'a8dd939:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
for old, now in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert now['work_records'][:len(old['work_records'])] == old['work_records']
        assert len(now['work_records']) == len(old['work_records']) + 2
        assert {k:v for k,v in old.items() if k != 'work_records'} == {k:v for k,v in now.items() if k != 'work_records'}
    else:
        assert old == now
assert {k:v for k,v in baseline.items() if k != 'tasks'} == {k:v for k,v in current.items() if k != 'tasks'}
assert not subprocess.check_output(['git', 'diff', '--name-only', 'a8dd939', '--',
    'manuscripts', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT)
subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
failure = {'kind': 'preparation_metadata_patch_failure', 'scientific_execution': False,
    'reason': 'First resource-note sanitization diff lacked the original last-line newline; fixed by constructing newline-terminated diff lines.',
    'raw_log_retained_untracked': str(HERE / 'review-attempt1.log'),
    'raw_log_sha256': sha(HERE / 'review-attempt1.log'),
    'successful_retry_log': str(HERE / 'review-attempt2.log'),
    'raw_unrelated_host_command_lines_excluded_from_git': True}
failure_note = HERE / 'review-attempt1-summary.json'
if failure_note.exists():
    assert read(failure_note) == failure
else:
    with failure_note.open('x') as stream:
        json.dump(failure, stream, indent=2)
        stream.write('\n')
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', str(PLAN.relative_to(ROOT))]
files += ['CaMRA/navigation/' + name for name in (
    'scripts/refit_nav065_staged.py', 'tests/test_nav065_staged_refit.py',
    'config/nav065_staged_refit_proposal_v7.json', 'reports/nav065_staged_refit_preflight_v7.json',
    'reports/nav065_staged_refit_v7_preparation_review.json', 'docs/nav065_staged_refit_v7_20261118.md')]
files += [str((HERE / name).relative_to(ROOT)) for name in (
    'worklog.py', 'preflight.py', 'review.py', 'finalize.py', 'opening.log',
    'basic-attempt1.log', 'diagnose-attempt1.log', 'resources.json', 'review-attempt2.log', 'review-attempt1-summary.json')]
subprocess.run(['git', 'add', '-f', '--', *files], cwd=ROOT, check=True)
staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
assert set(staged) == set(files)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'diff', '--cached', '--stat'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Prepare NAV-065 staged critic refit v7'], cwd=ROOT, check=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT)
print(json.dumps({'commit': commit, 'tracked_tree_clean': True, 'formal_updates': 0,
    'proposal_approved': False, 'nav065_status': False, 'nav066_started': False,
    'preparation_pass': True, 'scientific_pass': False}))
