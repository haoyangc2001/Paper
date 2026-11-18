"""Commit only authorized NAV-065 v11 preparation, preserving unrelated artifacts."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
report = json.loads((HERE / 'final.json').read_bytes())
assert report['preparation_pass'] and not report['proposal_approved']
assert report['formal_head_solves'] == report['formal_optimizer_steps'] == 0
assert hashlib.sha256((ROOT / 'doc/plan/20261109_camra_navigation_implementation.json').read_bytes()).hexdigest() == report['plan_sha256']

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

assert git('rev-parse', 'HEAD') == report['base_git'] == '912e31c76a42fdd3076f57890a52939a7371d451'
files = [
    'AGENTS.md', 'CaMRA/navigation/README.md',
    'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/scripts/probe_nav065_ridge.py',
    'CaMRA/navigation/tests/test_nav065_ridge_probe.py',
    'CaMRA/navigation/config/nav065_ridge_probe_proposal_v11.json',
    'CaMRA/navigation/docs/nav065_ridge_probe_v11_20261118.md',
    'CaMRA/navigation/reports/nav065_ridge_preflight_v11.json',
    'CaMRA/navigation/reports/nav065_ridge_probe_v11_preparation_review.json'
]
files += [str(HERE.relative_to(ROOT) / name) for name in (
    'context.py', 'open.py', 'preflight.py', 'check.py', 'close.py', 'final.py', 'commit.py',
    'context.log', 'open.log', 'validation-before.log', 'basic-attempt1.log',
    'preflight-attempt1.log', 'check-attempt1.log', 'close.log', 'validation-after.log',
    'final.log', 'final.json')]
for name in files:
    assert (ROOT / name).is_file(), name
    assert not name.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/'))
    assert not name.endswith(('.tex', '.bib', '.pdf'))
assert set(git('diff', '--cached', '--name-only').splitlines()) <= set(files)
git('add', '-f', '--', *files)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
lint = subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, text=True, capture_output=True)
raw_log = str(HERE.relative_to(ROOT) / 'context.log')
raw_log_whitespace = lint.returncode != 0
if raw_log_whitespace:
    assert lint.returncode == 2 and raw_log in lint.stdout, lint.stdout + lint.stderr
    # Keep historical failure output byte-for-byte; all other staged files must pass.
    git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', '.', ':(exclude)' + raw_log)
print(git('diff', '--cached', '--stat'), flush=True)
print(git('commit', '-m', 'Prepare NAV-065 v11 regularized full-train head probe'), flush=True)
assert not git('diff', '--cached', '--name-only') and not git('diff', '--name-only')
result = {'commit': git('rev-parse', 'HEAD'), 'files': files, 'pushed': False,
    'raw_context_failure_preserved': True, 'raw_log_whitespace_exception': raw_log_whitespace,
    'all_other_staged_files_diff_check_pass': True, 'formal_head_solves': 0,
    'formal_optimizer_updates': 0, 'NAV065': False, 'proposal_approved': False}
with (HERE / 'commit.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
