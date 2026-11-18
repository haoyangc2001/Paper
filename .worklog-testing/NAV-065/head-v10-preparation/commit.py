"""Commit only this authorized preparation; no push or candidate fitting."""
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert json.loads((HERE / 'final.json').read_bytes())['final_preparation_pass']
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert git('rev-parse', 'HEAD') == 'bdbc248e261778a68c8531b8a302c46d9ef1c61b'
files = [
    'AGENTS.md', 'CaMRA/navigation/README.md',
    'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/scripts/probe_nav065_head.py',
    'CaMRA/navigation/tests/test_nav065_head_probe.py',
    'CaMRA/navigation/config/nav065_head_probe_proposal_v10.json',
    'CaMRA/navigation/docs/nav065_head_probe_v10_20261118.md',
    'CaMRA/navigation/reports/nav065_head_preflight_v10.json',
    'CaMRA/navigation/reports/nav065_head_probe_v10_preparation_review.json'
]
files += [str(HERE.relative_to(ROOT) / name) for name in (
    'context.py', 'check.py', 'close.py', 'verify-final.py', 'commit.py',
    'context.log', 'validation-before.log', 'basic-attempt1.log', 'basic-attempt2.log',
    'preflight-attempt1.log', 'check-attempt1.log', 'close-attempt1.log',
    'validation-after.log', 'final-attempt1.log', 'final.json', 'commit-attempt1.log')]
for name in files:
    assert (ROOT / name).is_file(), name
    assert not name.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/'))
    assert not name.endswith(('.tex', '.bib'))
assert set(git('diff', '--cached', '--name-only').splitlines()) <= set(files)
git('add', '-f', '--', *files)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
# Pytest's original failure log contains trailing spaces. Retain the raw log
# and check every other staged file; never rewrite evidence to pass lint.
raw_log = str(HERE.relative_to(ROOT) / 'basic-attempt1.log')
whitespace = subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, text=True, capture_output=True)
assert whitespace.returncode == 2 and raw_log in whitespace.stdout
git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check', '--', '.', ':(exclude)' + raw_log)
print(git('diff', '--cached', '--stat'), flush=True)
print(git('commit', '-m', 'Prepare NAV-065 v10 head expressibility probe'), flush=True)
assert not git('diff', '--cached', '--name-only') and not git('diff', '--name-only')
result = {'commit': git('rev-parse', 'HEAD'), 'files': files, 'pushed': False,
    'raw_test_failure_log_preserved_with_whitespace': True, 'all_other_staged_files_diff_check_pass': True,
    'formal_candidate_solves': 0, 'formal_optimizer_updates': 0, 'NAV065': False}
with (HERE / 'commit.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
