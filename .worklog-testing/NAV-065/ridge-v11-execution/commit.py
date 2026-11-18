"""Commit only the executed v11 outcome; no push, retry or extra fitting."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
review = json.loads((HERE / 'final.json').read_bytes())
assert review['delivery_integrity_pass'] and review['formal_candidate_head_solves'] == 2
assert review['audit_head_solves'] == 0 and not review['scientific_pass']
for name, digest in review['evidence_hashes'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert git('rev-parse', 'HEAD') == json.loads((HERE / 'preregister.json').read_bytes())['git']
files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/docs/nav065_ridge_probe_v11_review_20261118.md',
    'CaMRA/navigation/reports/nav065_ridge_probe_v11_review.json']
files += [str(HERE.relative_to(ROOT) / name) for name in (
    'preregister.log', 'preregister.json', 'execution.json', 'validation-approved.log', 'launch-attempt1.log',
    'audit.py', 'audit-attempt1.log', 'close.py', 'close-attempt1.log', 'final.py', 'final-attempt1.log',
    'validation-after.log', 'final.json', 'commit.py')]
for name in files:
    assert (ROOT / name).is_file(), name
    assert not name.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'CaMRA/locomotion/', 'CaMRA/calibration/'))
    assert not name.endswith(('.tex', '.bib', '.pdf'))
assert not git('diff', '--cached', '--name-only')
git('add', '-f', '--', *files)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check')
print(git('diff', '--cached', '--stat'), flush=True)
print(git('commit', '-m', 'Record NAV-065 v11 full-train ridge probe outcome'), flush=True)
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
result = {'commit': git('rev-parse', 'HEAD'), 'files': files, 'pushed': False,
    'formal_head_solves': 2, 'audit_head_solves': 0, 'optimizer_updates': 0, 'new_rollouts': 0,
    'development_scoring_rounds': 0, 'scientific_pass': False, 'NAV065': False, 'NAV066_started': False}
with (HERE / 'commit.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False), flush=True)
