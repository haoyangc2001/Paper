"""Verify and locally commit only the prepared NAV-065 v8 change set."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
NAV = ROOT / 'CaMRA/navigation'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert git('rev-parse', 'HEAD') == '78919cc0b4b0431d455c6990c764f89dac137c16'
assert not git('diff', '--cached', '--name-only'), 'preserve preexisting staged changes'
assert not (HERE / 'commit.json').exists(), 'inspect existing receipt; never repeat'
baseline = json.loads(git('show', 'HEAD:doc/plan/20261109_camra_navigation_implementation.json'))
current = json.loads(PLAN.read_bytes())
assert {k: v for k, v in current.items() if k != 'tasks'} == {k: v for k, v in baseline.items() if k != 'tasks'}
for old, new in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert new['work_records'][:len(old['work_records'])] == old['work_records']
        assert len(new['work_records']) == len(old['work_records']) + 2
        assert {k: v for k, v in new.items() if k != 'work_records'} == {k: v for k, v in old.items() if k != 'work_records'}
    else:
        assert new == old
states = {t['id']: t['status'] for t in current['tasks']}
assert sum(states.values()) == 52
assert next(t['id'] for t in current['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])) == 'NAV-065'
subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
                str(PLAN), '--mode', 'update'], cwd=ROOT, check=True)
report = json.loads((NAV / 'reports/nav065_pilot_refit_v8_preparation_review.json').read_bytes())
assert report['preparation_pass'] and not report['scientific_pass'] and report['basic_tests_passed'] == 57
assert report['formal_optimizer_steps'] == report['new_rollouts'] == report['development_scoring_rounds'] == 0
assert report['source_gate_exact_match'] and report['original_g4'] == 'FAIL' and not report['proposal_approved']
for name, digest in report['evidence'].items():
    assert sha(Path(name)) == digest, name
for name, key in [('basic-attempt1.log', 'basic_log_sha256'), ('preflight-attempt1.log', 'preflight_log_sha256'), ('resources.json', 'resources_sha256')]:
    assert sha(HERE / name) == report[key]
v2 = json.loads((NAV / 'reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
assert len(v2) == 33
for name, digest in v2.items():
    assert sha(Path(name)) == digest, name
assert not git('diff', '--name-only', '--', 'manuscripts', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT / 'HiRA', text=True).strip()
assert not (NAV / 'config/nav065_pilot_refit_prereg_v8.json').exists()
assert not Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_v8').exists()

files = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
         'doc/plan/20261109_camra_navigation_implementation.json']
files += ['CaMRA/navigation/' + name for name in (
    'scripts/refit_nav065_pilot.py', 'tests/test_nav065_pilot_refit.py',
    'config/nav065_pilot_refit_proposal_v8.json', 'docs/nav065_pilot_refit_v8_20261118.md',
    'reports/nav065_pilot_refit_preflight_v8.json', 'reports/nav065_pilot_refit_v8_preparation_review.json')]
files += [str((HERE / name).relative_to(ROOT)) for name in (
    'preflight.py', 'review.py', 'resources.json', 'commit.py')]
logs = [str((HERE / name).relative_to(ROOT)) for name in (
    'basic-attempt1.log', 'preflight-attempt1.log', 'review-attempt1.log')]
assert all((ROOT / name).is_file() for name in files + logs)
subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
subprocess.run(['git', 'add', '-f', '--', *logs], cwd=ROOT, check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files + logs)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Prepare NAV-065 single-candidate critic refit v8'], cwd=ROOT, check=True)
commit = git('rev-parse', 'HEAD')
assert set(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').splitlines()) == set(files + logs)
assert not git('diff', '--cached', '--name-only') and not git('diff', '--name-only')
receipt = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'cwd': str(ROOT), 'python': sys.executable,
    'commit': commit, 'files': sorted(files + logs), 'push': False,
    'basic_checks': 57, 'train_episodes_diagnosed': 972, 'formal_optimizer_steps': 0,
    'completed_tasks': 52, 'NAV-065': False, 'NAV-066_started': False,
    'new_pilot_optimizer_quota_awaiting_confirmation': 400}
with (HERE / 'commit.json').open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
