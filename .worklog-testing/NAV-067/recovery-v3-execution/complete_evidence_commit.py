"""Commit saved offline verification evidence after the coordinator watcher commits."""
from pathlib import Path
import hashlib
import json
import os
import socket
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE')
assert not (HERE / 'REVIEW_HELD.json').exists()
watcher = json.loads((HERE / 'REVIEW_COMMITTED.json').read_bytes())
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert head == watcher['git_head']
assert not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
report = json.loads((ROOT / 'CaMRA/navigation/reports/nav067_recovery_v3_execution_review.json').read_bytes())
assert report['status'] == 'PILOT_TRACE_PASS_NAV067_INCOMPLETE'
assert report['saved_episode_results'] == report['trace_audited_episodes'] == 6
assert not report['main_agent_saved_episode_review_failures']
assert not report['main_agent_saved_runtime_review_failures']
for name, field in (
    ('review_saved_episode.py', 'main_agent_saved_episode_review_source_sha256'),
    ('review_runtime.py', 'main_agent_saved_runtime_review_source_sha256'),
):
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == report[field]
names = ['review_saved_episode.py', 'review_runtime.py', 'L1_observe.py',
    'L1-observational-diagnostics.json', 'L1-observational-diagnostics.log',
    'REVIEW_COMMITTED.json', Path(__file__).name]
for level in ('L1_empty', 'L2_single_detour', 'L3_multi_path'):
    names.extend(['runtime-review-' + level + suffix for suffix in ('.json', '.log')])
    for method in ('uncalibrated', 'full'):
        names.extend(['main-agent-' + level + '-' + method + suffix for suffix in ('.json', '.log')])
paths = [str((HERE / name).relative_to(ROOT)) for name in names]
assert all((ROOT / p).is_file() for p in paths)
for row in report['main_agent_saved_episode_reviews']:
    name = 'main-agent-' + row['level'] + '-' + row['method'] + '.json'
    assert json.loads((HERE / name).read_bytes()) == row
for row in report['main_agent_saved_runtime_reviews']:
    assert json.loads((HERE / ('runtime-review-' + row['level'] + '.json')).read_bytes()) == row
subprocess.run(['git', 'add', '-f', '--', *paths], cwd=ROOT, check=True)
staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines())
assert staged and staged.issubset(set(paths)), staged
subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'chore(nav): retain NAV-067 offline trace and runtime verification evidence'], cwd=ROOT, check=True)
print(json.dumps({'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'new_simulation_starts': 0, 'training_updates': 0, 'W_queries': 0, 'remote_push': False,
    'NAV_067_status': False, 'committed_paths': sorted(staged)}, ensure_ascii=False, indent=2), flush=True)
