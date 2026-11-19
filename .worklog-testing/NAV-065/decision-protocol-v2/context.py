"""Read-only protocol preparation context; no model execution or new data."""
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
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
p = json.loads(PLAN.read_bytes())
states = {t['id']: t['status'] for t in p['tasks']}
current = next(t for t in p['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
assert len(states) == 67 and sum(states.values()) == 52 and current['id'] == 'NAV-065'
with (HERE / 'validation-before.log').open('x') as f:
    subprocess.run([PYTHON, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT, check=True)
v2 = json.loads((ROOT / 'CaMRA/navigation/reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
for name, digest in v2.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
protected = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', 'doc/plan/20261109_camra_navigation_implementation.json']
protected += [str(f.relative_to(ROOT)) for f in (ROOT / 'CaMRA/navigation/config').glob('*.json')]
protected += [str(f.relative_to(ROOT)) for f in (ROOT / 'CaMRA/navigation/reports').glob('nav065*review.json')]
baseline = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': PYTHON, 'python_version': sys.version, 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'v2_files': v2, 'hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in protected}, 'plan': p, 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader'], text=True)}
with (HERE / 'baseline.json').open('x') as f:
    json.dump(baseline, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in baseline.items() if k not in ('plan', 'hashes', 'v2_files')}, ensure_ascii=False))
print('v2_files_verified=' + str(len(v2)))
print('current_task=' + current['id'])
print(json.dumps({'v7_candidates': json.loads((ROOT / 'CaMRA/navigation/reports/nav065_staged_refit_v7_review.json').read_bytes())['train_candidates']}, ensure_ascii=False))
