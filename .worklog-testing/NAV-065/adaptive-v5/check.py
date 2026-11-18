"""Main-agent basic verification; no formal sampling or optimization."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
OUT = ROOT / '.worklog-testing/NAV-065/adaptive-v5'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
inputs = [p for folder in ('src', 'scripts', 'config', 'tests') for p in (NAV / folder).rglob('*')
          if p.is_file() and '__pycache__' not in p.parts]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = {str(p): sha(p) for p in inputs}
results = []
for name, command in [
    ('resources', ['bash', '-c', 'hostname\nnvidia-smi --query-gpu=name,uuid,memory.free --format=csv,noheader\n' + sys.executable + ' --version']),
    ('tests-attempt1', [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_nav065_adaptive_refit.py', 'tests/test_nav065_guarded_refit.py',
        'tests/test_nav065_boundary_refit.py', 'tests/test_nav065_saved_refit.py']),
    ('diagnosis-attempt1', [sys.executable, '-u', 'scripts/refit_nav065_adaptive.py',
        '--diagnose', 'reports/nav065_train_coverage_v5.json']),
    ('plan-validation-attempt1', [sys.executable,
        '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
        str(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'), '--mode', 'update']),
]:
    with (OUT / (name + '.log')).open('x') as log:
        result = subprocess.run(command, cwd=NAV, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append({'name': name, 'command': command, 'cwd': str(NAV), 'returncode': result.returncode})
    print(name, result.returncode, flush=True)
assert source == {str(p): sha(p) for p in inputs}, 'source changed during verification'
report = {'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
          'timestamp': datetime.now().astimezone().isoformat(), 'source_sha256': source,
          'results': results, 'formal_optimizer_steps': 0, 'new_rollouts': 0,
          'development_scoring_rounds': 0}
with (OUT / 'check-attempt1.json').open('x') as stream:
    json.dump(report, stream, indent=2)
