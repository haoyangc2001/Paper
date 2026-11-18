"""Main-agent basic checks and read-only real-data diagnosis."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
env = {**os.environ, 'PYTHONPATH': str(NAV / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
with (HERE / 'basic-attempt1.log').open('x') as f:
    result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
        'tests/test_nav065_local_probe.py'], cwd=NAV, env=env, stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0, 'inspect basic-attempt1.log; no diagnosis on failed basics'
with (HERE / 'preflight-attempt1.log').open('x') as f:
    result = subprocess.run([sys.executable, '-B', '-u', 'scripts/probe_nav065_local.py',
        '--diagnose', 'reports/nav065_local_probe_preflight_v9.json'],
        cwd=NAV, env=env, stdout=f, stderr=subprocess.STDOUT)
assert result.returncode == 0, 'inspect preflight-attempt1.log; do not overwrite'
print('BASIC_AND_READONLY_PREFLIGHT_DONE', flush=True)
