"""One approved v5 attempt, launched on tiger11 inside tmux; no automatic retries."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_adaptive_refit_launch_20261118_v5')
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'

def write(name, payload):
    with (OUT / name).open('x') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
OUT.mkdir(exist_ok=False)
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copy2(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json', OUT / 'before-plan.json')
write('START.json', {'time': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'pid': os.getpid(), 'python': sys.executable, 'cwd': str(NAV),
    'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'authorization': 'Latest user confirms reviewed v5: 432 train/194400 steps; 300 trials each/1800 total; one saved development scoring only after all train gates.',
    'direct_tiger11_exception': 'Previously approved; no SSH requirement repeated.',
    'test_identity': 'main agent, legacy workflow; no dedicated subagents'})
commands = [
    ('resources', ['bash', '-c', 'hostname\nnvidia-smi --query-gpu=name,uuid,memory.free --format=csv,noheader\n' + PYTHON + ' --version']),
    ('validate-before', [PYTHON, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
        str(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'), '--mode', 'update']),
    ('tests', [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_nav065_adaptive_refit.py', 'tests/test_nav065_guarded_refit.py',
        'tests/test_nav065_boundary_refit.py', 'tests/test_nav065_saved_refit.py']),
    ('training', [PYTHON, '-u', 'scripts/refit_nav065_adaptive.py', '--registration',
        'config/nav065_adaptive_refit_prereg_v5.json']),
]
try:
    sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
    import refit_nav065_adaptive as A
    protocol, pin = A.approved_registration(NAV / 'config/nav065_adaptive_refit_prereg_v5.json')
    assert not (A.R.WORK / protocol['output_name']).exists(), 'Never restart a prior attempt'
    hashes = {}
    for directory in ('src', 'scripts', 'config', 'tests'):
        for path in (NAV / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                relative = path.relative_to(NAV)
                dest = OUT / 'source_snapshot' / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                hashes[str(relative)] = sha(path)
    write('source_before.json', hashes)
    write('registration.json', {'path': str(NAV / 'config/nav065_adaptive_refit_prereg_v5.json'), 'pin': pin,
        'resource_limit': protocol['resource_limit']})
    for name, command in commands:
        write(name + '-command.json', {'command': command, 'cwd': str(NAV), 'env':
            {key: env[key] for key in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
        with (OUT / (name + '.log')).open('x') as stream:
            result = subprocess.run(command, cwd=NAV, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write(name + '-exit.json', {'returncode': result.returncode, 'time': datetime.now().astimezone().isoformat()})
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(name + ' failed; preserve attempt, do not retry automatically')
    assert all(sha(NAV / name) == digest for name, digest in hashes.items()), 'Source changed during run'
    write('EXIT.json', {'returncode': 0, 'time': datetime.now().astimezone().isoformat(),
        'source_unchanged': True, 'scientific_result': 'Read train_gate/G4; exit code alone is not acceptance'})
except BaseException as exc:
    write('ERROR.json', {'error': repr(exc), 'time': datetime.now().astimezone().isoformat()})
    raise
