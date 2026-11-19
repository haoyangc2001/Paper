"""Single approved v3 launch; never retries an existing attempt."""

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
PYTHON = Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python')
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_boundary_refit_launch_20261118_v3')
REG = NAV / 'config/nav065_boundary_refit_prereg_v3.json'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'


def stamp():
    return datetime.now().astimezone().isoformat()


def write(name, data):
    with (LAUNCH / name).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    assert sys.executable == str(PYTHON)
    assert not LAUNCH.exists()
    assert not REG.exists()
    assert not (Path('/pub/data/caohy/work/camra_navigation') / 'nav065_boundary_refit_v3').exists()
    LAUNCH.mkdir()
    shutil.copyfile(PLAN, LAUNCH / 'before-plan.json')
    before = json.loads(PLAN.read_text())
    statuses = {t['id']: t['status'] for t in before['tasks']}
    current = next(t for t in before['tasks'] if not t['status'] and
                   all(statuses[d] for d in t['depends_on']))
    assert current['id'] == 'NAV-065'
    assert len(current['work_records']) == 21
    source = json.loads((ROOT / '.worklog-testing/NAV-065/boundary-v3-basic/report.json').read_text())
    for name, sha in source['source_hashes'].items():
        assert digest(ROOT / name) == sha, name
    sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
    import refit_nav065_boundary as B
    p = B.R.load(B.PROPOSAL)
    assert not p['approved']
    p.update(approved=True, status='APPROVED', authorization_reference=(
        'User confirmation 2026-11-18 of the reviewed v3 proposal: only full seed1/seed2 each at most '
        '300 new critic updates, total at most 600; four passing sources unchanged; zero new rollouts; '
        'all six train recognition gates before at most one replay of 180 development episodes; '
        'legacy main-agent basic verification without test subagents.'))
    p['implementation_hashes'] = {name: digest(NAV / name) for name in B.BOUND_INPUTS}
    registration_text = json.dumps(p, ensure_ascii=False, indent=2) + '\n'
    pin = hashlib.sha256(registration_text.encode()).hexdigest()
    patch = ('*** Begin Patch\n*** Add File: '+str(REG)+'\n'+
             ''.join('+'+line+'\n' for line in registration_text.splitlines())+
             '*** Add File: '+str(REG.with_suffix('.json.sha256'))+'\n+'+pin+'\n*** End Patch\n')
    with (LAUNCH / 'registration.patch').open('x') as stream:
        stream.write(patch)
    write('resources.json', {
        'timestamp': stamp(), 'host': socket.getfqdn(), 'cwd': str(NAV),
        'python': sys.executable, 'python_version': sys.version, 'tmux': os.environ['TMUX'],
        'window': subprocess.check_output(['tmux','display-message','-p','-t',os.environ['TMUX_PANE'],
                                          '#{session_name}:#{window_name} pane=#{pane_id}'], text=True).strip(),
        'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total',
                                        '--format=csv'], text=True),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status': subprocess.check_output(['git','status','--short','-uno'],cwd=ROOT,text=True),
        'registration_pin': pin,
    })
    paths = set(p['implementation_hashes']) | {
        'scripts/refit_nav065_deployment.py', 'scripts/evaluate_nav065_deployment.py',
        'src/camra_navigation/algorithm/deployment_oracle.py',
        'config/nav065_saved_refit_prereg_v2.json', 'config/nav065_saved_refit_prereg_v2.json.sha256',
        'config/nav065_refit_prereg_v1.json', 'config/nav065_refit_prereg_v1.json.sha256',
        'config/nav065_deployment_prereg_v1.json', 'config/nav065_deployment_prereg_v1.json.sha256',
    }
    frozen = LAUNCH / 'source_snapshot'
    for name in paths:
        destination = frozen / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(NAV / name, destination)
    write('source_before.json', {name: digest(NAV / name) for name in sorted(paths)})
    print('READY registration pin', pin, flush=True)


def launch():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    assert not (LAUNCH / 'START.json').exists()
    sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
    import refit_nav065_boundary as B
    protocol, pin = B.approved_registration(REG)
    assert pin == json.loads((LAUNCH / 'resources.json').read_text())['registration_pin']
    for name, sha in json.loads((LAUNCH / 'source_before.json').read_text()).items():
        assert digest(NAV / name) == sha, name
    out = B.R.WORK / protocol['output_name']
    assert not out.exists()
    # Check actual Python command lines, excluding this wrapper and shell commands.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if args and 'python' in Path(args[0]).name and '--registration' in args:
            assert not any(Path(a).name == 'refit_nav065_boundary.py' for a in args), 'v3 already running'
    # New quota and settings are now frozen; old unapproved proposal stays unchanged.
    command = [str(PYTHON), '-u', 'scripts/refit_nav065_boundary.py', '--registration', str(REG)]
    env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    with (LAUNCH / 'training.log').open('x') as log:
        child = subprocess.Popen(command, cwd=NAV, env=env, stdout=log, stderr=subprocess.STDOUT)
        write('START.json', {'timestamp': stamp(), 'pid': child.pid, 'wrapper_pid': os.getpid(),
                             'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'cwd': str(NAV),
                             'command': command, 'registration_pin': pin, 'output': str(out),
                             'log': str(LAUNCH / 'training.log')})
        print('STARTED', child.pid, 'pin', pin, flush=True)
        code = child.wait()
    status = ('INFRASTRUCTURE_OR_INTEGRITY_ERROR' if code else
              'HELD_TRAIN_GATE' if (out / 'HELD_TRAIN_GATE.json').exists() else
              B.R.load(out / 'gate_G4_alignment_v2.json')['verdict'] if
              (out / 'gate_G4_alignment_v2.json').exists() else 'INCOMPLETE_NO_VERDICT')
    write('EXIT.json', {'timestamp': stamp(), 'exit_code': code, 'status': status,
                        'registration_pin': pin})
    print('FINISHED', code, status, flush=True)


if __name__ == '__main__':
    {'init': init, 'launch': launch}[sys.argv[1]]()
