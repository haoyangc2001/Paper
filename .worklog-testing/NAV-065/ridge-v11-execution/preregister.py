"""Validate and locally commit the approved protocol before its single run."""
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
sys.path.insert(0, str(NAV / 'scripts'))
import probe_nav065_ridge as G
protocol, pin = G.approved_registration(NAV / 'config/nav065_ridge_probe_prereg_v11.json')
assert not (G.R.WORK / protocol['output_name']).exists()
baseline = json.loads((HERE / 'baseline.json').read_bytes())
plan = json.loads((ROOT / 'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
assert {k: v for k, v in plan.items() if k != 'tasks'} == {k: v for k, v in baseline['plan'].items() if k != 'tasks'}
for old, new in zip(baseline['plan']['tasks'], plan['tasks']):
    if old['id'] == 'NAV-065':
        assert {k: v for k, v in old.items() if k != 'work_records'} == {k: v for k, v in new.items() if k != 'work_records'}
        assert new['work_records'][:-1] == old['work_records']
    else:
        assert old == new
with (HERE / 'validation-preregister.log').open('x') as f:
    subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'), '--mode', 'update'], stdout=f, stderr=subprocess.STDOUT, check=True)
git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
assert git('rev-parse', 'HEAD') == baseline['git']
files = ['CaMRA/navigation/config/nav065_ridge_probe_prereg_v11.json', 'CaMRA/navigation/config/nav065_ridge_probe_prereg_v11.json.sha256',
    'doc/plan/20261109_camra_navigation_implementation.json']
files += [str(HERE.relative_to(ROOT) / name) for name in ('context.py', 'launch.py', 'preregister.py', 'context.log', 'validation-before.log', 'validation-preregister.log')]
assert not git('diff', '--cached', '--name-only')
git('add', '-f', '--', *files)
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
git('-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check')
print(git('commit', '-m', 'Register approved NAV-065 v11 two-head ridge probe'), flush=True)
assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
with (HERE / 'preregister.json').open('x') as f:
    json.dump({'git': git('rev-parse', 'HEAD'), 'pin': pin, 'files': files, 'pushed': False, 'formal_head_solves_before_launch': 0}, f, ensure_ascii=False, indent=2)
print('Preregistration committed; one authorized launch may proceed.', flush=True)
