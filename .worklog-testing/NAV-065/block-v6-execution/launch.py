"""Run the explicitly approved v6 once on tiger11/tmux, with no retries."""
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import textwrap

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = ROOT / '.worklog-testing/NAV-065/block-v6-execution'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_block_refit_launch_20261118_v6')
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
VALIDATOR = '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py'
REG = NAV / 'config/nav065_block_refit_prereg_v6.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
os.chdir(NAV)
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src')]
import refit_nav065_blocks as V

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
env = {**os.environ, 'PYTHONPATH': 'src:scripts', 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}

def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def apply(patch):
    subprocess.run(['apply_patch'], cwd=ROOT, text=True, input=patch, check=True)

def add(path, content):
    assert not path.exists()
    apply('*** Begin Patch\n*** Add File: ' + str(path) + '\n' +
          ''.join('+' + line + '\n' for line in content.splitlines()) + '*** End Patch\n')

def append(record):
    original = PLAN.read_text()
    start = original.index('"work_records": [', original.index('"id": "NAV-065"')) + len('"work_records": ')
    _, length = json.JSONDecoder().raw_decode(original[start:])
    changed = original[:start + length - 1].rstrip() + ',\n' + textwrap.indent(json.dumps(record, ensure_ascii=False, indent=1), '    ') + '\n   ]' + original[start + length:]
    diff = ''.join(list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True)))[2:])
    diff = re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
    apply('*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n' + diff + '*** End Patch\n')

def command(name, args):
    write(LAUNCH / (name + '-command.json'), {'command': args, 'cwd': str(NAV),
          'env': {k: env[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
    with (LAUNCH / (name + '.log')).open('x') as stream:
        result = subprocess.run(args, cwd=NAV, env=env, stdout=stream, stderr=subprocess.STDOUT)
    write(LAUNCH / (name + '-exit.json'), {'returncode': result.returncode, 'time': datetime.now().astimezone().isoformat()})
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed; preserve attempt; no automatic retry')

# Read and validate the complete live plan, never infer its state from a summary.
plan = json.loads(PLAN.read_bytes())
states = {t['id']: t['status'] for t in plan['tasks']}
task = next(t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
assert task['id'] == 'NAV-065' and sum(states.values()) == 52
assert not REG.exists() and not REG.with_suffix('.json.sha256').exists()
proposal = json.loads(V.PROPOSAL.read_bytes())
assert not (V.R.WORK / proposal['output_name']).exists()
LAUNCH.mkdir(exist_ok=False)
shutil.copy2(PLAN, LAUNCH / 'before-plan.json')
write(LAUNCH / 'START.json', {'time': datetime.now().astimezone().isoformat(),
    'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'pid': os.getpid(),
    'python': sys.executable, 'cwd': str(NAV), 'source_git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'authorization': 'Latest confirmation approves reviewed v6: reuse 972 train trajectories/372471 steps; 300 trials per candidate/1800 total; four-step blocks and fixed-source 5% MAE/0.02 Pearson working guards; unchanged final criteria; one old180 development scoring only after all six train pass.',
    'direct_tiger11_exception': 'Previously explicitly approved same-host execution; do not request it again.',
    'reviewer': 'main agent using create-plan-worklog-legacy; no test subagents'})
try:
    command('resources', ['bash', '-c', 'hostname\nnvidia-smi --query-gpu=name,uuid,memory.free --format=csv,noheader\n' + PYTHON + ' --version\ntmux list-windows -a -F "#{session_name}:#{window_name}:#{window_active}"'])
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    with (LAUNCH / 'processes-before.log').open('x') as stream:
        stream.write(processes)
    conflicts = [line for line in processes.splitlines() if re.search(r'python(?:\d+(?:\.\d+)*)?\s+(?:-u\s+)?(?:\S*/)?refit_nav065_\S+\.py\s+--registration', line)]
    assert not conflicts, 'An existing NAV-065 fitting process must be reconciled first'
    v2 = json.loads((NAV / 'reports/nav064_terminal_tail_training_release_v1.json').read_bytes())['v2_verified_files']
    for path, digest in v2.items():
        assert sha(Path(path)) == digest
    command('validate-before', [PYTHON, VALIDATOR, str(PLAN), '--mode', 'update'])
    protocol = {**proposal, 'approved': True, 'status': 'APPROVED',
        'authorization_reference': 'User confirms the reviewed NAV-065 v6 proposal in the latest reply: 确认; new 300 per candidate/1800 total and working block guards explicitly included.',
        'implementation_hashes': {name: sha(NAV / name) for name in V.BOUND_INPUTS}}
    add(REG, json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    add(REG.with_suffix('.json.sha256'), sha(REG) + '\n')
    protocol, pin = V.approved_registration(REG)
    write(LAUNCH / 'registration.json', {'path': str(REG), 'pin': pin, 'resource_limit': protocol['resource_limit'], 'v2_verified_files': v2})
    hashes = {}
    for directory in ('src', 'scripts', 'config', 'tests'):
        for path in (NAV / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                relative = path.relative_to(NAV)
                destination = LAUNCH / 'source_snapshot' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                hashes[str(relative)] = sha(path)
    write(LAUNCH / 'source_before.json', hashes)
    append({'timestamp': datetime.now().astimezone().isoformat(),
        'summary': 'NAV-065/1.3.1 v6用户确认后开工：冻结新试步额度及短段工作保护，准备一次正式离线回修',
        'actions': ['重读根AGENTS、legacy执行规则、完整计划及NAV-065约束/既有记录；首个依赖满足任务仍NAV-065，NAV-064=true。',
            '沿用dec3bf4已验收隔离入口与972条train/372471步；用户最新确认明确批准每候选300新试步/总1800和每4步一段、固定源5%MAE/0.02相关性工作保护，拒绝段步数计入。',
            'tiger11/tmux核验资源、无并发拟合、33项v2身份；通过apply_patch新增独立批准登记和SHA256，不改原proposal=false及旧run。'],
        'files_changed': [str(REG.relative_to(ROOT)), str(REG.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), '.worklog-testing/NAV-065/block-v6-execution/launch.py'],
        'commands': ['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v6-run；cwd=' + str(NAV) + '；Python=' + PYTHON,
            '同环境 <Python> .worklog-testing/NAV-065/block-v6-execution/launch.py（绝对路径）；commands及资源逐项保存在' + str(LAUNCH)],
        'results': ['批准登记pin=' + pin, '本项为NAV-066提供匹配固定actor的可信双critic；本轮仅NAV-065离线回修，无新回合、actor更新或物理运行。',
            '保持共同停止目标/W/beta=0、最终v5 train_gate与原G4、固定v2及校准/地图/时序/单位/分片；不访问budget_selection/final。',
            '63基本检查、源身份、实际试步/段接受回滚、CPU重载及原门禁为验收；六train全过后只一次原180 development重评；科学达标和下游交付尚未完成。'],
        'evidence': ['用户最新确认', 'CaMRA/navigation/docs/nav065_block_refit_v6_20261118.md', 'CaMRA/navigation/reports/nav065_block_refit_v6_preparation_review.json', str(LAUNCH / 'registration.json')],
        'remaining_work': ['保存批准版本后执行一次正式v6；按真实产物复核，受影响门禁未过保持false且不启动NAV-066。'],
        'blockers': []})
    command('tests', [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        'tests/test_nav065_block_refit.py', 'tests/test_nav065_adaptive_refit.py',
        'tests/test_nav065_guarded_refit.py', 'tests/test_nav065_boundary_refit.py', 'tests/test_nav065_saved_refit.py'])
    command('validate-approved', [PYTHON, VALIDATOR, str(PLAN), '--mode', 'update'])
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
    files = [str(REG.relative_to(ROOT)), str(REG.with_suffix('.json.sha256').relative_to(ROOT)),
        str(PLAN.relative_to(ROOT)), str((HERE / 'launch.py').relative_to(ROOT))]
    subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
    assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()) == set(files)
    command('approval-commit', ['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'])
    with (LAUNCH / 'approval-commit-result.log').open('x') as stream:
        subprocess.run(['git', 'commit', '-m', 'Register approved NAV-065 critic block refit v6'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(LAUNCH / 'training_start.json', {'time': datetime.now().astimezone().isoformat(), 'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'registration_pin': pin})
    command('training', [PYTHON, '-u', 'scripts/refit_nav065_blocks.py', '--registration', str(REG.relative_to(NAV))])
    assert all(sha(NAV / name) == digest for name, digest in hashes.items()), 'Source changed during run'
    write(LAUNCH / 'EXIT.json', {'returncode': 0, 'time': datetime.now().astimezone().isoformat(), 'source_unchanged': True,
        'scientific_result': 'Inspect real train_gate/G4, not this process exit code'})
except BaseException as exc:
    write(LAUNCH / 'ERROR.json', {'error': repr(exc), 'time': datetime.now().astimezone().isoformat()})
    raise
