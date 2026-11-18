"""Register, commit and execute the single explicitly approved NAV-065 v7 run."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7')
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
REG = NAV / 'config/nav065_staged_refit_prereg_v7.json'
VALIDATOR = '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
sys.path.insert(0, str(ROOT / '.worklog-testing/NAV-065/staged-v7-preparation'))
from worklog import append
import refit_nav065_staged as S

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())
def write(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
def add(path, content):
    patch = '*** Begin Patch\n*** Add File: ' + str(path) + '\n' + ''.join('+' + line + '\n' for line in content.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, cwd=ROOT, check=True)
def command(name, args):
    write(LAUNCH / (name + '.command.json'), {'args': args, 'cwd': str(NAV), 'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable})
    with (LAUNCH / (name + '.log')).open('x') as stream:
        subprocess.run(args, cwd=NAV, stdout=stream, stderr=subprocess.STDOUT, check=True)

LAUNCH.mkdir(exist_ok=False)
try:
    plan = read(PLAN)
    states = {t['id']: t['status'] for t in plan['tasks']}
    current = next(t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
    assert plan['document']['version'] == '1.3.1' and current['id'] == 'NAV-065' and sum(states.values()) == 52
    proposal = read(S.PROPOSAL)
    assert proposal['approved'] is False and not REG.exists()
    assert not (S.R.WORK / proposal['output_name']).exists()
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT)
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
    review = read(NAV / 'reports/nav065_staged_refit_v7_preparation_review.json')
    assert review['preparation_pass'] and review['basic_tests_passed'] == 79
    for name, digest in review['evidence'].items():
        assert sha(Path(name)) == digest
    process_text = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    conflicts = [line.split()[0] for line in process_text.splitlines() if re.search(r'python(?:\d+(?:\.\d+)*)?\s+(?:-u\s+)?(?:\S*/)?refit_nav065_\S+\.py\s+--registration', line)]
    assert not conflicts, 'Existing NAV-065 fitting process must be reconciled'
    v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
    for name, digest in v2.items():
        assert sha(Path(name)) == digest
    write(LAUNCH / 'resources.json', {'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
        'python': sys.executable, 'version': sys.version, 'cwd': str(NAV), 'active_refit_processes': 0,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True, timeout=20),
        'v2_verified_files': v2, 'ssh_exception': 'Existing user approval permits direct same-host tiger11 tmux execution.'})
    command('validate-before', [PYTHON, VALIDATOR, str(PLAN), '--mode', 'update'])
    protocol = {**proposal, 'approved': True, 'status': 'APPROVED',
        'authorization_reference': 'Latest user confirmation of the reviewed v7 plan: 确认. One new 150 numeric + 150 boundary step run per candidate, six candidates <=1800; process monotone guards removed, final gates unchanged.',
        'implementation_hashes': {name: sha(NAV / name) for name in S.BOUND_INPUTS}}
    add(REG, json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    add(REG.with_suffix('.json.sha256'), sha(REG) + '\n')
    protocol, pin = S.approved_registration(REG)
    write(LAUNCH / 'registration.json', {'path': str(REG), 'pin': pin, 'resource_limit': protocol['resource_limit']})
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
        'summary': 'NAV-065/1.3.1 v7用户确认后开工：冻结两阶段过程规则及新额度，执行一次离线回修',
        'actions': ['重读根AGENTS、legacy技能/执行规则、模块说明和完整JSON，核对NAV-064=true、NAV-065为首个依赖满足任务及v7准备证据。',
            '用户确认已审阅v7方案：每候选最多数值150+边界150新试步、总1800；取消离线过程单调保护，原最终标准保持，不复用v6余量。',
            'tiger11/tmux现场核验绝对解释器、GPU、无并发拟合、33项v2身份及79基本测试对应源码hash；新增独立批准登记/pin和来源快照，不改原proposal=false。'],
        'files_changed': [str(REG.relative_to(ROOT)), str(REG.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), str((HERE / 'launch.py').relative_to(ROOT))],
        'commands': ['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v7-run；cwd=' + str(NAV) + '；Python=' + PYTHON,
            '<Python> -u .worklog-testing/NAV-065/staged-v7-execution/launch.py（绝对路径）；完整非敏感命令/资源/源码快照保存于' + str(LAUNCH)],
        'results': ['批准登记pin=' + pin, '本项为NAV-066提供匹配固定actor的可信双critic；仅复用972条train/372471步，无新回合、actor更新、物理执行或预算选择/final访问。',
            '共同停止cost/reach目标/W/beta=0、原v5最终train/G4、v2及匹配校准、地图/时序/单位/分片保持；允许本次已批准的分阶段优化过程变化。',
            '主agent按legacy验收，无测试子agent。全部train通过、六模型保存及CPU重载后仅一次原180 development重评，原G4决定交付；未达标保持false。'],
        'evidence': ['用户最新明确确认', 'c8f41d1383bbd452dffe7332d0007b933aa2e0f4', 'CaMRA/navigation/reports/nav065_staged_refit_v7_preparation_review.json', str(LAUNCH / 'registration.json')],
        'remaining_work': ['本地提交批准版本，执行一次新run并按真实产物验收；未全部通过不启动NAV-066。'], 'blockers': []})
    command('validate-approved', [PYTHON, VALIDATOR, str(PLAN), '--mode', 'update'])
    files = [str(REG.relative_to(ROOT)), str(REG.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), str((HERE / 'launch.py').relative_to(ROOT))]
    subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
    assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()) == set(files)
    command('approval-diff-check', ['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'])
    with (LAUNCH / 'approval-commit.log').open('x') as stream:
        subprocess.run(['git', 'commit', '-m', 'Register approved NAV-065 staged critic refit v7'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(LAUNCH / 'training_start.json', {'time': datetime.now().astimezone().isoformat(), 'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'registration_pin': pin})
    command('training', [PYTHON, '-u', 'scripts/refit_nav065_staged.py', '--registration', str(REG.relative_to(NAV))])
    assert all(sha(NAV / name) == digest for name, digest in hashes.items()), 'Source changed during run'
    write(LAUNCH / 'EXIT.json', {'returncode': 0, 'time': datetime.now().astimezone().isoformat(),
        'source_unchanged': True, 'scientific_result': 'Inspect real train_gate/G4, not process exit.'})
except BaseException as exc:
    write(LAUNCH / 'ERROR.json', {'error': repr(exc), 'time': datetime.now().astimezone().isoformat()})
    raise
