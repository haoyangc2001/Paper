"""Register and execute the user-approved, one-candidate NAV-065 v8 pilot once."""
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
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8')
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
REG = NAV / 'config/nav065_pilot_refit_prereg_v8.json'
VALIDATOR = '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == PYTHON
sys.path[:0] = [str(NAV / 'scripts'), str(NAV / 'src'), str(ROOT / '.worklog-testing/NAV-065/staged-v7-preparation')]
from worklog import append
import refit_nav065_pilot as P

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())
ENV = {**os.environ, 'PYTHONPATH': str(NAV / 'src') + os.pathsep + str(NAV / 'scripts'),
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}

def write(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def add(path, content):
    assert not path.exists()
    patch = '*** Begin Patch\n*** Add File: ' + str(path) + '\n' + ''.join('+' + line + '\n' for line in content.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, cwd=ROOT, check=True)

def command(name, args, cwd=NAV):
    write(LAUNCH / (name + '.command.json'), {'args': args, 'cwd': str(cwd), 'host': socket.getfqdn(),
        'tmux': os.environ['TMUX'], 'python': sys.executable, 'PYTHONPATH': ENV['PYTHONPATH']})
    with (LAUNCH / (name + '.log')).open('x') as stream:
        subprocess.run(args, cwd=cwd, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)

LAUNCH.mkdir(exist_ok=False)
try:
    plan = read(PLAN)
    states = {t['id']: t['status'] for t in plan['tasks']}
    current = next(t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
    assert plan['document']['version'] == '1.3.1' and current['id'] == 'NAV-065' and sum(states.values()) == 52
    write(LAUNCH / 'plan_before.json', plan)
    write(LAUNCH / 'context.json', {'plan_sha256': sha(PLAN), 'global_constraints': plan['global_constraints'],
        'execution_alignment_policy': plan['execution_alignment_policy'], 'uncertainty_policy': plan['uncertainty_policy'],
        'upstream_correction_policy': plan['upstream_correction_policy'],
        'tasks': [{k: v for k, v in t.items() if k != 'work_records'} for t in plan['tasks']],
        'current_work_records': current['work_records']})
    proposal = read(P.PROPOSAL)
    assert proposal['approved'] is False and not REG.exists()
    assert proposal['resource_limit']['update_candidates'] == [P.TARGET]
    assert proposal['fit']['numeric_steps_max'] == 300 and proposal['fit']['boundary_steps_max'] == 100
    assert proposal['resource_limit']['optimizer_trials_total_max'] == 400
    assert not (P.R.WORK / proposal['output_name']).exists()
    assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT)
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT / 'HiRA')
    review = read(NAV / 'reports/nav065_pilot_refit_v8_preparation_review.json')
    assert review['preparation_pass'] and review['basic_tests_passed'] == 57
    for name, digest in review['evidence'].items():
        assert sha(Path(name)) == digest, name
    process_text = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
    conflicts = [line.split()[0] for line in process_text.splitlines() if re.search(r'python(?:\d+(?:\.\d+)*)?\s+(?:-u\s+)?(?:\S*/)?refit_nav065_\S+\.py\s+--registration', line)]
    assert not conflicts, 'Existing fitting process must be reconciled, never duplicate'
    v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
    assert len(v2) == 33
    for name, digest in v2.items():
        assert sha(Path(name)) == digest, name
    write(LAUNCH / 'resources.json', {'host': socket.getfqdn(), 'tmux': os.environ['TMUX'],
        'python': sys.executable, 'version': sys.version, 'cwd': str(NAV), 'active_refit_processes': 0,
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True, timeout=20),
        'v2_verified_files': v2, 'ssh_exception': 'Existing user approval permits direct same-host tiger11/tmux execution.'})
    command('validate-before', [PYTHON, VALIDATOR, str(PLAN), '--mode', 'update'])
    protocol = {**proposal, 'approved': True, 'status': 'APPROVED',
        'authorization_reference': 'Latest user confirmation: 确认, approving the reviewed v8 full_s0-only pilot, <=300 numeric Adam + <=100 boundary SGD updates, <=400 total, saved train only, no new rollout/development scoring or automatic expansion.',
        'implementation_hashes': {name: sha(NAV / name) for name in P.BOUND_INPUTS}}
    add(REG, json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    add(REG.with_suffix('.json.sha256'), sha(REG) + '\n')
    protocol, pin = P.approved_registration(REG)
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
        'summary': 'NAV-065/1.3.1 v8用户确认后开工：冻结单候选数值优先过程与400步新额度，执行一次独立pilot',
        'actions': ['重读根AGENTS、legacy技能/规则及当前交接和方案；完整解析计划，核对全局约束、当前任务/依赖及原历史证据。',
            '登记用户最新确认：固定full_s0，最多300数值Adam+100边界无动量SGD/总400；数值未过不切换，原train标准保持；不复用旧余量。',
            'tiger11/tmux核验解释器、GPU、无并发拟合、33项v2及57基本检查对应hash；新增批准登记/pin和只读源码快照，原proposal=false保留。'],
        'files_changed': [str(REG.relative_to(ROOT)), str(REG.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), str((HERE / 'launch.py').relative_to(ROOT))],
        'commands': ['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v8-run；cwd=' + str(NAV) + '；Python=' + PYTHON,
            '<Python> -u .worklog-testing/NAV-065/pilot-v8-execution/launch.py（绝对路径）；完整命令/资源/源码快照=' + str(LAUNCH)],
        'results': ['批准登记pin=' + pin, '本项为未来NAV-066提供可信同actor双critic；本pilot仅验证full_s0原train要求，不自动扩六候选、开发评分、部署或预算搜索。',
            '复用该候选162条train（90旧+72临界），六候选972条仅来源核验；新增回合0、开发评分0。来源固定v7末次，无中间checkpoint选择。',
            '保持原共同停止cost/reach目标/W、beta=0、网络/obs-v7、原v5 train/G4、六actor、固定v2/匹配校准及地图/时序/J/分片。',
            '主agent按legacy验收，无测试子agent。验证每步门禁/起点日志、阶段/末次CPU重载及源码/actor身份；单候选通过也不等于NAV-065通过，未过保持false。'],
        'evidence': ['用户最新明确确认', 'f6ce79f92b60eaa007656a7fbe553d629e86575c', 'CaMRA/navigation/reports/nav065_pilot_refit_v8_preparation_review.json', str(LAUNCH / 'registration.json')],
        'remaining_work': ['提交批准版本并执行一次独立pilot，按真实产物验收和保存结果；保持原G4 FAIL、NAV-065=false及NAV-066阻塞。'], 'blockers': []})
    files = [str(REG.relative_to(ROOT)), str(REG.with_suffix('.json.sha256').relative_to(ROOT)), str(PLAN.relative_to(ROOT)), str((HERE / 'launch.py').relative_to(ROOT))]
    subprocess.run(['git', 'add', '--', *files], cwd=ROOT, check=True)
    assert set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()) == set(files)
    command('approval-diff-check', ['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], ROOT)
    command('approval-commit', ['git', 'commit', '-m', 'Register approved NAV-065 single-candidate pilot v8'], ROOT)
    write(LAUNCH / 'training_start.json', {'time': datetime.now().astimezone().isoformat(),
        'git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'registration_pin': pin})
    print('APPROVED_AND_STARTED', pin, flush=True)
    command('training', [PYTHON, '-u', 'scripts/refit_nav065_pilot.py', '--registration', str(REG.relative_to(NAV))])
    assert all(sha(NAV / name) == digest for name, digest in hashes.items()), 'Source changed during pilot'
    write(LAUNCH / 'EXIT.json', {'returncode': 0, 'time': datetime.now().astimezone().isoformat(),
        'source_unchanged': True, 'scientific_result': 'Inspect PILOT_FINISHED/train_gate; process exit is not scientific pass.'})
    print('FINISHED', flush=True)
except BaseException as exc:
    write(LAUNCH / 'ERROR.json', {'error': repr(exc), 'time': datetime.now().astimezone().isoformat()})
    raise
