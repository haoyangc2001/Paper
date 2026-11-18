"""Main-agent preparation audit. No real optimizer steps or development scoring."""
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import textwrap

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = ROOT / '.worklog-testing/NAV-065/block-v6-preparation'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
read = lambda path: json.loads(path.read_bytes())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

tests = (HERE / 'basic-attempt1.log').read_text()
assert '63 passed' in tests and 'failed' not in tests
diag = read(NAV / 'reports/nav065_block_refit_preflight_v6.json')
assert (diag['train_episodes'], diag['train_steps'], diag['new_rollouts'],
        diag['optimizer_steps'], diag['development_scoring_rounds']) == (972, 372471, 0, 0, 0)
assert len(diag['candidates']) == 6
assert sum(c['gate']['additional_false_feasible'] for c in diag['candidates']) == 27
assert all(c['gate']['original_grid']['pass'] and not c['gate']['pass'] for c in diag['candidates'])
proposal = read(NAV / 'config/nav065_block_refit_proposal_v6.json')
assert proposal['approved'] is False
assert not (Path('/pub/data/caohy/work/camra_navigation') / proposal['output_name']).exists()
assert not (NAV / 'config/nav065_block_refit_prereg_v6.json').exists()
previous = read(NAV / 'reports/nav065_adaptive_refit_v5_review.json')
for name, digest in previous['evidence']['output_hashes'].items():
    assert sha(Path(previous['evidence']['output']) / name) == digest
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for path, digest in v2.items():
    assert sha(Path(path)) == digest
baseline = json.loads(subprocess.check_output(['git', 'show', 'd9f8bb6:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
current = read(PLAN)
assert {k: v for k, v in current.items() if k != 'tasks'} == {k: v for k, v in baseline.items() if k != 'tasks'}
for old, now in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert now['work_records'][:len(old['work_records'])] == old['work_records']
        assert {k: v for k, v in now.items() if k != 'work_records'} == {k: v for k, v in old.items() if k != 'work_records'}
    else:
        assert old == now
assert sum(t['status'] for t in current['tasks']) == 52
assert not subprocess.check_output(['git', 'diff', '--name-only', 'd9f8bb6', '--', 'RC-PPO', 'HiRA', 'manuscripts'], cwd=ROOT)
resources = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.free', '--format=csv,noheader'], text=True, timeout=20)
with (HERE / 'resources.log').open('x') as stream:
    stream.write(socket.getfqdn() + '\n' + sys.executable + '\n' + resources)
files = ['scripts/refit_nav065_blocks.py', 'tests/test_nav065_block_refit.py',
         'config/nav065_block_refit_proposal_v6.json', 'reports/nav065_block_refit_preflight_v6.json',
         'docs/nav065_block_refit_v6_20261118.md']
report = {'task': 'NAV-065', 'schema': 'camra-nav065-v6-preparation-review',
    'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'cwd': str(NAV), 'python': sys.executable,
    'reviewer': 'main agent, legacy; no test subagents', 'basic_tests_passed': 63,
    'saved_train_episodes': 972, 'saved_train_steps': 372471, 'v2_verified_files': len(v2),
    'formal_optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'preparation_pass': True, 'scientific_pass': False, 'original_g4': 'FAIL',
    'nav065_status': False, 'completed_tasks': 52, 'nav066_started': False,
    'proposal_approved': False, 'source_commit': 'd9f8bb6',
    'evidence': {str(NAV / name): sha(NAV / name) for name in files},
    'test_log_sha256': sha(HERE / 'basic-attempt1.log'),
    'diagnosis_log_sha256': sha(HERE / 'diagnose-attempt1.log'),
    'limitations': ['New optimizer/working guards not approved or scientifically exercised.',
        '27 train false-feasible starts remain; original development/G4 result retained.',
        'Train truth uses the unchanged oracle; prior v5 independent raw-J audit is inherited only through matching hashes.']}
write(NAV / 'reports/nav065_block_refit_v6_preparation_review.json', report)
record = {
    'timestamp': report['timestamp'],
    'summary': 'NAV-065/1.3.1 v6准备完成：63基本检查与972条train只读预检通过，正式新额度和工作保护待确认',
    'actions': ['实现隔离refit_nav065_blocks.py及未批准提案；按旧/新、起点/沿途/边界和reach目标正负分层回归，原标签不变。',
        '实现最多4步一段、源锚定工作波动界、拒绝整段时模型/Adam/随机数恢复及实际试步计量；原v5最终train和原G4消费者保持。',
        '主agent在tiger11/tmux运行63项基本测试，含15项新检查；未派测试子agent。',
        '只读核验972条train/372471步、v5产物hash、432条追加预算顺序/未裁剪账本及真实actor动作，六actor和33项v2身份未改；诊断回归/符号梯度。',
        '保留第一次补丁参数过长的工程失败及第二次成功日志；同步说明、交接和入口，保持计划历史和所有任务状态。'],
    'files_changed': ['CaMRA/navigation/' + name for name in files] + [
        'CaMRA/navigation/reports/nav065_block_refit_v6_preparation_review.json',
        'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
        'doc/plan/20261109_camra_navigation_implementation.json', '.worklog-testing/NAV-065/block-v6-preparation/'],
    'commands': ['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v6-basic/nav065-v6-diagnose/nav065-v6-review；cwd=CaMRA/navigation；Python=' + sys.executable,
        'PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <Python> -m pytest -q -p no:cacheprovider tests/test_nav065_block_refit.py tests/test_nav065_adaptive_refit.py tests/test_nav065_guarded_refit.py tests/test_nav065_boundary_refit.py tests/test_nav065_saved_refit.py',
        '同环境 <Python> -u scripts/refit_nav065_blocks.py --diagnose reports/nav065_block_refit_preflight_v6.json',
        '同环境 <Python> -u .worklog-testing/NAV-065/block-v6-preparation/review.py（绝对脚本路径）；legacy validate_worklog.py --mode update；git -c core.whitespace=cr-at-eol diff --check'],
    'results': ['63/63基本测试；真实train来源/预算/动作预检通过，972条/372471步；正式optimizer更新0、新增回合0、development重评0。',
        '到达critic在本次固定分层批次的回归/符号梯度余弦full三组约-0.95/-0.66/-0.73，支持局部方向冲突，不证明新版一定能拟合。',
        '原27次新增train误放行、原G4 FAIL、NAV-065=false、52/67及NAV-066未启动保持；局部实现/来源检查通过不等于科学通过。',
        'v6 proposal=false，无批准登记/正式run；请求每候选300新试步/总1800，复用972 train，无新增回合；工作保护例外和数字额度待明确确认，不能重用v5余量。',
        '原目标、共同停止、W、beta、最终G4及v5 train标准、六actor/v2及匹配校准、地图/时序/单位、分片及只读参考/论文保持；只有六组train全过后才一次原180 development重评、原G4决定交付。'],
    'evidence': ['CaMRA/navigation/reports/nav065_block_refit_v6_preparation_review.json',
        'CaMRA/navigation/reports/nav065_block_refit_preflight_v6.json',
        'CaMRA/navigation/config/nav065_block_refit_proposal_v6.json', '.worklog-testing/NAV-065/block-v6-preparation/'],
    'remaining_work': ['明确批准新工作保护例外和试步额度后，独立冻结登记、现场复核并执行一次新run；全train和原G4均过才完成NAV-065并交付NAV-066。'],
    'blockers': ['正式新数字额度与工作保护例外尚未确认；本轮仅准备，原价值边界误判仍未解决。']}
original = PLAN.read_text()
start = original.index('"work_records": [', original.index('"id": "NAV-065"')) + len('"work_records": ')
_, length = json.JSONDecoder().raw_decode(original[start:])
changed = original[:start + length - 1].rstrip() + ',\n' + textwrap.indent(json.dumps(record, ensure_ascii=False, indent=1), '    ') + '\n   ]' + original[start + length:]
diff = ''.join(list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True)))[2:])
diff = re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
subprocess.run(['apply_patch'], input='*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n' + diff + '*** End Patch\n', text=True, cwd=ROOT, check=True)
subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], check=True)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
print(json.dumps({'preparation_pass': True, 'scientific_pass': False, 'basic_tests': 63,
    'formal_updates': 0, 'new_rollouts': 0, 'development_scoring': 0, 'v2_files': len(v2)}))
