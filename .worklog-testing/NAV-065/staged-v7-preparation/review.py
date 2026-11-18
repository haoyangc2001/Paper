"""Main-agent, legacy acceptance review of v7 preparation, not scientific pass."""
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

from worklog import append, ROOT, PLAN, VALIDATOR

NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())

# This newly created preparation resource note only needs the no-concurrency
# result. Exclude unrelated host command lines from the versioned artifact.
resource_path = HERE / 'resources.json'
resources = read(resource_path)
if 'processes' in resources:
    assert not any('scripts/refit_nav065_' in line and '--registration' in line and 'python' in line
                   for line in resources.pop('processes').splitlines())
    resources['active_nav065_refit_processes'] = 0
    original = resource_path.read_text()
    changed = json.dumps(resources, indent=2) + '\n'
    diff = ''.join(list(difflib.unified_diff([line + '\n' for line in original.splitlines()], changed.splitlines(True)))[2:])
    diff = re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
    subprocess.run(['apply_patch'], input='*** Begin Patch\n*** Update File: ' + str(resource_path) + '\n' + diff + '*** End Patch\n',
                   text=True, cwd=ROOT, check=True, capture_output=True)

log = (HERE / 'basic-attempt1.log').read_text()
assert '79 passed' in log and 'failed' not in log
diag = read(NAV / 'reports/nav065_staged_refit_preflight_v7.json')
assert diag['saved_gate_exact_match'] and len(diag['candidates']) == 6
assert (diag['train_episodes'], diag['train_steps'], diag['new_rollouts'], diag['optimizer_steps'],
        diag['development_scoring_rounds']) == (972, 372471, 0, 0, 0)
assert sum(len(c['starts']) for c in diag['candidates']) == 972
assert all(c['gate']['original_grid']['pass'] and not c['gate']['pass'] for c in diag['candidates'])
assert sum(s['classification'] == 'false_feasible' for c in diag['candidates'] for s in c['starts']) == 25
assert all(c['numeric_stage_required'] for c in diag['candidates'])
reasons = {}
for c in diag['candidates']:
    for reason, count in c['v6_rejected_reasons'].items():
        reasons[reason] = reasons.get(reason, 0) + count
assert reasons == {'previously_correct_start': 23, 'original_grid': 4}
proposal = read(NAV / 'config/nav065_staged_refit_proposal_v7.json')
assert proposal['approved'] is False
assert not (NAV / 'config/nav065_staged_refit_prereg_v7.json').exists()
assert not (Path('/pub/data/caohy/work/camra_navigation') / proposal['output_name']).exists()
review6 = read(NAV / 'reports/nav065_block_refit_v6_review.json')
for name, digest in review6['evidence']['output_hashes'].items():
    assert sha(Path(review6['evidence']['output']) / name) == digest
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest

baseline = json.loads(subprocess.check_output(['git', 'show', 'a8dd939:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
current = read(PLAN)
assert {k: v for k, v in current.items() if k != 'tasks'} == {k: v for k, v in baseline.items() if k != 'tasks'}
for old, now in zip(baseline['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert now['work_records'][:len(old['work_records'])] == old['work_records']
        assert {k: v for k, v in now.items() if k != 'work_records'} == {k: v for k, v in old.items() if k != 'work_records'}
    else:
        assert now == old
assert sum(t['status'] for t in current['tasks']) == 52
assert not subprocess.check_output(['git', 'diff', '--name-only', 'a8dd939', '--',
    'manuscripts', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
names = ['scripts/refit_nav065_staged.py', 'tests/test_nav065_staged_refit.py',
    'config/nav065_staged_refit_proposal_v7.json', 'reports/nav065_staged_refit_preflight_v7.json',
    'docs/nav065_staged_refit_v7_20261118.md']
report = {'schema': 'camra-nav065-v7-preparation-review', 'task': 'NAV-065',
    'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'cwd': str(NAV), 'python': sys.executable,
    'reviewer': 'main agent under legacy; no test subagents', 'basic_tests_passed': 79,
    'saved_train_episodes': 972, 'saved_train_steps': 372471, 'v2_verified_files': len(v2),
    'preparation_pass': True, 'scientific_pass': False, 'formal_optimizer_steps': 0,
    'new_rollouts': 0, 'development_scoring_rounds': 0, 'original_g4': 'FAIL',
    'nav065_status': False, 'completed_tasks': 52, 'nav066_started': False,
    'proposal_approved': False, 'remaining_train_false_feasible': 25,
    'v6_rejected_block_reasons': reasons, 'source_commit': 'a8dd939',
    'evidence': {str(NAV / name): sha(NAV / name) for name in names},
    'basic_log_sha256': sha(HERE / 'basic-attempt1.log'),
    'diagnosis_log_sha256': sha(HERE / 'diagnose-attempt1.log'),
    'resources_sha256': sha(resource_path),
    'production_chain': ['refit_nav065_staged.fit -> unchanged refit_nav065_adaptive.train_gate after every step',
        'refit_nav065_staged.run -> all six saved/reloaded and train gates pass -> unchanged refit_nav065_deployment.evaluate -> original G4'],
    'limitations': ['Staged optimization is not scientifically exercised in this preparation.',
        'No rejected v6 per-start logits/checkpoints exist; exact changed-start identities cannot be retrospectively reconstructed. No real optimizer replay.',
        'Real source reconstruction uses the unchanged oracle; independent raw-J/common-stop evidence is inherited from v6 only through matching output hashes.',
        'Source 25 false-feasible starts remain. Original G4 and downstream block retained.']}
with (NAV / 'reports/nav065_staged_refit_v7_preparation_review.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
append({'timestamp': report['timestamp'],
    'summary': 'NAV-065/1.3.1 v7准备完成：79基本测试、972条train预检通过；新阶段规则与训练额度待确认',
    'actions': ['实现隔离两阶段critic拟合入口和未批准v7提案，先原标签数值回归再原符号梯度纠错；离线过程不再逐段锁住旧正确起点。',
        '逐步调用未改的v5完整train门禁，首次全过停止；补齐每步起点数值、误放行/误拒绝及主档变化记录，源模型与旧产物保留。',
        '主agent按legacy在tiger11/tmux完成79基本测试，含16项v7检查；未派测试子agent。',
        '只读核验972条train/372471步、v6来源/全部输出hash、实际train门禁及起点预测，33项v2匹配；未正式拟合、新采样或开发评分。',
        '同步AGENTS/导航README/交接/方案及机器报告；初次文档补丁因README标题不匹配整体拒绝，改正标题后应用，未改旧实验；资源元数据补丁因末尾无换行首次失败，补齐diff行尾后成功，失败日志保留。资源记录只保留任务预检结论，剔除无关进程命令。'],
    'files_changed': ['CaMRA/navigation/' + n for n in names] + ['CaMRA/navigation/reports/nav065_staged_refit_v7_preparation_review.json',
        'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
        'doc/plan/20261109_camra_navigation_implementation.json', '.worklog-testing/NAV-065/staged-v7-preparation/'],
    'commands': ['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v7-prepare/nav065-v7-basic/nav065-v7-diagnose/nav065-v7-review；cwd=CaMRA/navigation；Python=' + sys.executable,
        'PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <Python> -m pytest -q -p no:cacheprovider tests/test_nav065_staged_refit.py tests/test_nav065_block_refit.py tests/test_nav065_adaptive_refit.py tests/test_nav065_guarded_refit.py tests/test_nav065_boundary_refit.py tests/test_nav065_saved_refit.py',
        '同环境 <Python> -u .worklog-testing/NAV-065/staged-v7-preparation/preflight.py及review.py（绝对路径）',
        'legacy validate_worklog.py --mode update；git -c core.whitespace=cr-at-eol diff --check'],
    'results': ['79/79基本测试和真实train预检通过；六候选的原train门禁重算与v6精确一致，33项v2身份未改。',
        '正式optimizer步数/新回合/development重评均0；原25次误放行、完整train 0/6、G4 FAIL、NAV-065=false、52/67及NAV-066未启动保持。',
        '新提案为每候选数值150+边界150、总1800新试步；不转移阶段或旧run余量；离线过程保护变化及新额度尚未批准。',
        '原共同停止cost/reach目标/W/beta、最终v5 train/G4、actor/v2及校准、地图/时序/单位/分片和只读参考/论文均保持。受控生产入口测试证明仅六组train全过、保存CPU重载后才一次原180 development重评，原G4决定交付。',
        'v6的23次旧正确起点保护/4次旧五档拒绝可复算，但缺中间逐起点预测，无法断定具体损失类型；未重放真实optimizer。新版分阶段是否有效尚待正式验证，不保证全部预测正确。'],
    'evidence': ['CaMRA/navigation/reports/nav065_staged_refit_v7_preparation_review.json', 'CaMRA/navigation/reports/nav065_staged_refit_preflight_v7.json',
        'CaMRA/navigation/config/nav065_staged_refit_proposal_v7.json', '.worklog-testing/NAV-065/staged-v7-preparation/'],
    'remaining_work': ['明确确认新阶段过程规则及每候选150+150/总1800额度后，冻结批准登记/hash、现场复核并运行一次新目录；全部train与原G4通过后才能交付NAV-066。'],
    'blockers': ['新数值额度与取消过程单调保护尚待确认；本轮仅准备，critic最终验收未通过。']})
subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)
print(json.dumps({'preparation_pass': True, 'scientific_pass': False, 'basic_tests': 79,
    'formal_updates': 0, 'new_rollouts': 0, 'development_scoring': 0, 'v2_files': len(v2)}))
