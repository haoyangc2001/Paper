"""Legacy main-agent preparation acceptance; no candidate optimization."""
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
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
sys.path.insert(0, str(ROOT / '.worklog-testing/NAV-065/staged-v7-preparation'))
from worklog import append, VALIDATOR
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

basic_log = (HERE / 'basic-attempt1.log').read_text()
assert '57 passed' in basic_log and 'failed' not in basic_log
diag = read(NAV / 'reports/nav065_pilot_refit_preflight_v8.json')
assert diag['source_gate_exact_match'] and len(diag['candidates']) == 6
assert (diag['train_episodes'], diag['train_steps'], diag['optimizer_steps'],
        diag['new_rollouts'], diag['development_scoring_rounds']) == (972, 372471, 0, 0, 0)
assert sum(len(c['starts']) for c in diag['candidates']) == 972
assert sum(c['input_aliases']['rows'] for c in diag['candidates']) == 372471
assert all(c['input_aliases']['duplicate_input_groups'] == 0 for c in diag['candidates'])
assert all(not c['gate']['pass'] for c in diag['candidates'])
false_feasible = sum(s['classification'] == 'false_feasible' for c in diag['candidates'] for s in c['starts'])
false_infeasible = sum(s['classification'] == 'false_infeasible' for c in diag['candidates'] for s in c['starts'])
assert (false_feasible, false_infeasible) == (19, 64)
opposing = sum(c['v7_journal']['boundary_gradients']['critic_reach']['opposing_steps'] for c in diag['candidates'])
assert opposing == 874
proposal = read(NAV / 'config/nav065_pilot_refit_proposal_v8.json')
assert proposal['approved'] is False and proposal['resource_limit']['optimizer_trials_total_max'] == 400
assert proposal['resource_limit']['development_scoring_rounds_max'] == 0
assert proposal['resource_limit']['update_candidates'] == ['nav064_terminal_tail_v1_full_s0']
assert not (NAV / 'config/nav065_pilot_refit_prereg_v8.json').exists()
assert not (Path('/pub/data/caohy/work/camra_navigation') / proposal['output_name']).exists()
resources = read(HERE / 'resources.json')
assert resources['v2_verified_files'] == 33 and resources['active_nav065_refit_processes'] == 0
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
for name, digest in v2.items():
    assert sha(Path(name)) == digest
launch = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7')
for name, digest in read(launch / 'source_before.json').items():
    assert sha(NAV / name) == digest
review7 = read(NAV / 'reports/nav065_staged_refit_v7_review.json')
root7 = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_v7')
for name, digest in review7['evidence']['output_hashes'].items():
    assert sha(root7 / name) == digest
assert sha(root7 / 'fitted_candidates.json') == proposal['source_manifest_sha256']

def check_history():
    baseline = json.loads(subprocess.check_output(['git', 'show', 'HEAD:doc/plan/20261109_camra_navigation_implementation.json'], cwd=ROOT))
    current = read(PLAN)
    assert {k: v for k, v in current.items() if k != 'tasks'} == {k: v for k, v in baseline.items() if k != 'tasks'}
    for old, now in zip(baseline['tasks'], current['tasks'], strict=True):
        if old['id'] == 'NAV-065':
            assert now['work_records'][:len(old['work_records'])] == old['work_records']
            assert {k: v for k, v in now.items() if k != 'work_records'} == {k: v for k, v in old.items() if k != 'work_records'}
        else:
            assert old == now
    assert sum(t['status'] for t in current['tasks']) == 52
    assert not subprocess.check_output(['git', 'diff', '--name-only', '--',
        'manuscripts', 'RC-PPO', 'HiRA', 'CaMRA/locomotion', 'CaMRA/calibration'], cwd=ROOT)
    subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)

check_history()
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
names = ['scripts/refit_nav065_pilot.py', 'tests/test_nav065_pilot_refit.py',
    'config/nav065_pilot_refit_proposal_v8.json', 'reports/nav065_pilot_refit_preflight_v8.json',
    'docs/nav065_pilot_refit_v8_20261118.md']
report = {'schema': 'camra-nav065-pilot-v8-preparation-review', 'task': 'NAV-065',
    'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
    'tmux': os.environ['TMUX'], 'cwd': str(NAV), 'python': sys.executable,
    'reviewer': 'main agent under create-plan-worklog-legacy; no test subagents',
    'basic_tests_passed': 57, 'preparation_pass': True, 'scientific_pass': False,
    'saved_train_episodes': 972, 'saved_train_steps': 372471, 'v2_verified_files': len(v2),
    'formal_optimizer_steps': 0, 'new_rollouts': 0, 'development_scoring_rounds': 0,
    'source_gate_exact_match': True, 'exact_duplicate_input_groups': 0,
    'v7_reach_opposing_gradient_steps': opposing, 'v7_reach_boundary_steps': 900,
    'source_false_feasible': false_feasible, 'source_false_infeasible': false_infeasible,
    'original_g4': 'FAIL', 'nav065_status': False, 'completed_tasks': 52,
    'nav066_started': False, 'proposal_approved': False,
    'evidence': {str(NAV / name): sha(NAV / name) for name in names},
    'basic_log_sha256': sha(HERE / 'basic-attempt1.log'),
    'preflight_log_sha256': sha(HERE / 'preflight-attempt1.log'),
    'resources_sha256': sha(HERE / 'resources.json'),
    'production_chain': ['refit_nav065_pilot.fit -> unchanged stratified regression / original component-joint signs -> original v5 train_gate after every step',
        'numeric checks are mandatory before boundary; regression-first correction uses plain SGD and never amplifies a small sign gradient',
        'refit_nav065_pilot.run -> save and CPU-reload fixed pilot; other five unchanged; no development/evaluate/G4 call even if pilot passes',
        'future six-candidate expansion requires separate bounded review, all original train/G4 checks and no automatic downstream release'],
    'limitations': ['No candidate optimizer was executed; synthetic toy optimizer tests are only basic verification.',
        'Exact-input uniqueness does not prove sufficient observation or neural representation/generalization.',
        'First-order sampled-MSE protection is not a finite-step numerical/classification or convergence guarantee.',
        'Original raw-J/common-stop audit is inherited from matching v7 evidence; this round reconstructed all train gates without changing the oracle.',
        'v7 failed final models remain the fixed sources; no intermediate checkpoint or development-based candidate selection.']}
with (NAV / 'reports/nav065_pilot_refit_v8_preparation_review.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
append({'timestamp': report['timestamp'],
    'summary': 'NAV-065/1.3.1 v8准备收尾：57基本检查与全量train输入诊断通过，单候选400步提案待确认',
    'actions': ['实现隔离v8单候选入口：先原数值门槛、再保留回归方向并只投影/限幅符号纠错；边界无动量SGD，不放大微小纠错梯度。',
        '直接调用未改的v5 train_gate、原目标及分层回归；记录全部起点分类变化、阶段/末次模型，数值未过不强行切换，pilot通过也不调用development或交付部署包。',
        '主agent在tiger11/tmux运行57项基本检查及972条train/372471步只读诊断，无测试子agent；原v7源码/产物、固定actor及33项v2身份复核匹配。',
        '按每actor检查全量float32输入精确碰撞、相邻预算标签及v7阶段梯度，实际末次train门禁重建与保存结果逐项一致；未正式优化、采样或开发评分。',
        '同步未批准提案、诊断/准备报告、方案、AGENTS/导航README/交接和当前任务追加记录，保留旧证据与全部状态。'],
    'files_changed': ['CaMRA/navigation/' + name for name in names] + [
        'CaMRA/navigation/reports/nav065_pilot_refit_v8_preparation_review.json',
        'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
        'doc/plan/20261109_camra_navigation_implementation.json', '.worklog-testing/NAV-065/pilot-v8-preparation/'],
    'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav065-v8-basic/nav065-v8-preflight/nav065-v8-review；cwd=CaMRA/navigation；Python=' + sys.executable,
        'PYTHONPATH=src <Python> -m pytest -q tests/test_nav065_pilot_refit.py tests/test_nav065_staged_refit.py tests/test_nav065_guarded_refit.py tests/test_deployment_oracle.py',
        '<Python> -u .worklog-testing/NAV-065/pilot-v8-preparation/preflight.py、review.py（绝对路径）；legacy validate_worklog.py --mode update'],
    'results': ['57基本检查通过，972条train/372471步预检通过，原v5 gate与v7末次逐项一致；各actor无完全重复输入，不等于观测充分。',
        'v7到达网络900次边界更新有874次梯度相反；full_s0 L3-s1003预算仅变3.8007 J，到达分量+0.0021004→−3，W跨零，支持陡变目标与优化冲突诊断，不能推断唯一根因。',
        '新增提案固定full_s0/v7末次来源，最多300纯数值+100边界、共400新更新；其他五模型不动，新回合/开发评分0，通过也不自动扩展。提案approved=false，无新批准登记或正式run。',
        '原共同停止cost/reach/W、beta=0、网络/obs-v7、六actor、v2/校准及地图/时序/J单位/分片保持；真实生产入口使用原v5 gate，后续六候选与原G4仍必须验收。',
        '准备与执行完整性证据不等于科学通过；原19误放行/64误拒绝、0/6完整train、G4 FAIL、NAV-065=false、52/67和NAV-066未启动保持。'],
    'evidence': ['CaMRA/navigation/reports/nav065_pilot_refit_preflight_v8.json',
        'CaMRA/navigation/reports/nav065_pilot_refit_v8_preparation_review.json',
        'CaMRA/navigation/config/nav065_pilot_refit_proposal_v8.json',
        'CaMRA/navigation/docs/nav065_pilot_refit_v8_20261118.md',
        '.worklog-testing/NAV-065/pilot-v8-preparation/basic-attempt1.log',
        '.worklog-testing/NAV-065/pilot-v8-preparation/preflight-attempt1.log'],
    'remaining_work': ['确认具体单候选额度及新优化过程后，另冻批准登记/pin并现场复核，只执行一次新目录试验；不续跑v7或转移旧额度。',
        'pilot若通过仍需另行评审六候选扩展；全部原train/G4及部署包通过后才能完成NAV-065和进入NAV-066。'],
    'blockers': ['新过程及400次optimizer数字额度尚未明确批准；当前真实科学失败未解决，正式训练及下游保持阻塞。']})
check_history()
print(json.dumps({'preparation_pass': True, 'scientific_pass': False, 'basic_tests': 57,
    'formal_optimizer_steps': 0, 'completed_tasks': 52, 'proposal_approved': False}))
