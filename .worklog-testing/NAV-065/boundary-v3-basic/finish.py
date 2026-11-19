"""Record the main-agent basic checks without granting scientific release."""

from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
E = ROOT / '.worklog-testing/NAV-065/boundary-v3-basic'
NAV = ROOT / 'CaMRA/navigation'
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = json.loads((E / 'before-plan.json').read_text())
old_text = PLAN.read_text()
current = json.loads(old_text)
assert [t['id'] for t in before['tasks']] == [t['id'] for t in current['tasks']]
for a, b in zip(before['tasks'], current['tasks']):
    if a['id'] != 'NAV-065':
        assert a == b
    else:
        assert a['status'] == b['status'] is False
        assert b['work_records'][:len(a['work_records'])] == a['work_records']
        assert len(b['work_records']) == len(a['work_records']) + 1
        assert {k: v for k, v in a.items() if k != 'work_records'} == {
            k: v for k, v in b.items() if k != 'work_records'}

proposal_path = NAV / 'config/nav065_boundary_refit_proposal_v3.json'
preflight_path = NAV / 'reports/nav065_boundary_refit_preflight_v3.json'
proposal = json.loads(proposal_path.read_text())
preflight = json.loads(preflight_path.read_text())
assert proposal['approved'] is False
assert preflight['proposal_sha256'] == sha(proposal_path)
assert (preflight['optimizer_updates'], preflight['new_rollouts'],
        preflight['development_scoring_rounds']) == (0, 0, 0)
assert preflight['train_episodes'] == 540 and preflight['train_steps'] == 207298
assert sum(c['pass'] for c in preflight['candidates']) == 4
assert sum(b['false_feasible'] for c in preflight['candidates'] for b in c['by_budget'].values()) == 4
assert '22 passed' in (E / 'tests-final.log').read_text()
assert 'nav065-boundary-v3 pane=' in (E / 'window-verified.log').read_text()
assert not (Path('/pub/data/caohy/work/camra_navigation') / proposal['output_name']).exists()

# Retain the old v2 implementation pins, not merely a fresh hash of altered code.
v2 = json.loads((NAV / 'config/nav065_saved_refit_prereg_v2.json').read_text())
for name, digest in v2['implementation_hashes'].items():
    assert sha(NAV / name) == digest

files = ['CaMRA/navigation/scripts/refit_nav065_boundary.py',
         'CaMRA/navigation/config/nav065_boundary_refit_proposal_v3.json',
         'CaMRA/navigation/tests/test_nav065_boundary_refit.py',
         'CaMRA/navigation/docs/nav065_boundary_refit_v3_20261118.md',
         'CaMRA/navigation/reports/nav065_boundary_refit_preflight_v3.json']
report = {
    'task': 'NAV-065', 'reviewer': 'main_agent', 'independent': False,
    'skill': 'create-plan-worklog-legacy', 'status': 'BASIC_PASS_AWAITING_NEW_QUOTA',
    'timestamp': datetime.now().astimezone().isoformat(), 'hostname': socket.getfqdn(),
    'basic_pass': True, 'scientific_pass': False, 'formal_optimizer_updates': 0,
    'new_rollouts': 0, 'development_scoring_rounds': 0,
    'synthetic_gradient_test': 'one toy optimizer step; no production tensors updated',
    'tests': {'passed': 22, 'log': str(E / 'tests-final.log')},
    'preflight': {'train_episodes': 540, 'train_steps': 207298,
                  'passing_candidates': 4, 'false_feasible': 4,
                  'full_s2_policy_primary_actual_feasible': 17, 'primary_total': 18},
    'protected': {'old_v2_bound_code_unchanged': True,
                  'fixed_v2_inputs': '33 files verified by pinned production preflight',
                  'actor_actions': 'all 207298 saved train actions replayed within 1e-6',
                  'formal_run_directory_created': False},
    'source_hashes': {p: sha(ROOT / p) for p in files},
    'evidence_hashes': {name: sha(E / name) for name in (
        'tests.log', 'tests-final.log', 'preflight.log', 'validation-opening.log',
        'validation-opening-attempt2.log', 'window-verified.log', 'before-plan.json')},
    'known_limitations': [
        'No real v3 fitting or development scoring; convergence and G4 remain unproven.',
        'Train recognition rule changed only in a new protocol; old failed attempt unchanged.',
        'Four source passes include two nonprimary false-infeasibles, retained as diagnostics.',
        'First validator invocation used unsupported --previous; error log retained.',
        'Initial host log named the selected monitor window; window-verified.log records actual pane.'],
    'downstream': 'NAV-065=false, 52/67, existing G4 FAIL; NAV-066 not started',
}
with (E / 'report.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')

t = next(t for t in current['tasks'] if t['id'] == 'NAV-065')
old_verification = t['verification_result']
t['work_records'].append({
    'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 边界符号回修v3入口及基础验证完成；仅准备，科学门禁未放行',
    'actions': [
        '按用户指定legacy执行，主agent直接测试，不安排专用测试子agent。',
        '新增隔离v3，固定actor并保留共同停止原目标；增加训练起点分量符号辅助损失，部署W/beta和development判据不变。',
        'train主预算识别要求按真实可行数分列；full s2须接受17并拒绝超时，不要求用critic使固定轨迹18/18成功，不回填旧v2 PASS。',
        'tiger11/tmux实际预检540条train/207298步，源actor确定性动作及真值复算；校验原登记、源模型和33项v2资源。',
        '维护独立未批准提案及来源绑定、零更新保留通过模型、首次通过即停、train失败阻断开发与单次重评调用链。'],
    'files_changed': files + ['AGENTS.md', 'CaMRA/navigation/docs/implementation_handoff.md',
                            'doc/plan/20261109_camra_navigation_implementation.json', str(E)],
    'commands': [
        'tmux=nav064-r14-repair-20261115:nav065-boundary-v3, pane=%482; cwd=/home/caohy/repositories/Paper/CaMRA/navigation; Python=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python',
        'PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -m pytest -q -p no:cacheprovider tests/test_nav065_boundary_refit.py tests/test_nav065_saved_refit.py',
        'PYTHONPATH=src:scripts /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -u scripts/refit_nav065_boundary.py --preflight --report reports/nav065_boundary_refit_preflight_v3.json',
        'legacy validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update；追加历史另与before-plan.json逐项比对。'],
    'results': [
        '22项基础测试通过；合成单步更新actor不变、两critic有梯度；正式优化0、新回合0、development重评0。',
        '四组源模型train通过；full s1/s2各两次假可行仍失败。主档固定策略可行分别18/18与17/18，不能将统计门槛修正当作修复四次误判。',
        '未批准提案拟只对两个失败候选各最多300、合计600新更新；其余四组零更新，全部train通过才复用180 development一次，未创建正式run。',
        '旧v2绑定代码散列不变；新辅助项只改critic优化重心，原目标精确值辅助损失为0；Go2/v2/时序/预算单位/共同停止与开发评分保持。',
        '首次legacy校验误用--previous参数，保留失败日志后改为其支持的--mode update通过；初始host日志窗口名取了当前选中monitor，另以实际pane更正记录。',
        '局部基础检查PASS；全局端到端科学验收仍缺实际拟合、开发G4与交付，52/67、NAV-065=false，NAV-066不启动。',
        '本次前verification_result保留：' + old_verification],
    'evidence': [str(E / 'report.json'), str(E / 'tests-final.log'), str(E / 'preflight.log'),
                 str(E / 'window-verified.log'), 'CaMRA/navigation/reports/nav065_boundary_refit_preflight_v3.json',
                 'CaMRA/navigation/docs/nav065_boundary_refit_v3_20261118.md'],
    'remaining_work': ['冻结新增优化额度及独立批准登记；执行有界拟合；六组train全通过后唯一开发复验，原G4通过才能交付NAV-066。'],
    'blockers': ['旧额度已耗尽，新提案approved=false，不能执行正式优化；两组仍有四次假可行，无新G4 PASS或部署包。'],
})
t['verification_result'] = ('NAV-065边界回修v3：主agent基础22测试及540条train预检PASS；'
                            '正式拟合0、开发重评0、新额度待确认。旧G4 FAIL有效，NAV-065=false。')
new_text = json.dumps(current, ensure_ascii=False, indent=1) + '\n'
diff = ''.join(list(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True), n=3))[2:])
diff = re.sub(r'^@@.*?@@.*$', '@@', diff, flags=re.M)
with (E / 'closeout.patch').open('x') as stream:
    stream.write('*** Begin Patch\n*** Update File: doc/plan/20261109_camra_navigation_implementation.json\n'
                 + diff + '*** End Patch\n')
print('BASIC evidence PASS; closure patch prepared, not applied')
