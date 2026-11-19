"""Append a truthful failed NAV-065 review without changing any task status."""
from datetime import datetime
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
data = json.loads(PLAN.read_text())
task = next(t for t in data['tasks'] if t['id'] == 'NAV-065')
assert task['status'] is False and len(task['work_records']) == 17
assert sum(t['status'] for t in data['tasks']) == 52
prior = json.loads((ROOT / '.worklog-testing/NAV-065/main-verification/report.json').read_text())
review = json.loads((ROOT / 'CaMRA/navigation/reports/nav065_saved_refit_v2_review.json').read_text())
assert review['integrity_verification'] == 'PASS' and review['scientific_verdict'] == 'HELD_TRAIN_GATE'
stamp = datetime.now().astimezone().isoformat()
rid = 'NAV-065-MAIN-20261118-saved-refit-v2'
command = ('cwd=/home/caohy/repositories/Paper/CaMRA/navigation; '
    'tmux=nav064-r14-repair-20261115:nav065-saved-review; '
    'PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
    '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -u '
    '/home/caohy/repositories/Paper/.worklog-testing/NAV-065/saved-refit-v2-review/audit.py')

def evidence(name):
    path = ROOT / name
    return {'path': name, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}

ev = [evidence(name) for name in [
    '.worklog-testing/NAV-065/saved-refit-v2-review/audit.py',
    '.worklog-testing/NAV-065/saved-refit-v2-review/audit.log',
    '.worklog-testing/NAV-065/saved-refit-v2-review/train_cases.json',
    'CaMRA/navigation/reports/nav065_saved_refit_v2_review.json']]
mutation = copy.deepcopy(prior['mutation_testing'])
mutation['reason'] = ('复用本轮前真实执行且仍匹配未变生产源码的主agent隔离变异证据；'
    'S.approved_registration在事后复核中验证原测试/实现hash。未声称新派测试员或新跑变异。')
for item in mutation['evidence']:
    assert evidence(item['path']) == item

def check(cid, result, expected, actual):
    return {'id': cid, 'kind': 'command', 'result': result, 'invocation': command,
            'exit_code': 0, 'expected': expected, 'actual': actual, 'evidence': ev}

report = {'report_id': rid, 'task_id': task['id'], 'timestamp': stamp,
    'tester': {'agent_id': '/root', 'implementation_agent_id': '/root', 'independent': False},
    'verdict': 'fail',
    'summary': '主agent直接复核：640新增critic更新、540原train、4/6 train通过、4次假可行；development未重评、无新G4/部署包，NAV-065保持false。',
    'task_contract': {k: copy.deepcopy(task[k]) for k in (
        'id', 'goal', 'depends_on', 'constraints_and_notes', 'acceptance_criteria', 'test_requirements')},
    'tested_snapshot': json.loads((HERE / 'snapshot.json').read_text()),
    'checks': [
        check('CHECK-001', 'fail', '六候选身份可复算且完整交付通过的候选包。',
              '六组CPU重载、hash及actor逐tensor一致；4/6仅train通过，整体HELD，无部署包，未满足交付验收。'),
        check('CHECK-002', 'blocked', '新候选全部train通过后统一复验180条旧development一次。',
              '540条train真值/预测及全部误判已复算，暴力suffix argmin与J/事件交叉核验通过；train失败，按批准协议没有评分development，缺口保留。'),
        check('CHECK-003', 'pass', 'actor/v2不变、两critic更新、预算不饱和、更新/早停符合批准登记。',
              '33项v2经生产preflight核验，原受保护输入hash一致；六组actor逐tensor不变/两critic更新，预算裁剪0；更新10/10/10/10/300/300、合计640，首次通过即停。'),
        check('CHECK-004', 'fail', '六候选达到既定价值门槛才生成新G4和交付包。',
              'train假可行13→4，full s1/s2仍失败，按规则HELD；主档full s2真实17/18，批准train 18/18在固定actor下不可达；门槛原样保留，原G4 FAIL仍有效。')],
    'mutation_testing': mutation,
    'cross_validation': {'status': 'pass', 'method': '从540条原始轨迹的预算/距离/安全历史暴力枚举后缀最小停止点，核对生产oracle的每步stop/reach/W；以原始J账本和终止事件交叉判定全部可行标签。主agent实施，不伪称独立测试员。', 'evidence': ev}}
with (HERE / 'report.json').open('x') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write('\n')
old_verification = task['verification_result']
task['test_reports'].append(report)
task['verification_result'] = (rid + '：主agent直接验证，离线重拟合v2已结束；640新增更新、0新增采样，4/6 train通过、4次假可行。train门槛未过，development未重评，无新G4/交付包，原G4 FAIL有效；NAV-065=false、NAV-066未启动。')
task['work_records'].append({'timestamp': stamp,
    'summary': 'NAV-065/1.3.1 保存轨迹离线回修v2结束并完成主agent事后审计，HELD_TRAIN_GATE；新增发现train主档门槛的适用性缺口',
    'actions': ['核对15:00:49至15:01:35实际运行及EXIT/HELD、六份新模型与批准登记；不重启训练、不派专用测试子agent。',
        '逐CPU模型对照原actor张量、两critic变化及全部原受保护输入hash；生产预检核验33项v2，逐行检查更新序列、每10次门禁和首次通过即停。',
        '复算540条train、207298步；以暴力suffix argmin和原始J/安全/到达事件交叉核验共同停止、真值和全部4次假可行；未评分development。',
        '如实登记方案检查遗漏：批准train主档要求18/18之前未核实full s2实际只有17/18；本轮不改变门槛、不回填PASS。'],
    'files_changed': ['CaMRA/navigation/reports/nav065_saved_refit_v2_review.json',
        'CaMRA/navigation/docs/nav065_saved_refit_v2_review_20261118.md',
        'CaMRA/navigation/docs/implementation_handoff.md', 'AGENTS.md',
        '.worklog-testing/NAV-065/saved-refit-v2-review/', str(PLAN.relative_to(ROOT))],
    'commands': [command,
        'tmux run-shell: capture_test_snapshot.py doc/plan/20261109_camra_navigation_implementation.json --task NAV-065',
        '收尾后使用validate_worklog.py --mode update --previous .worklog-testing/NAV-065/before-saved-refit-approved-20261118.json；实际结果另存validation.log。'],
    'results': ['新登记pin=a6fe42eb12e94fa857af86702a8b617d2a93d02561e763620d887178afc4705f；新更新10/10/10/10/300/300，共640；新增采样0、development重评0。',
        '四候选train通过；full s1/s2各300仍失败。起点假可行13→4、假不可行0→2；成本起点MAE1.585→1.994 J、沿途1.343→1.431 J，不称全指标改善。',
        '剩3次k1.0超预算放行及1次k1.15超时放行。超时为full s2/train L2-s1006，45秒距目标0.577576 m，半径0.5 m、余31.103 J；critic拟合不能使固定actor多走7.76厘米。',
        '批准train 18/18与该固定轨迹不相容，哪怕完美critic也无法同时达到18次正确接受和零假可行；缺口不掩盖真实4次误判，原标准/失败/日志/模型全保留。',
        '局部完整性核验PASS，科学验收HELD；未产新G4或部署包，原G4 FAIL和NAV-064 development验收作为各自原分母证据保留；52/67、NAV-065=false、NAV-066及后续未启动。',
        '目标/生产源码、beta=0、共同停止、评分、原数据与冻结v2不变；新价值结果仅保留为未放行尝试，不能被预算搜索消费。',
        '旧verification_result保留于本条：' + old_verification],
    'evidence': ['CaMRA/navigation/reports/nav065_saved_refit_v2_review.json',
        'CaMRA/navigation/docs/nav065_saved_refit_v2_review_20261118.md',
        '.worklog-testing/NAV-065/saved-refit-v2-review/report.json',
        '.worklog-testing/NAV-065/saved-refit-v2-review/train_cases.json',
        '/pub/data/caohy/work/camra_navigation/nav065_saved_refit_launch_20261118_v1/EXIT.json',
        '/pub/data/caohy/work/camra_navigation/nav065_saved_refit_v2/HELD_TRAIN_GATE.json'],
    'remaining_work': ['后续先评审train门槛的适用性和针对小正reach/预算边界符号的拟合方案；保持原development及零假可行标准，不在本轮自行变更。',
        '重新取得所需更新/协议变更明确授权，并在相关门禁真正通过后再进入NAV-066。'],
    'blockers': ['full s1/s2各300新增更新上限已耗尽，仍4次假可行，不能分配其他候选未用额度或追加训练。',
        'full s2主预算train真实可行17/18，批准train18/18无法由固定actor的critic拟合单独实现；需后续明确评审协议。',
        'train未全通过，按协议不得重评分development；无新G4/部署包，NAV-066依赖未满足。']})
encoded = json.dumps(data, ensure_ascii=False, indent=1).replace('\n', '\r\n') + '\r\n'
PLAN.write_bytes(encoded.encode())
print(rid, 'appended; NAV-065 remains false; completion count=52/67')
