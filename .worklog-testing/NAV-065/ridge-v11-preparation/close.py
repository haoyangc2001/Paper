"""Generate one append-only closing patch after main-agent legacy verification."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
baseline = json.loads((HERE / 'baseline.json').read_bytes())
plan = json.loads(PLAN.read_bytes())
review = json.loads((ROOT / 'CaMRA/navigation/reports/nav065_ridge_probe_v11_preparation_review.json').read_bytes())
assert review['preparation_pass'] and review['formal_head_solves'] == 0
old = next(t for t in baseline['plan']['tasks'] if t['id'] == 'NAV-065')
current = next(t for t in plan['tasks'] if t['id'] == 'NAV-065')
assert not current['status'] and len(current['work_records']) == len(old['work_records']) + 1
assert current['work_records'][:-1] == old['work_records']
record = {
    'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 v11准备收尾：全train正则输出层入口、38基本测试和只读预检通过，真实拟合待2次新额度',
    'actions': [
        '实现隔离probe_nav065_ridge.py及未批准v11具体提案；固定v8末次full_s0，原/已有新增全轨迹、全部起点及原v5边界行共同加权，原数值标签保持。',
        '每个原输出层仅拟用一次增广float64最小二乘，对相对源改变量加入固定0.01正则并转回float32；实现独立批准/hash守卫、CPU重载、四head张量限制、原v5生产train_gate和逐组/原25点退化报告。',
        '主agent运行38基本测试；真实只读包装器禁用候选求解、Adam/SGD、rollout和开发评分，完成162条train/59877步预检。未派专门测试子agent。',
        '主agent复核完整计划/依赖、原门禁一致、v10原始产物、固定来源及六原模型和33项v2散列；同步AGENTS、导航README和交接。开工辅助脚本混合换行失败原样留证，独立open.py仅恢复追加。'
    ],
    'files_changed': [
        'CaMRA/navigation/scripts/probe_nav065_ridge.py',
        'CaMRA/navigation/tests/test_nav065_ridge_probe.py',
        'CaMRA/navigation/config/nav065_ridge_probe_proposal_v11.json',
        'CaMRA/navigation/docs/nav065_ridge_probe_v11_20261118.md',
        'CaMRA/navigation/reports/nav065_ridge_preflight_v11.json',
        'CaMRA/navigation/reports/nav065_ridge_probe_v11_preparation_review.json',
        'AGENTS.md', 'CaMRA/navigation/README.md',
        'CaMRA/navigation/docs/implementation_handoff.md',
        'doc/plan/20261109_camra_navigation_implementation.json',
        '.worklog-testing/NAV-065/ridge-v11-preparation/'
    ],
    'commands': [
        'tmux=nav064-r14-repair-20261115:nav065-v11-preparation; host=tiger11.ios.ac.cn; cwd=/home/caohy/repositories/Paper/CaMRA/navigation',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -m pytest tests/test_nav065_ridge_probe.py tests/test_nav065_head_probe.py -q -p no:cacheprovider',
        '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B .worklog-testing/NAV-065/ridge-v11-preparation/preflight.py',
        '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B .worklog-testing/NAV-065/ridge-v11-preparation/check.py'
    ],
    'results': [
        '38 passed；真实只读接线162条/59877步，四组33223/26654/162/866行、各质量1/4，合计60905个加权行出现；未新增回合或丢弃沿途状态。',
        '成本/到达的条件改变量界约0.0529575/5.4404447；该界仅限制整体正则目标，不能保证每组误差、符号或泛化，不是放行标准。',
        '原全量train_gate与v9源预检完全一致，源162起点误放行13、误拒绝1保留。真实候选求解/梯度更新/新回合/开发评分均0，未生成v11正式run或批准登记。',
        'RC-PPO共同停止/双critic数值目标/W/beta=0及实际原生产门禁未变；actor/body/架构、obs-v7、六原模型、冻结v2与校准及只读参考/论文未改。',
        '本轮准备完整性通过不等于科学通过；原G4 FAIL、52/67、NAV-065=false、原verification_result及NAV-066未启动保持。'
    ],
    'evidence': [
        'CaMRA/navigation/reports/nav065_ridge_probe_v11_preparation_review.json',
        'CaMRA/navigation/reports/nav065_ridge_preflight_v11.json',
        '.worklog-testing/NAV-065/ridge-v11-preparation/basic-attempt1.log',
        '.worklog-testing/NAV-065/ridge-v11-preparation/preflight-attempt1.log',
        '.worklog-testing/NAV-065/ridge-v11-preparation/check-attempt1.log',
        '.worklog-testing/NAV-065/ridge-v11-preparation/context.log'
    ],
    'remaining_work': [
        '评审已落地v11提案：一个full_s0副本、两个原输出层各一次新求解，总2次；梯度更新/新回合/开发评分0，不扫参数、不自动扩展。',
        '获得具体新额度后新建独立批准登记/hash，再做该次小验证及完整train影响审阅；不能把准备或train通过当NAV-065/G4通过。'
    ],
    'blockers': [
        '按主计划uncertainty_policy第7项及legacy执行规则，新真实模型拟合次数/过程待明确确认；v10已结束，不挪用旧额度或进入NAV-066。'
    ]
}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
sentinel = '\n      ],\n   "test_reports": ['
end = text.index(sentinel, start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len(sentinel)]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + sentinel
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + s for s in tail.splitlines()) + '\n' + '\n'.join('+' + s for s in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'closing.patch').open('x') as f:
    f.write(patch)
print('Closing patch generated; no task status or prior record is changed.')
