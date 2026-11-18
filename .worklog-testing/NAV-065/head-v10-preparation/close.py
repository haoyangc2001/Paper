"""Generate an append-only NAV-065 v10 preparation closing patch."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
plan = json.loads(PLAN.read_bytes())
review_path = ROOT / 'CaMRA/navigation/reports/nav065_head_probe_v10_preparation_review.json'
review = json.loads(review_path.read_bytes())
assert review['preparation_pass'] and review['basic_tests_passed'] == 33
assert not review['scientific_pass'] and not review['proposal_approved']
record = {
    'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 v10准备收尾：末层数值条件已诊断，33基本检查通过，2次真实求解提案待批准',
    'actions': [
        '固定v8末次full_s0和既有162条train/59877步，重建原完整train_gate和v9同一25起点；只读核验v9全部400试步/401评估及原始hash。',
        '计算v8/v9两个critic末层特征的SVD秩、目标投影残差和精确插值的最小权重改变量范数；没有构造或赋值真实候选求解权重。',
        '新增隔离末层求解入口、未批准提案及基本测试；保留原v9局部检查与原v5完整90+72 train_gate的生产调用链，绑定批准/实现hash并保护原模型。',
        '主agent验证33项基本检查、只读预检、原六模型和v2身份；同步README、AGENTS、交接及详细方案，保留首次两个合成替身接口失败和全部历史。'
    ],
    'files_changed': [
        'CaMRA/navigation/scripts/probe_nav065_head.py',
        'CaMRA/navigation/tests/test_nav065_head_probe.py',
        'CaMRA/navigation/config/nav065_head_probe_proposal_v10.json',
        'CaMRA/navigation/reports/nav065_head_preflight_v10.json',
        'CaMRA/navigation/reports/nav065_head_probe_v10_preparation_review.json',
        'CaMRA/navigation/docs/nav065_head_probe_v10_20261118.md',
        'CaMRA/navigation/docs/implementation_handoff.md',
        'CaMRA/navigation/README.md', 'AGENTS.md',
        'doc/plan/20261109_camra_navigation_implementation.json',
        '.worklog-testing/NAV-065/head-v10-preparation/'
    ],
    'commands': [
        'tmux=nav064-r14-repair-20261115:nav065-v10-preparation; cwd=/home/caohy/repositories/Paper/CaMRA/navigation; host=tiger11.ios.ac.cn',
        sys.executable + ' -B -u scripts/probe_nav065_head.py --diagnose reports/nav065_head_preflight_v10.json',
        sys.executable + ' -B -m pytest -q tests/test_nav065_head_probe.py tests/test_nav065_local_probe.py',
        sys.executable + ' -B -u ' + str(HERE / 'check.py'),
        sys.executable + ' -B -u ' + str(Path(__file__)),
        sys.executable + ' -B -u ' + str(HERE / 'verify-final.py')
    ],
    'results': [
        '准备复核preparation_pass=true、基本检查33通过；真实候选求解/优化更新/新回合/开发评分均0，未创建正式run或批准登记。',
        'v8/v9两个末层特征在rcond=1e-12下均行秩25，目标投影残差近零；v8成本/到达精确插值的最小改变量范数42.47/552.04，当前末层范数0.330/1.324。成本常见float32数值秩18，数值条件敏感。',
        '上述仅为25点精确插值条件，不是达到原误差线所需范数，不证明float32稳定拟合、全任务表达充分、泛化、唯一根因或必须换架构。',
        '新提案仅一个full_s0副本、两个原输出层各一次线性求解/总2次，固定actor/body；优化0步/新采样0/开发评分0、不扫参数、不重试、不自动扩展。真实求解仍是新拟合，须新批准。',
        'RC-PPO固定actor后双critic估值原理及原共同停止标签/W/beta/评分保持，生产原门禁直接调用已接通并经合成小网络验证；真实候选生产前向与全train效果尚待批准后执行。',
        '六原模型/actor、33项v2和旧实现/登记hash保持；新诊断代码未进入部署包，NAV-065=false、52/67、原G4 FAIL及NAV-066未启动保持。'
    ],
    'evidence': [
        'CaMRA/navigation/reports/nav065_head_preflight_v10.json',
        str(review_path.relative_to(ROOT)),
        'CaMRA/navigation/docs/nav065_head_probe_v10_20261118.md',
        'CaMRA/navigation/config/nav065_head_probe_proposal_v10.json',
        str(HERE.relative_to(ROOT)) + '/basic-attempt1.log',
        str(HERE.relative_to(ROOT)) + '/basic-attempt2.log',
        str(HERE.relative_to(ROOT)) + '/check-attempt1.log'
    ],
    'remaining_work': [
        '评审并明确批准独立v10两次输出层求解额度，或暂缓；批准后才冻结登记并核验资源实际执行一次。',
        '批准后的诊断须同时披露原局部与全train影响；局部通过也不等于NAV-065科学通过，不能提前进入NAV-066或部署。'
    ],
    'blockers': [
        'v9单次400更新已结束，新的真实输出层求解仍是模型拟合；主计划uncertainty_policy的正式试验/资源额度变化尚待明确确认，不重用旧额度。',
        'NAV-065原科学验收缺口未关，真实float32求解及全量影响尚未验证；原G4 FAIL保留。'
    ]
}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
end = text.index('\n      ],\n   "test_reports": [', start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len('\n      ],\n   "test_reports": [')]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n      ],\n   "test_reports": ['
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + s for s in tail.splitlines()) + '\n' + '\n'.join('+' + s for s in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'closing.patch').open('x') as f:
    f.write(patch)
print(json.dumps({'timestamp': record['timestamp'], 'patch': str(HERE / 'closing.patch'), 'NAV065': False}, ensure_ascii=False))
