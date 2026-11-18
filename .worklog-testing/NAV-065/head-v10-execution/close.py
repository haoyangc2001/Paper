"""Append the executed v10 diagnostic outcome without changing old task states."""
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
review_path = ROOT / 'CaMRA/navigation/reports/nav065_head_probe_v10_review.json'
review = json.loads(review_path.read_bytes())
assert review['integrity_pass'] and review['formal_candidate_head_solves'] == 2 and review['audit_head_solves'] == 0
assert review['local_after']['local_pass'] and not review['full_train_after']['pass']
record = {'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 v10已结束：2次输出层求解，25点通过但全量估值严重退化，科学验收仍FAIL',
    'actions': ['按用户确认的独立批准登记执行一次固定full_s0副本诊断：两个原输出层各一次求解，保留全部原始产物和仅供诊断的checkpoint。',
        '由legacy主agent重建162条已存train/59877步，复核25点原标签、实际float32预测、原v9局部和原v5全train生产门禁；审计不再求解。',
        '复核CPU重载、仅四个head张量变化、六原模型/actor和33项v2身份，确认原42项实现散列及准备阶段33基本测试证据有效。',
        '逐项披露局部与全量退化，追加结果文档、机器审阅、README/AGENTS/交接和主计划，保留旧准备及失败记录。'],
    'files_changed': ['CaMRA/navigation/config/nav065_head_probe_prereg_v10.json', 'CaMRA/navigation/config/nav065_head_probe_prereg_v10.json.sha256',
        str(review_path.relative_to(ROOT)), 'CaMRA/navigation/docs/nav065_head_probe_v10_review_20261118.md',
        'CaMRA/navigation/docs/implementation_handoff.md', 'CaMRA/navigation/README.md', 'AGENTS.md', str(PLAN.relative_to(ROOT)), str(HERE.relative_to(ROOT)) + '/'],
    'commands': [sys.executable + ' -B -u ' + str(HERE / 'launch.py'),
        sys.executable + ' -B -u /home/caohy/repositories/Paper/CaMRA/navigation/scripts/probe_nav065_head.py --registration /home/caohy/repositories/Paper/CaMRA/navigation/config/nav065_head_probe_prereg_v10.json',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ' + sys.executable + ' -B -u ' + str(HERE / 'audit.py'),
        'tmux=nav064-r14-repair-20261115:nav065-v10-execution; cwd=/home/caohy/repositories/Paper/CaMRA/navigation; host=tiger11.ios.ac.cn'],
    'results': ['实际正式求解2、复核额外求解0、梯度更新0、新回合0、开发评分0；本次额度已结束，不重启、转移、扫参数或追加。',
        '25个已见train起点局部门禁PASS：成本/到达分量符号错误11→0/9→0，联合符号错误13→0；真实float32最大残差约2.16e-6/1.08e-4，证明这些点在当前固定body上确实可拟合。',
        '全量162起点误放行13→13、误拒绝1→58；原90回合主预算实际可行起点正确识别18→2，旧新增72误放行11→3，所选866沿途边界错判97→439。',
        '原90回合到达相对MAE15.65%→11946.00%，旧新增72回合43.85%→10794.54%；完整train_gate仍FAIL，成本亦退化，不能用局部成功替代全量科学验收。',
        '原六模型/actor及v2完整身份不变，未部署诊断副本；原18/18实际可行成果保留，下降的是副本识别能力。RC-PPO固定actor估值原理、共同停止目标、W/beta、架构/obs-v7及原评分不变。',
        '执行完整性及生产原门禁重载复核通过，准备阶段33基本检查证据与未变实现一致；NAV-065科学验收仍未满足，保持false、52/67、原G4 FAIL，无新G4/部署包或NAV-066执行。'],
    'evidence': [str(review_path.relative_to(ROOT)), 'CaMRA/navigation/docs/nav065_head_probe_v10_review_20261118.md',
        'CaMRA/navigation/config/nav065_head_probe_prereg_v10.json', 'approval_pin=' + review['approval_pin'],
        review['evidence']['output'], str(HERE.relative_to(ROOT)) + '/execution.json', str(HERE.relative_to(ROOT)) + '/audit-attempt1.log',
        'diagnostic_checkpoint_sha256=' + review['checkpoint_sha256']],
    'remaining_work': ['依据现有结果分析原轨迹、临界预算和沿途目标共同拟合的稳定性；25点表达能力已实测，不用本局部通过替代全任务能力或泛化。',
        '后续若改变拟合过程或额度，先提出独立可评审方案；本次不自动扩大到六候选、开发评估、架构修改或NAV-066。'],
    'blockers': ['原完整train及NAV-065科学验收仍失败；本次授权的2次求解已结束，无继续拟合或下游额度。']}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
end = text.index('\n      ],\n   "test_reports": [', start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len('\n      ],\n   "test_reports": [')]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n      ],\n   "test_reports": ['
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + line for line in tail.splitlines()) + '\n' + '\n'.join('+' + line for line in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'closing.patch').open('x') as f:
    f.write(patch)
print(json.dumps({'timestamp': record['timestamp'], 'patch': str(HERE / 'closing.patch'), 'NAV065': False}, ensure_ascii=False))
