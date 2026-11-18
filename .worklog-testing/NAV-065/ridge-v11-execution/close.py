"""Generate append-only result documentation and one closing plan record."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
review_path = NAV / 'reports/nav065_ridge_probe_v11_review.json'
r = json.loads(review_path.read_bytes())
assert r['integrity_pass'] and r['formal_candidate_head_solves'] == 2 and r['audit_head_solves'] == 0
assert not r['full_train_after']['pass']
pin = r['approval_pin']
doc = '''# NAV-065 v11结果：全量拟合更稳定，关键误放行未解决

用户确认具体v11提案后，2026-11-18于tiger11/tmux完成一次full_s0副本验证；按create-plan-worklog-legacy由主agent验收，无测试子agent。两个原输出层各求解一次，共2次；梯度更新、新回合、开发集评分均0。原G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持。

## 实际执行与来源

固定v8末次full_s0来源，复用其原90条及已有新增72条train，共162条/59877步。原/新增全轨迹、162起点及原866个边界行四组等质量并在组内按到达目标正负分层，共60905个加权行出现。原数值目标、actor、两个critic body、架构、obs-v7、共同停止、W/beta=0和原v5 train/G4不变。

按[已批准登记](../config/nav065_ridge_probe_prereg_v11.json)执行固定0.01正则、rcond=1e-12的两个增广float64最小二乘，转回原float32网络。批准pin为`PIN_VALUE`，执行源码Git为`8b3a98988aad35b412e738106f809043b3ff5e16`，原未批准提案保留为历史。实际执行21:27:28–21:28:00，未重试、扫参数或追加。

## 全量结果

以下全部是固定full_s0的已存train诊断，不是新development、物理执行或部署结果。

| 指标 | 固定v8源 | v11副本 | 含义 |
| --- | ---: | ---: | --- |
| 原90条主预算k=1.15实际安全可行 | 18/18 | 18/18 | actor与真实轨迹未变 |
| 同18个起点正确识别可行 | 18/18 | 18/18 | 避免了v10降至2/18的大退化 |
| 全162起点错误放行 | 13 | 13 | 关键风险未修好 |
| 全162起点错误拒绝 | 1 | 1 | 联合起点混淆计数未变 |
| 原90条全时刻到达相对MAE | 15.65% | 15.00% | 小幅改善 |
| 已有新增72条到达相对MAE | 43.85% | 41.31% | 仍高于原35%线 |
| 新增到达Pearson | 0.5722 | 0.6129 | 仍低于原0.70线 |
| 原866边界行联合符号错误 | 97 | 94 | 小幅改善，远未归零 |
| 原90条全时刻联合W相对MAE | 6.46% | 20.96% | 诊断指标变差，不能只报改善 |
| 全162起点到达分量符号错误 | 28 | 34 | 独立到达分量退化 |
| 原866边界行到达分量符号错误 | 264 | 290 | 联合略好不表示分量全好 |

完整原v5 train_gate仍FAIL，失败类别与源一致：原预算起点误放行、新增到达相对误差/Pearson不达线、新增起点误放行和所选边界符号错误。原25点局部门禁也FAIL，联合错误13→13，到达相对MAE69.86%→64.19%；没有使用这些点单独优化。

原轨迹全时刻联合错误604→503，新增全时刻4346→4076；但新增全时刻错误拒绝28→31。行数是沿途观测数，不能当作独立回合或起点成功率的分母。[机器审阅](../reports/nav065_ridge_probe_v11_review.json)保留全部组、逐起点及分量结果。

## 数值与完整性复核

实际成本/到达输出层改变量范数约0.002892/0.937267，均在源目标比较界0.0529575/5.4404447以内；对应整体正则目标从0.000028045→0.000027693、0.168966→0.156158。两个增广设计均秩257，转回float32后的驻点残差约8.6e-9/1.23e-8。该检查未再求解候选，支持本次固定二次问题的求解没有明显数值异常；不保证任何单个状态或符号正确。

主agent从原保存轨迹重新构建全部标签/权重，重跑原v5完整train_gate、原25点检查及全部起点/组统计，CPU重载确认仅四个head张量改变。原六模型、六actor、源body、33项冻结v2身份和全部旧结果保持。38项基本测试与本次47项绑定源码散列一致，未无故重跑或派测试子agent；审计额外求解0。

原始产物`/pub/data/caohy/work/camra_navigation/nav065_ridge_probe_v11/`，诊断模型`diagnostic-ridge.pt`；启动、批准、复核日志在`.worklog-testing/NAV-065/ridge-v11-execution/`，tmux为`nav064-r14-repair-20261115:nav065-v11-execution`，绝对解释器`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。执行退出码0只表示入口完成；原门禁实际FAIL。

## 结论与边界

全量共同拟合和限制改变量避免了v10的大幅过拟合，但本次设置只带来小幅数值改善，关键13次误放行未解决，部分指标仍退化。RC-PPO预算根搜索依赖W正负；整体回归目标下降不足以交付可靠的预算判断。

不能由这一固定特征、固定权重和固定正则的失败断言架构或RC-PPO无解，也不能直接证明需要更多采样或更大网络。本次线性子问题已求解，后续应先分析现有样本的特征、符号边界与拟合折中，再评审有区分度的新方案；不以继续增加相同过程次数代替诊断。

本次2次额度已结束，不追加求解、梯度更新、新采样或开发评分，不换架构，不扩展六候选、不生成新G4或部署包。NAV-065保持false；NAV-066继续等待有效上游。准备说明和旧轮结果保留为历史时点。
'''.replace('PIN_VALUE', pin)
short = '用户确认v11具体提案后在tiger11/tmux完成一次full_s0副本验证，两原输出层各求解一次/共2次，梯度更新/新回合/开发评分0。复用162条train/59877步，原主预算实际可行及识别18/18保留；起点误放行13→13、误拒绝1→1，新增到达相对MAE43.85%→41.31%、Pearson0.5722→0.6129，边界联合错误97→94。原全时刻W相对MAE6.46%→20.96%、起点到达符号错误28→34，仍有退化，完整train FAIL。legacy主agent原生产门禁/CPU重载/仅四head变化/六原模型及33项v2复核通过，38基本测试与未变源码匹配，无测试子agent。原G4 FAIL、52/67、NAV-065=false及NAV-066未启动；2次额度已结束，不追加或自动扩展。'
handoff = '\n## 2026-11-18 NAV-065 v11结束：稳住全量，关键误判仍未减少\n\n' + short + '\n\n详见[本轮结果](nav065_ridge_probe_v11_review_20261118.md)、[机器审阅](../reports/nav065_ridge_probe_v11_review.json)及[已批准登记](../config/nav065_ridge_probe_prereg_v11.json)，pin为`' + pin + '`。原始产物`/pub/data/caohy/work/camra_navigation/nav065_ridge_probe_v11/`；tmux `nav064-r14-repair-20261115:nav065-v11-execution`，日志`.worklog-testing/NAV-065/ridge-v11-execution/`。执行完整性通过不等于科学通过；下一步先诊断已有train特征/符号/拟合折中，不自动开始新拟合。下方准备和未批准措辞保留为历史时点。\n'
agent = '\n### NAV-065最新补充（2026-11-18，全量正则输出层v11已结束）\n\n' + short + '见[本轮结果](CaMRA/navigation/docs/nav065_ridge_probe_v11_review_20261118.md)。上方v11未批准为历史时点，本次确认只授权已结束的一个独立尝试。\n'
readme = '\n当前补充（2026-11-18）：NAV-065 v11已完成两次批准输出层求解；原主预算18/18保留，162起点误放行仍13、误拒绝仍1，新增到达相对MAE43.85%→41.31%，部分W/分量符号指标退化，完整train仍FAIL。主agent复核通过不等于科学通过；52/67、NAV-065=false、原G4 FAIL、NAV-066未启动，不自动追加。见[本轮结果](docs/nav065_ridge_probe_v11_review_20261118.md)。下方准备和旧轮状态保留为历史。\n'
record = {'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 v11已结束：2次全量正则输出层求解，18/18识别保留但13次误放行未减，train仍FAIL',
    'actions': ['按用户确认的独立批准登记执行一次固定full_s0副本全量正则拟合，两原输出层各一次求解，保留全部产物及诊断checkpoint。',
        'legacy主agent重新构建162条已存train/59877步及60905个加权行出现，逐项复核标签/权重散列、原v5完整train_gate、原25点、全部起点/组和float32目标/驻点残差；审计禁止额外求解、优化及开发评分。',
        '复核CPU重载、仅四head变化、六原模型/actor及33项冻结v2，38基本测试证据与47项批准源码绑定一致；无专门测试子agent。',
        '披露改善和退化，生成结果说明与机器审阅，同步AGENTS/README/交接并仅追加当前NAV-065记录；原准备和失败证据保留。'],
    'files_changed': ['CaMRA/navigation/config/nav065_ridge_probe_prereg_v11.json', 'CaMRA/navigation/config/nav065_ridge_probe_prereg_v11.json.sha256', str(review_path.relative_to(ROOT)),
        'CaMRA/navigation/docs/nav065_ridge_probe_v11_review_20261118.md', 'CaMRA/navigation/docs/implementation_handoff.md', 'CaMRA/navigation/README.md', 'AGENTS.md', str(PLAN.relative_to(ROOT)), str(HERE.relative_to(ROOT)) + '/'],
    'commands': [sys.executable + ' -B -u ' + str(HERE / 'launch.py'),
        sys.executable + ' -B -u /home/caohy/repositories/Paper/CaMRA/navigation/scripts/probe_nav065_ridge.py --registration /home/caohy/repositories/Paper/CaMRA/navigation/config/nav065_ridge_probe_prereg_v11.json',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ' + sys.executable + ' -B -u ' + str(HERE / 'audit.py'),
        'tmux=nav064-r14-repair-20261115:nav065-v11-execution; cwd=/home/caohy/repositories/Paper/CaMRA/navigation; host=tiger11.ios.ac.cn'],
    'results': ['真实求解2、审计额外求解0、梯度更新0、新回合0、开发评分0；本次额度已结束，不重试、扫描、追加或自动扩展。',
        '主预算原实际可行及识别18/18保留，全162起点误放行13→13、误拒绝1→1；原25点联合错误13→13，局部仍FAIL。',
        '原/新增全时刻到达相对MAE15.65%→15.00%/43.85%→41.31%，新增Pearson0.5722→0.6129，所选866边界联合错误97→94；完整train_gate仍FAIL。',
        '退化保留：原全时刻W相对MAE6.46%→20.96%，全起点到达分量符号错误28→34、边界分量264→290；整体目标改善不保证符号正确。',
        '实际成本/到达输出层改变量约0.002892/0.937267，float32驻点残差约8.6e-9/1.23e-8；仅为该固定特征/权重/正则的证据，不证明架构不可行。',
        '原RC-PPO原理/共同停止/W/beta、actor/body/架构/obs-v7、六原模型、v2/校准和评分不变；未部署副本。执行完整性通过，科学验收未满足，NAV-065=false、52/67、原G4 FAIL及NAV-066未启动。'],
    'evidence': [str(review_path.relative_to(ROOT)), 'CaMRA/navigation/docs/nav065_ridge_probe_v11_review_20261118.md',
        'CaMRA/navigation/config/nav065_ridge_probe_prereg_v11.json', 'approval_pin=' + pin, r['evidence']['output'],
        str(HERE.relative_to(ROOT)) + '/execution.json', str(HERE.relative_to(ROOT)) + '/audit-attempt1.log', 'diagnostic_checkpoint_sha256=' + r['checkpoint_sha256']],
    'remaining_work': ['先分析已有train的特征、符号边界及拟合折中，提出能区分原因的具体方案；当前固定二次问题已求解，不以盲目追加同类次数替代诊断。',
        '后续新拟合、改过程/架构、开发评分或六候选扩展须明确评审具体方案；NAV-066继续等待有效上游。'],
    'blockers': ['原完整train及NAV-065科学验收仍失败；本次2次拟合额度已结束，无继续拟合或下游授权。']}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
sentinel = '\n      ],\n   "test_reports": ['
end = text.index(sentinel, start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len(sentinel)]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + sentinel
patch = '*** Begin Patch\n*** Add File: ' + str(NAV / 'docs/nav065_ridge_probe_v11_review_20261118.md') + '\n' + '\n'.join('+' + line for line in doc.splitlines()) + '\n'
for relative, addition, marker in (('AGENTS.md', agent, '## 前置状态：v2 选择与历史校准复核（2026-11-09）'),
        ('CaMRA/navigation/docs/implementation_handoff.md', handoff, None), ('CaMRA/navigation/README.md', readme, None)):
    path = ROOT / relative
    content = path.read_text()
    anchor = marker or content.splitlines()[0]
    replacement = addition.lstrip('\n') + '\n' + anchor if marker else anchor + '\n' + addition.rstrip('\n')
    patch += '*** Update File: ' + str(path) + '\n@@\n-' + anchor + '\n' + '\n'.join('+' + line for line in replacement.splitlines()) + '\n'
patch += '*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + line for line in tail.splitlines()) + '\n' + '\n'.join('+' + line for line in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'closing.patch').open('x') as f:
    f.write(patch)
print(json.dumps({'timestamp': record['timestamp'], 'patch': str(HERE / 'closing.patch'), 'NAV065': False}, ensure_ascii=False))
