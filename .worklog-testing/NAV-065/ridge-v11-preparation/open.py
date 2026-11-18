"""Finish opening after preserving the mixed-newline parsing failure in context.log."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
baseline = json.loads((HERE / 'baseline.json').read_bytes())
assert json.loads(PLAN.read_bytes()) == baseline['plan']
record = {'timestamp': datetime.now().astimezone().isoformat(), 'summary': 'NAV-065/1.3.1 v11开工：按legacy准备全train分层与正则化输出层验证', 'actions': ['重新读取AGENTS、legacy技能及执行/字段规则、README与当前交接，完整解析计划并验证update；按原数组确认NAV-065及NAV-064依赖。', '核验tiger11/tmux、绝对解释器、GPU、Git/子模块和33项冻结v2；沿用同机执行授权。context.py已保存完整现场与计划，但生成追加补丁时未兼容混合换行；保留失败日志，本脚本仅恢复记录追加。'], 'files_changed': ['doc/plan/20261109_camra_navigation_implementation.json', str(HERE.relative_to(ROOT)) + '/'], 'commands': [baseline['python'] + ' -B ' + str(HERE / 'context.py'), baseline['python'] + ' -B ' + str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav065-v11-preparation；cwd=' + str(ROOT)], 'results': ['NAV-065/1.3.1保持false，52/67、原G4 FAIL；目标为RC-PPO Phase II固定actor后的同目标双critic估值，服务下游价值预算搜索。', '只读复核固定v8末次full_s0的162条train/59877步及v10失败；允许隔离实现、基本验证、文档与提案准备，不执行真实候选拟合。', '拟同时使用原/新增全轨迹、全部起点和原沿途边界目标，限制输出层改变量；共同停止、W、beta=0、obs-v7、actor/body/架构、原v5 train/G4和冻结v2保持。', '验收为全量接线、稳定性与批准守卫、生产原门禁调用、主agent基本测试和来源复核；准备不等于科学通过，不启动NAV-066。'], 'evidence': [str(HERE / 'baseline.json'), str(HERE / 'validation-before.log'), 'CaMRA/navigation/reports/nav065_head_probe_v10_review.json'], 'remaining_work': ['完成隔离全量正则求解入口、只读预检、基本测试、具体新额度提案和收尾交接。'], 'blockers': ['v10已结束；真实新候选求解须明确评审新次数/过程，不挪用旧额度。']}
text = PLAN.read_text()
start = text.index('"id": "NAV-065"')
end = text.index('\n      ],\n   "test_reports": [', start)
tail_start = text.rfind('\n  "blockers": [', start, end) + 1
tail = text[tail_start:end + len('\n      ],\n   "test_reports": [')]
new = tail[:tail.rfind('\n}')] + '\n},\n' + json.dumps(record, ensure_ascii=False, indent=2) + '\n      ],\n   "test_reports": ['
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n@@\n' + '\n'.join('-' + s for s in tail.splitlines()) + '\n' + '\n'.join('+' + s for s in new.splitlines()) + '\n*** End Patch\n'
with (HERE / 'opening.patch').open('x') as f:
    f.write(patch)
print('Opening patch ready; scientific task remains false.')
