"""Append preparation acceptance, leaving actual diagnostic and NAV-043 pending."""
from datetime import datetime
import difflib
import json
from pathlib import Path

path = Path('doc/plan/20261109_camra_navigation_implementation.json')
old = path.read_text()
plan = json.loads(old)
task = next(t for t in plan['tasks'] if t['id'] == 'NAV-043')
assert len(task['work_records']) == 56 and not task['status']
ev = Path('.worklog-testing/NAV-043/supported-audit-v3')
assert (ev/'verify.exit').read_text().strip() == '0'
support = json.loads((ev/'support.json').read_text())
assert support['new_main_agent_checks'] == 5 and support['actual_parameter_solves'] == 0
record = dict(timestamp=datetime.now().astimezone().isoformat(),
 summary='NAV-043/1.5.0保存复核入口准备收尾：5项基本检查通过，无新拟合，原确认仍待答',
 actions=['新增独立只读审阅器；保存9解法方程/谱、30病例数组/前缀/同折基线/原门槛及启动退出证据核对，未改封存拟合源。',
          '主agent在tiger11/tmux做5项合成基本检查及实际缺批准拒绝；没有测试子agent、真实或合成参数求解。',
          '封存独立PREPARED_ONLY支持清单，更新AGENTS/README/交接/准备文档；原提案及readiness保持，不新建APPROVED。',
          '复查RC-PPO目标/生产45秒链与上下游版本无变；审阅入口不是生产转移模型消费者，不能完成NAV-043。'],
 files_changed=['CaMRA/navigation/scripts/audit_nav043_supported_response.py',
                'CaMRA/navigation/tests/test_supported_response_audit_v3.py',
                str(ev), 'AGENTS.md', 'README.md', 'CaMRA/navigation/README.md',
                'CaMRA/navigation/docs/implementation_handoff.md',
                'CaMRA/navigation/docs/nav043_supported_response_preparation_20261120.md', str(path)],
 commands=['%712: bash .worklog-testing/NAV-043/supported-audit-v3/basic.sh -> exit0, 5 tests',
           '%713: bash .worklog-testing/NAV-043/supported-audit-v3/verify.sh -> exit0'],
 results=['5项基本检查全过，合成/真实响应参数求解0；缺批准实际exit1且未新建报告/锁/输出。',
          '原13项检查来源、提案a3b67ab3/readiness965fbf8e、295/90/29身份及108原文件保持，原HELD保留。',
          '原55条NAV-043记录及67任务定义/状态保持；真实应用/回合/W/网络更新/校准0，NAV-043=false、55/67、G5未过。'],
 evidence=[str(ev/'basic.log'), str(ev/'support.json'), str(ev/'actual-refusal.json')],
 remaining_work=['原30例/最多9次CPU求解/1小时范围仍待确认；批准后先登记/预检/提交，再一次运行原封存入口。',
                 '完整输出后主agent实际执行保存结果审阅，分列执行完整性和诊断；当前只有基本检查，未做全输出验收。',
                 '后续仍需运动支持域/状态、可执行参考程序/J和独立90秒适用性/受影响训练及部署门禁。'],
 blockers=['没有新的明确拟合确认，自动续行和历史物理恢复确认不能转成新拟合授权；不重复索要、不扩展或借旧额度。',
           '尚无真实v3输出；当前准备不解除NAV-043实际导航/适用性与G5缺口。'])
start = old.index('[', old.index('"work_records":', old.index('"id": "NAV-043"')))
_, length = json.JSONDecoder().raw_decode(old[start:])
end = start+length-1
encoded = '\n'.join('        '+s for s in json.dumps(record, ensure_ascii=False, indent=2).splitlines())
new = old[:end].rstrip()+',\n'+encoded+'\n      '+old[end:]
diff = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))
print('*** Begin Patch\n*** Update File: '+str(path))
for line in diff[2:]:
    print('@@' if line.startswith('@@') else line, end='\n' if line.startswith('@@') else '')
print('*** End Patch')
