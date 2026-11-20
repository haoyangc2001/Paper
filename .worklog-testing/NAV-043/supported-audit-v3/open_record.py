"""Append-only NAV-043 opening alignment patch."""
import difflib
import json
from datetime import datetime
from pathlib import Path

p=Path('doc/plan/20261109_camra_navigation_implementation.json')
old=p.read_text(); j=json.loads(old)
t=next(t for t in j['tasks'] if t['id']=='NAV-043')
assert len(t['work_records'])==55
r=dict(timestamp=datetime.now().astimezone().isoformat(),
 summary='NAV-043/1.5.0只读审阅准备开工：原CPU拟合问题仍待确认，不启动拟合，补保存输出复核',
 actions=['重新读取根AGENTS、legacy规则、当前完整计划/任务及依赖、导航README/交接和封存v3源码；%711预检通过。',
          '上轮已提交8fbaad4，提案a3b67ab3/readiness965fbf8e保持；本轮只新增独立保存输出审阅器与主agent少量合成反例，不改封存拟合源。',
          '本项为后续有限诊断提供独立参数/轨迹/门槛复核，原理和生产45秒链不改；不把自动续行视为新科学批准。'],
 files_changed=[str(p)],
 commands=['tiger11/tmux %711: bash .worklog-testing/NAV-043/supported-audit-v3/preflight.sh -> exit0, legacy VALID'],
 results=['NAV-043为首个依赖满足的false任务，55/67历史true、55条原记录保持。','295/90/29身份与封存来源保持，APPROVED/正式输出/launch.lock不存在，新应用/回合/真实拟合/W/更新0。'],
 evidence=['.worklog-testing/NAV-043/supported-audit-v3/preflight.json',
           '.worklog-testing/NAV-043/supported-audit-v3/preflight-plan.log',
           'CaMRA/navigation/config/nav043_supported_response_proposal_v3.json'],
 remaining_work=['新增只消费完整保存结果的审阅入口，核对9保存解、30自由递推/前缀/门禁及运行账本；以少量合成反例验证不额外拟合。',
                 '原9次CPU求解/1小时的确认问题保持待答，不重复索要，不扩展方案或启动实际拟合。'],
 blockers=['本轮无真实v3拟合输出可审；审阅准备不能解除NAV-043真实闭环/适用性/G5缺口。'])
start=old.index('[',old.index('"work_records":',old.index('"id": "NAV-043"')))
_,size=json.JSONDecoder().raw_decode(old[start:]);end=start+size-1
record='\n'.join('        '+s for s in json.dumps(r,ensure_ascii=False,indent=2).splitlines())
new=old[:end].rstrip()+',\n'+record+'\n      '+old[end:]
d=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
print('*** Begin Patch\n*** Update File: '+str(p))
for line in d[2:]:print('@@' if line.startswith('@@') else line,end='\n' if line.startswith('@@') else '')
print('*** End Patch')
