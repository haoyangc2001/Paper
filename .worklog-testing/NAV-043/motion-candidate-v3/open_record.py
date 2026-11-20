"""Produce an append-only plan patch; no experiment."""
import difflib
import json
from datetime import datetime
from pathlib import Path

p = Path('doc/plan/20261109_camra_navigation_implementation.json')
old = p.read_text()
j = json.loads(old)
t = next(t for t in j['tasks'] if t['id'] == 'NAV-043')
assert len(t['work_records']) == 53
t['work_records'].append({
    'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-043/1.5.0续行开工：根据12完整工况与18旧导航准备隔离运动候选，未拟合',
    'actions': ['重新读取AGENTS、legacy规则、计划全文/任务及依赖、导航README/交接；%704预检通过。',
                '为RC-PPO确定性名义转移准备运动响应候选；仅新增隔离实现/PROPOSED协议和主agent基本检查，无生产消费者。',
                '固定v2/功率/足迹/alpha/delta/45秒生产链及全部失败保持，原HELD不补造、不重采；90秒/obs-v9仍候选。'],
    'files_changed': [str(p)],
    'commands': ['tiger11/tmux %704: bash .worklog-testing/NAV-043/motion-candidate-v3/preflight.sh -> exit0'],
    'results': ['NAV-066/067依赖结案；NAV-043首个可执行且false，55/67历史true和53条旧记录保持。',
                '295保护身份/90输入/29冻结包及108采集原文件保持，原物理进程消失，严格HELD保留。',
                '新真实拟合/应用/业务回合/W/网络更新0，不安排测试子agent。'],
    'evidence': ['.worklog-testing/NAV-043/motion-candidate-v3/preflight.json',
                 'CaMRA/navigation/reports/nav043_motion_probe_recovery_held_review_v2.json'],
    'remaining_work': ['先做无求解的数据支持/秩检查，再封存低速/零命令/q记忆候选与有限诊断。',
                       '新拟合/物理/训练需具体范围及独立批准；参考程序/J及90秒残余验证仍待后续。'],
    'blockers': ['NAV-043四项实际导航验收和G5未过，本次仅准备不能放行。']})
start = old.index('[', old.index('"work_records":', old.index('"id": "NAV-043"')))
_, size = json.JSONDecoder().raw_decode(old[start:])
end = start + size - 1
record = '\n'.join('        ' + s for s in json.dumps(t['work_records'][-1], ensure_ascii=False, indent=2).splitlines())
new = old[:end].rstrip() + ',\n' + record + '\n      ' + old[end:]
d = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))
print('*** Begin Patch\n*** Update File: ' + str(p))
print(''.join(d[2:]), end='')
print('*** End Patch')
