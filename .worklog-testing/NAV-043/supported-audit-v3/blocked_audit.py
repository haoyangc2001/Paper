"""Revalidate the same missing scope confirmation; do not fit or launch."""
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
EV = ROOT/'.worklog-testing/NAV-043/supported-audit-v3'
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c
import audit_nav043_supported_response as audit

c.context()
plan_path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
old = plan_path.read_text()
plan = json.loads(old)
by = {task['id']: task for task in plan['tasks']}
eligible = [task['id'] for task in plan['tasks'] if not task['status']
            and all(by[dep]['status'] for dep in task['depends_on'])]
assert eligible == ['NAV-043'] and sum(task['status'] for task in plan['tasks']) == 55
task = by['NAV-043']
assert len(task['work_records']) == 57 and not task['status']
p, ready = c.pin(c.PROPOSAL), c.pin(c.READINESS)
support = c.pin(EV/'support.json')
for group in (p['source_hashes'], ready['source_hashes'], support['source_hashes']):
    for path, digest in group.items():
        assert c.base.sha(path) == digest, path
assert not c.APPROVAL.exists() and not (c.HERE/'launch.lock').exists()
assert not Path(p['output_root']).exists() and not audit.REPORT.exists()
processes = subprocess.check_output(['ps', '-eo', 'pid,comm,args'], text=True)
live = []
for line in processes.splitlines()[1:]:
    columns = line.split(None, 2)
    if len(columns) != 3 or not columns[1].startswith('python'):
        continue
    if ('launch_nav043_supported_response_once.py' in columns[2]
            or 'import nav043_supported_response as c; c.run(' in columns[2]):
        live.append(line)
assert not live, live
at = datetime.now().astimezone().isoformat()
report = dict(status='SAME_SCOPE_CONFIRMATION_BLOCKER_REVALIDATED', at=at,
    pane=os.environ['TMUX_PANE'], proposal_sha256=c.base.sha(c.PROPOSAL),
    readiness_sha256=c.base.sha(c.READINESS), support_sha256=c.base.sha(EV/'support.json'),
    previous_turn_classification='progress: saved-output auditor implemented, checked and committed',
    repeated_condition='new finite response-v3 fit scope has no explicit confirmation',
    observations=['response-v3 proposal/preparation ended waiting for confirmation',
                  'saved-output auditor preparation ended waiting for the same confirmation',
                  'current automatic continuation revalidated unchanged missing confirmation'],
    approval_exists=False, fitting_processes=live, formal_output_exists=False,
    once_lock_exists=False, review_exists=False, plan_true=55, NAV043_passed=False,
    ready_to_resume_scope=p['limits'], new_fits=0, new_applications=0,
    new_episodes=0, W_queries=0, actor_critic_updates=0,
    remaining_authorized_preparation='none necessary before the pending fixed diagnostic',
    next_action='mark persistent goal blocked; resume only after scope confirmation or other user decision')
c.base.write_new(EV/'blocked-audit.json', report)
record = dict(timestamp=at,
 summary='NAV-043/1.5.0同一新拟合确认阻塞复核：准备已提交，无在跑任务，等待明确范围决定',
 actions=['重新读取AGENTS、legacy规则及完整计划，核对NAV-043依赖、验收和全部历史；上一目标轮为实际准备进展。',
          '复核原提案/readiness/新增审阅支持清单来源，独立批准/一次锁/正式输出/结果报告仍不存在，实际拟合进程0。',
          '运动候选准备、保存复核准备及本次自动续行均遇到同一新范围确认缺口；必要准备已齐，不再制造重复测试或新提案。',
          '仅登记阻塞和本地提交；科学任务保持false，持久目标按连续阻塞规则登记blocked，后续由用户确认原精确范围或改变路线后恢复。'],
 files_changed=[str(plan_path.relative_to(ROOT)), str(EV.relative_to(ROOT)/'blocked-audit.json'),
                str(EV.relative_to(ROOT)/'blocked_audit.py'), str(EV.relative_to(ROOT)/'blocked_audit.sh')],
 commands=['tiger11/tmux: bash .worklog-testing/NAV-043/supported-audit-v3/blocked_audit.sh'],
 results=['仅审计同一未决授权，新拟合/应用/回合/W/actor或critic更新0；没有可继续轮询的真实拟合进程。',
          '原模型/启动器/提案/readiness/审阅支持来源未变；NAV-043=false、55/67、G5未过，原失败和45秒/v2/alpha/delta保持。'],
 evidence=[str(EV.relative_to(ROOT)/'blocked-audit.json'),
           'CaMRA/navigation/config/nav043_supported_response_proposal_v3.json',
           str(EV.relative_to(ROOT)/'close.json')],
 remaining_work=['等待原30已暴露病例、最多9次CPU求解/1小时范围的明确确认；不重新询问同一问题。',
                 '确认后才独立登记APPROVED、预检/提交并一次运行原封存启动器，随后主agent只读审阅实际结果。',
                 '未解决的模型支持域/参考命令J/90秒适用性及受影响学习部署门禁仍按依赖推进，不能据准备通过结案。'],
 blockers=['同一真实阻塞持续：历史物理恢复确认未包含新运动响应拟合，自动续行不是用户答复；无可用输出支持后续验收。'])
start = old.index('[', old.index('"work_records":', old.index('"id": "NAV-043"')))
_, length = json.JSONDecoder().raw_decode(old[start:])
end = start+length-1
encoded = '\n'.join('        '+line for line in json.dumps(record, ensure_ascii=False, indent=2).splitlines())
new = old[:end].rstrip()+',\n'+encoded+'\n      '+old[end:]
diff = list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))
print('*** Begin Patch\n*** Update File: '+str(plan_path))
for line in diff[2:]:
    print('@@' if line.startswith('@@') else line, end='\n' if line.startswith('@@') else '')
print('*** End Patch')
