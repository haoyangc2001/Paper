"""Read the complete plan, validate, and append this preparation's opening."""
from datetime import datetime
import difflib
import json
import os
import re
import textwrap
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
VALIDATOR = '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)
original = PLAN.read_text()
p = json.loads(original)
states = {t['id']: t['status'] for t in p['tasks']}
current = next(t for t in p['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
assert current['id'] == 'NAV-065' and sum(states.values()) == 52
assert not any('v6准备开工' in r['summary'] for r in current['work_records'])
current['work_records'].append({
    'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 v6准备开工：现有train诊断、分层回归和小段更新入口；无正式拟合',
    'actions': ['重读根AGENTS、legacy技能/执行规则/字段规范及计划全文，核对首个可执行任务NAV-065、NAV-064=true及最新v5失败记录。'],
    'files_changed': ['doc/plan/20261109_camra_navigation_implementation.json', '.worklog-testing/NAV-065/block-v6-preparation/preflight.py'],
    'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav065-v6-prepare；cwd=CaMRA/navigation；Python=' + sys.executable, 'legacy validate_worklog.py --mode update'],
    'results': ['本项为NAV-066提供可信同actor双critic；消费v5六份最终模型及972条train/372471步，只读诊断和隔离实现获授权。', '保持actor、共同停止目标、W、beta=0、原G4/v5最终train标准、v2及匹配校准、地图/单位/分片；不访问final或budget_selection。', '允许修改导航新脚本/基本测试/未批准v6提案及文档；主agent验证，不派测试子agent。正式新更新/开发重评须先形成具体方案和明确额度，不挪用旧额度。', '验收基本测试、真实train预检、来源/接口及原目标一致性；本轮不标记NAV-065完成、不启动NAV-066。'],
    'evidence': ['用户最新要求：按create-plan-worklog-legacy基于前述分析开始推进', 'CaMRA/navigation/reports/nav065_adaptive_refit_v5_review.json', 'git d9f8bb6'],
    'remaining_work': ['核对v5拒绝原因及预算交界标签；实现可审阅的新版入口、基本验证、资源提案和交接。'],
    'blockers': []})
start = original.index('"work_records": [', original.index('"id": "NAV-065"')) + len('"work_records": ')
_, length = json.JSONDecoder().raw_decode(original[start:])
changed = (original[:start + length - 1].rstrip() + ',\n' +
           textwrap.indent(json.dumps(current['work_records'][-1], ensure_ascii=False, indent=2), '        ') +
           '\n      ]' + original[start + length:])
diff = ''.join(list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True)))[2:])
diff = re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
patch = '*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n' + diff + '*** End Patch\n'
subprocess.run(['apply_patch'], input=patch, text=True, cwd=ROOT, check=True)
subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)
print(json.dumps({'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable, 'selected_task': current['id'], 'completed': sum(states.values())}))
