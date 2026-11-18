"""Append-only NAV-065 v7 preparation records; run on tiger11 inside tmux."""
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import textwrap

ROOT = Path('/home/caohy/repositories/Paper')
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
VALIDATOR = '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py'


def append(record):
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)
    original = PLAN.read_text()
    plan = json.loads(original)
    states = {t['id']: t['status'] for t in plan['tasks']}
    task = next(t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
    assert task['id'] == 'NAV-065' and sum(states.values()) == 52
    assert not any(w['summary'] == record['summary'] for w in task['work_records'])
    start = original.index('"work_records": [', original.index('"id": "NAV-065"')) + len('"work_records": ')
    _, length = json.JSONDecoder().raw_decode(original[start:])
    changed = (original[:start + length - 1].rstrip() + ',\n' +
               textwrap.indent(json.dumps(record, ensure_ascii=False, indent=2), '        ') +
               '\n      ]' + original[start + length:])
    diff = ''.join(list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True)))[2:])
    diff = re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
    subprocess.run(['apply_patch'], input='*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n' + diff + '*** End Patch\n',
                   text=True, cwd=ROOT, check=True)
    subprocess.run([sys.executable, VALIDATOR, str(PLAN), '--mode', 'update'], check=True)


if __name__ == '__main__':
    append({
        'timestamp': datetime.now().astimezone().isoformat(),
        'summary': 'NAV-065/1.3.1 v7准备开工：分开离线训练过程与最终门禁，原目标和最终标准不变',
        'actions': ['按legacy重新读取根AGENTS、模块README、完整JSON及当前任务/依赖/历史记录，核对NAV-064=true和NAV-065为唯一可执行任务。'],
        'files_changed': ['doc/plan/20261109_camra_navigation_implementation.json', '.worklog-testing/NAV-065/staged-v7-preparation/worklog.py'],
        'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav065-v7-prepare；cwd=CaMRA/navigation；Python=' + sys.executable, 'legacy validate_worklog.py --mode update'],
        'results': ['作用：为NAV-066提供与固定actor匹配、可用于价值预算搜索的双critic；当前仅NAV-065隔离修正和基本验证。',
                    '上游：a8dd939的v6结果，972条train/372471步和六份v6最终模型；原G4仍FAIL、完整train 0/6，NAV-066未启动。',
                    '允许新增导航v7入口、基本测试、未批准资源提案和文档；不用测试子agent。保持六actor、共同停止cost/reach标签/W、beta=0、最终v5 train/G4及v2整套配置/校准、地图/时序/单位/分片。',
                    '验收：主agent基本测试、真实train只读来源/起点诊断、原目标及生产train/G4调用链、版本身份和工作记录校验；本轮不访问选择/final，不正式训练或开发重评。',
                    '新训练的具体额度与取消过程单调保护须完成可审阅准备后确认，不能使用已结束v6未用额度；代码准备通过不等于科学通过。'],
        'evidence': ['用户指定create-plan-worklog-legacy基于前述分析推进', 'CaMRA/navigation/reports/nav065_block_refit_v6_review.json', 'a8dd939f5caa712d19d6504b428af0e3d6b82cd9'],
        'remaining_work': ['准备两阶段critic拟合、逐起点变化日志和有限新额度；完成基本验证后提交供确认。'],
        'blockers': []})
