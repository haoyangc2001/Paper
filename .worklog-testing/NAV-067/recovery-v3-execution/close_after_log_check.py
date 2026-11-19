"""Preserve the failed watcher, restore append-only formatting, and commit locally."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from nav067_sources import read_manifest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
EXPECTED_HEAD = '411ce7bf71a5c47332aae133015adfae71dfffd7'
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE') == '%599'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == EXPECTED_HEAD
failure = json.loads((HERE / 'REVIEW_HELD.json').read_bytes())
assert "['git', 'diff', '--cached', '--check']" in failure['traceback']
for filename in ['COORDINATOR_STARTED.json', 'launch-L1_empty.json', 'launch-L2_single_detour.json', 'launch-L3_multi_path.json']:
    assert not Path('/proc', str(json.loads((HERE / filename).read_bytes())['pid'])).exists()
finished = json.loads((HERE / 'FINISHED.json').read_bytes())
assert finished['new_episodes'] == 6 and not (HERE / 'HELD.json').exists()
report_path = ROOT / 'CaMRA/navigation/reports/nav067_recovery_v3_execution_review.json'
report = json.loads(report_path.read_bytes())
assert report['status'] == 'PILOT_TRACE_PASS_NAV067_INCOMPLETE'
assert report['saved_episode_results'] == report['trace_audited_episodes'] == 6
assert not report['main_agent_saved_episode_review_failures']
assert not report['main_agent_saved_runtime_review_failures']
for name, field in [('review_saved_episode.py', 'main_agent_saved_episode_review_source_sha256'), ('review_runtime.py', 'main_agent_saved_runtime_review_source_sha256')]:
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == report[field]
for row in report['main_agent_saved_episode_reviews']:
    assert json.loads((HERE / ('main-agent-' + row['level'] + '-' + row['method'] + '.json')).read_bytes()) == row
for row in report['main_agent_saved_runtime_reviews']:
    assert json.loads((HERE / ('runtime-review-' + row['level'] + '.json')).read_bytes()) == row
read_manifest()
old_bytes = subprocess.check_output(['git', 'show', 'HEAD:' + str(PLAN.relative_to(ROOT))], cwd=ROOT)
old = json.loads(old_bytes)
current = json.loads(PLAN.read_bytes())
assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in current.items() if k != 'tasks'}
for a, b in zip(old['tasks'], current['tasks']):
    assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-067': assert a == b
task = next(t for t in current['tasks'] if t['id'] == 'NAV-067')
original_task = next(t for t in old['tasks'] if t['id'] == 'NAV-067')
assert len(task['work_records']) == len(original_task['work_records']) + 1
with (HERE / 'failed-finalizer-plan.json').open('xb') as stream: stream.write(PLAN.read_bytes())
task['work_records'].append({
    'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-067收尾复核：保留自动提交失败，恢复原计划换行并显式提交实际证据',
    'actions': ['核实原协调器及三地图进程已结束，6回合与6轨迹/3运行时核验均已保存通过；重新核验295项保护身份。',
        '自动收尾已生成真实报告，但Git空白检查因原始日志及计划CRLF转换返回2；保留REVIEW_HELD和失败计划快照，不重跑仿真或离线核验。',
        '以当前HEAD计划原字节为基础仅追加本任务收尾记录，逐项确认全部旧字段/状态/记录不变；原始日志逐字保留，源码/JSON/文档单独做空白检查。',
        '明确路径纳入6回合主agent核验、3项运行时核验、L1描述性诊断与自动提交失败证据，仅本地提交。'],
    'files_changed': [str(PLAN.relative_to(ROOT)), str(HERE.relative_to(ROOT))],
    'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav067-evidence-close @599/%599; ' + sys.executable + ' -B close_after_log_check.py',
        'legacy validate_worklog.py <NAV plan> --mode update; git diff --cached --check excluding unchanged raw log files; git commit (no push)'],
    'results': ['6/6接线轨迹与3/3运行时readback核验通过，0新增仿真/训练/W查询；NAV-067仍false、54/67。',
        '完整方法L1/L3安全及时到达、L2超时；无校准三图超时；六回合均超过共同预算，不构成W所选预算性能通过。',
        '本项工程证据不替代实际受控反例、同配置重复及NAV-043适用性验收；旧科学FAIL保留。'],
    'evidence': [str(report_path.relative_to(ROOT)), str(HERE.relative_to(ROOT)) + '/REVIEW_HELD.json',
        str(HERE.relative_to(ROOT)) + '/manual-review-receipt.json'],
    'remaining_work': ['取得真实受控反例和同配置重复的独立额度并完成NAV-067剩余验收，再按依赖进入NAV-043。'],
    'blockers': ['本次批准六回合已结束，真实受控反例与同配置重复仍无独立额度。']})
text = old_bytes.decode('utf-8')
offset = text.index('"id": "NAV-067"')
opening = text.index('[', text.index('"work_records"', offset))
records, length = json.JSONDecoder().raw_decode(text[opening:])
closing = opening + length - 1
assert records == original_task['work_records'] and b'\r\n' not in old_bytes
extra = task['work_records'][len(records):]
body = ',\n'.join('\n'.join('        ' + line for line in json.dumps(r, ensure_ascii=False, indent=2).splitlines()) for r in extra)
restored = text[:closing].rstrip(' \t\r\n') + ',\n' + body + '\n      ' + text[closing:]
assert json.loads(restored) == current
PLAN.write_bytes(restored.encode('utf-8'))
receipt = {'status': 'MANUAL_RECOVERY_OF_LOCAL_COMMIT_ONLY', 'at': datetime.now().astimezone().isoformat(),
    'parent_git_head': EXPECTED_HEAD, 'watcher_failure_preserved': True,
    'reason': 'Raw log whitespace plus non-semantic CRLF conversion in the finalizer snapshot; original HEAD layout restored with append-only records.',
    'saved_episode_results': 6, 'trace_audited_episodes': 6, 'runtime_reviews': 3,
    'protected_hashes_checked': report['protected_hashes_checked'], 'new_simulation_starts': 0,
    'training_updates': 0, 'W_queries': 0, 'remote_push': False, 'NAV_067_status': False}
with (HERE / 'manual-review-receipt.json').open('x') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
check = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], cwd=ROOT, capture_output=True, text=True)
with (HERE / 'manual-close-plan-check.log').open('x') as stream: stream.write(check.stdout + check.stderr)
assert check.returncode == 0, check.stdout + check.stderr
names = ['COORDINATOR_STARTED.json', 'REVIEW_WATCH_STARTED.json', 'FINISHED.json', 'REVIEW_HELD.json',
    'complete_once.py', 'finish_review.py', 'coordinator.log', 'terminal-review.log', 'final-plan-check.log',
    'live-stack-L1-1.log', 'review_saved_episode.py', 'review_runtime.py', 'L1_observe.py',
    'L1-observational-diagnostics.json', 'L1-observational-diagnostics.log', 'manual-review-receipt.json',
    'manual-close-plan-check.log', Path(__file__).name]
for level in ('L1_empty', 'L2_single_detour', 'L3_multi_path'):
    names.extend([prefix + level + '.json' for prefix in ('launch-', 'exit-', 'audit-')])
    names.append(level + '.log')
    names.extend(['runtime-review-' + level + suffix for suffix in ('.json', '.log')])
    for method in ('uncalibrated', 'full'):
        names.extend(['main-agent-' + level + '-' + method + suffix for suffix in ('.json', '.log')])
paths = [str(PLAN.relative_to(ROOT)), str(report_path.relative_to(ROOT)), 'CaMRA/navigation/docs/implementation_handoff.md']
paths.extend(str((HERE / name).relative_to(ROOT)) for name in names)
assert all((ROOT / p).is_file() for p in paths)
initial = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines())
assert initial.issubset(set(paths)), initial - set(paths)
subprocess.run(['git', 'add', '-f', '--', *paths], cwd=ROOT, check=True)
staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines())
assert staged == set(paths)
subprocess.run(['git', 'diff', '--cached', '--check', '--', '.', ':(exclude,glob)**/*.log'], cwd=ROOT, check=True)
for p in paths:
    assert subprocess.check_output(['git', 'show', ':' + p], cwd=ROOT) == (ROOT / p).read_bytes()
subprocess.run(['git', 'commit', '-m', 'chore(nav): preserve NAV-067 physical pilot results and verification evidence'], cwd=ROOT, check=True)
print(json.dumps({'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'committed_paths': len(paths), 'report_status': report['status'], 'NAV_067_status': False,
    'new_simulation_starts': 0, 'training_updates': 0, 'W_queries': 0, 'remote_push': False}, ensure_ascii=False, indent=2), flush=True)
