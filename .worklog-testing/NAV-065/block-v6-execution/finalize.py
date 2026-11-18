"""Append the verified v6 outcome and save only this task's authorized files."""
from collections import Counter
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
NAV = ROOT / 'CaMRA/navigation'
HERE = ROOT / '.worklog-testing/NAV-065/block-v6-execution'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_block_refit_launch_20261118_v6')
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
read = lambda path: json.loads(path.read_bytes())
r = read(LAUNCH / 'review.json')
assert r['execution_integrity_pass'] and not r['scientific_pass']
assert r['development_scoring_rounds'] == 0 and r['run_status'] == 'HELD_TRAIN_GATE'
old, new = read(LAUNCH / 'before-plan.json'), read(PLAN)
assert {k: v for k, v in old.items() if k != 'tasks'} == {k: v for k, v in new.items() if k != 'tasks'}
for a, b in zip(old['tasks'], new['tasks'], strict=True):
    if a['id'] == 'NAV-065':
        assert not b['status'] and b['work_records'][:-1] == a['work_records']
        assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
    else:
        assert a == b
assert sum(t['status'] for t in new['tasks']) == 52

def apply(patch):
    subprocess.run(['apply_patch'], cwd=ROOT, input=patch, text=True, check=True)

def add(path, content):
    assert not path.exists()
    apply('*** Begin Patch\n*** Add File: ' + str(path) + '\n' + ''.join('+' + s + '\n' for s in content.splitlines()) + '*** End Patch\n')

def prepend(path, anchor, content):
    assert path.read_text().count(anchor) == 1
    apply('*** Begin Patch\n*** Update File: ' + str(path) + '\n@@\n ' + anchor + '\n' + ''.join('+' + s + '\n' for s in ('\n' + content + '\n').splitlines()) + '*** End Patch\n')

trials = '/'.join(str(c['trials']) for c in r['train_candidates'])
updates = '/'.join(str(c['updates']) for c in r['train_candidates'])
selected_before = sum(c['initial_gate']['selected_sign_mismatches'] for c in r['train_candidates'])
selected_after = sum(c['selected_sign_mismatches'] for c in r['train_candidates'])
reasons = Counter()
for candidate in r['train_candidates']:
    reasons.update(candidate['rejected_block_reasons'])
summary = (f"用户确认v6方案后在tiger11/tmux完成一次独立离线回修。复用972条train/372471步，新增回合0；六候选试步{trials}，共{r['optimizer_trials']}次，接受更新{updates}，共{r['accepted_updates']}次，六actor不变。"
    f"旧五档train仍6/6通过；新增临界预算误放行{r['initial_train_false_feasible']}→{r['train_false_feasible']}，所选边界错判{selected_before}→{selected_after}，完整train {r['train_pass_candidates']}/6通过，状态HELD_TRAIN_GATE。"
    "按批准失败协议，开发重评0、无新G4/部署包；原G4 FAIL、52/67、NAV-065=false和NAV-066未启动保持。未用试步不能自动续跑、转移或重启本run。")
table = '| 候选 | 实际试步/接受更新 | 新增起点误放行 | 所选边界错判 | 新增reach相对MAE |\n| --- | --- | --- | --- | --- |\n'
for c in r['train_candidates']:
    b = c['initial_gate']
    table += f"| {c['run_id'].removeprefix('nav064_terminal_tail_v1_')} | {c['trials']}/{c['updates']} | {b['additional_false_feasible']}→{c['additional_false_feasible']} | {b['selected_sign_mismatches']}→{c['selected_sign_mismatches']} | {b['additional_metrics']['vh']['relative_mae']:.4f}→{c['additional_metrics']['vh']['relative_mae']:.4f} |\n"
document = '# NAV-065 v6执行与主agent复核\n\n' + summary + '\n\n' + table + '\n'
document += ('v6沿用共同停止的cost/reach标签、W=max(V_h,V_c-z/550)、beta=0及原最终v5 train/G4标准。训练采样分层、最多4步一段和固定源工作波动界均按用户确认执行，拒绝段实际步数计入。'
    '批准登记[prereg](../config/nav065_block_refit_prereg_v6.json)的pin为`' + r['registration_pin'] + '`；原未批准proposal保留，实际从批准提交773f433启动。\n\n')
document += '段末拒绝原因计数：`' + json.dumps(dict(reasons), ensure_ascii=False) + '`。这是日志中的直接保护触发，不能仅由计数推断唯一根因。工作波动界没有替代最终验收，短段允许暂时变化也未让本轮完整train通过。\n\n'
document += ('主agent按legacy验证，无测试子agent。63项基本测试通过；真实972条轨迹重新核对J账本、共同停止cost/reach目标及可行性，复算六候选完整train门禁，检查固定actor、原先正确起点、原结果、CPU重载、实际试步与段接受计量，33项冻结v2和所有来源快照hash一致。'
    '中间张量/Adam/RNG回滚使用基本受控测试验证；没有为复核额外重放优化器步骤。检查通过说明执行和记录可信，不表示科学达标。\n\n')
document += ('[机器复核报告](../reports/nav065_block_refit_v6_review.json)索引全部输出hash。正式产物：`' + r['evidence']['output'] + '`；启动/源码快照/资源/命令/原始日志/逐例复算：`' + r['evidence']['launch'] + '`。原模型、v5训练轨迹和失败日志保留，无final或预算选择集访问。\n\n'
    '后续先用现有train证据分析哪项保护与数值/符号目标冲突，不能以基本测试通过启动NAV-066，也不能把本次未用额度自动用于新尝试。若设计新训练、改资源或保护规则，另行形成具体可审阅方案并按计划确认。\n')
doc = NAV / 'docs/nav065_block_refit_v6_review_20261118.md'
add(doc, document)
prepend(NAV / 'README.md', '# 上层最小成本到达避障导航器',
    '2026-11-18最新结果：' + summary + ' 主agent按legacy完成63基本测试及原始轨迹/实际门禁复核，无测试子agent。见[v6结果](docs/nav065_block_refit_v6_review_20261118.md)。以下为历史时点。')
prepend(NAV / 'docs/implementation_handoff.md', '# 上层导航实施交接',
    '## 2026-11-18 NAV-065 v6结束：短段离线回修仍未过完整train\n\n' + summary + '\n\n'
    '主agent按legacy完成63基本测试、972条轨迹/J/共同停止及实际门禁、来源/33项v2、段计量与CPU重载核验，无专用测试子agent。见[v6详细结果](nav065_block_refit_v6_review_20261118.md)及[机器报告](../reports/nav065_block_refit_v6_review.json)。产物与源码快照独立保存；下一步先分析现有train失败，不能自动续跑。下方为历史时点。')
anchor = '## 前置状态：v2 选择与历史校准复核（2026-11-09）'
addition = '### NAV-065最新补充（2026-11-18，短段回修v6已结束）\n\n' + summary + ' 63基本测试和主agent原始轨迹/实际门禁复核通过，无测试子agent；见CaMRA/navigation/docs/nav065_block_refit_v6_review_20261118.md。上方v6准备/未批准为历史时点，本次确认只授权已结束的一个独立尝试。\n\n'
apply('*** Begin Patch\n*** Update File: ' + str(ROOT / 'AGENTS.md') + '\n@@\n-' + anchor + '\n' + ''.join('+' + s + '\n' for s in (addition + anchor).splitlines()) + '*** End Patch\n')
record = {'timestamp': datetime.now().astimezone().isoformat(),
    'summary': 'NAV-065/1.3.1 v6一次正式回修与legacy复核结束：完整train未过，HELD，无开发重评',
    'actions': ['按用户确认的独立v6登记执行一次新run，训练前本地提交773f433；复用972条train，不新增采样。',
        '分层回归及最多4步一段、源锚定工作保护实际执行；每个拒绝段内试步计入额度，拒绝时恢复模型/优化器/随机数，六actor不变。',
        '主agent按legacy完成63基本检查、972条轨迹/J/共同停止和可行性复算、六模型CPU重载、实际train门禁及段计量复核；未使用测试子agent。',
        '保持全部旧产物/失败、源快照和原门禁；同步结果报告、入口与交接，仅向当前任务追加工作记录。'],
    'files_changed': ['CaMRA/navigation/reports/nav065_block_refit_v6_review.json', str(doc.relative_to(ROOT)), 'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', str(PLAN.relative_to(ROOT)), '.worklog-testing/NAV-065/block-v6-execution/'],
    'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav065-v6-run/nav065-v6-review/nav065-v6-final；cwd=' + str(NAV) + '；Python=' + sys.executable,
        'PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 <Python> -u scripts/refit_nav065_blocks.py --registration config/nav065_block_refit_prereg_v6.json',
        '同环境 <Python> -u .worklog-testing/NAV-065/block-v6-execution/review.py（绝对路径）；legacy validate_worklog.py --mode update；git -c core.whitespace=cr-at-eol diff --check'],
    'results': [summary, '段末拒绝原因计数=' + json.dumps(dict(reasons), ensure_ascii=False),
        '局部执行完整性PASS，科学验收未过。RC-PPO预算增广/共同停止/同目标双critic/W/beta及最终v5 train/G4、actor、v2整套校准/地图/时序/单位/数据用途保持；不修改论文/参考源码。',
        '本轮生产refit入口实际调用原v5 train_gate；失败分支阻断原development/G4消费者，未创建部署包，不能交付NAV-066；旧NAV-064=true不替critic放行。',
        '更新前正确起点保留、33项v2/原来源hash一致；中间回滚机制以基本受控测试验，不额外复演正式optimizer。'],
    'evidence': ['CaMRA/navigation/reports/nav065_block_refit_v6_review.json', str(doc.relative_to(ROOT)), str(LAUNCH / 'review.json'), str(LAUNCH / 'train_cases.json'), str(LAUNCH / 'EXIT.json')],
    'remaining_work': ['先基于现有train日志分析保护触发与数值/符号冲突；新增训练/规则或额度须独立方案确认，全部train及原G4通过后再交付NAV-066。'],
    'blockers': ['完整train未通过，本次批准尝试按失败协议结束；不自动重启、用未用额度继续或转移候选余量。']}
original = PLAN.read_text()
start = original.index('"work_records": [', original.index('"id": "NAV-065"')) + len('"work_records": ')
_, length = json.JSONDecoder().raw_decode(original[start:])
changed = original[:start + length - 1].rstrip() + ',\n' + textwrap.indent(json.dumps(record, ensure_ascii=False, indent=1), '    ') + '\n   ]' + original[start + length:]
diff = ''.join(list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True)))[2:])
diff = re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
apply('*** Begin Patch\n*** Update File: ' + str(PLAN) + '\n' + diff + '*** End Patch\n')
final = read(PLAN)
for a, b in zip(old['tasks'], final['tasks'], strict=True):
    if a['id'] == 'NAV-065':
        assert b['work_records'][:-2] == a['work_records']
        assert {k: v for k, v in a.items() if k != 'work_records'} == {k: v for k, v in b.items() if k != 'work_records'}
    else:
        assert a == b
subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(PLAN), '--mode', 'update'], check=True)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT)
owned = ['AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
    str(doc.relative_to(ROOT)), 'CaMRA/navigation/reports/nav065_block_refit_v6_review.json', str(PLAN.relative_to(ROOT)),
    '.worklog-testing/NAV-065/block-v6-execution/review.py', '.worklog-testing/NAV-065/block-v6-execution/finalize.py']
subprocess.run(['git', 'add', '--', *owned], cwd=ROOT, check=True)
staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
assert set(staged) == set(owned)
assert not any(s.endswith(('.tex', '.bib')) or s.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/')) for s in staged)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
with (LAUNCH / 'result-commit.log').open('x') as stream:
    subprocess.run(['git', 'commit', '-m', 'Preserve NAV-065 critic block refit v6 result and verification'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
print('COMMIT', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), flush=True)
print(summary, flush=True)
