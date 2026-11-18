"""Publish the verified failed v7 result and append its existing task record."""
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7')
sys.path.insert(0, str(ROOT / '.worklog-testing/NAV-065/staged-v7-preparation'))
from worklog import append

read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def patch(path, changed, add=False):
    if add:
        assert not path.exists()
        content = '*** Add File: ' + str(path) + '\n' + ''.join('+' + s + '\n' for s in changed.splitlines())
    else:
        original = path.read_text()
        diff = ''.join(list(difflib.unified_diff(original.splitlines(True), changed.splitlines(True)))[2:])
        content = '*** Update File: ' + str(path) + '\n' + re.sub(r'^@@.*@@$', '@@', diff, flags=re.M)
    subprocess.run(['apply_patch'], input='*** Begin Patch\n' + content + '*** End Patch\n', text=True, cwd=ROOT, check=True)


def main():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    report_path = NAV / 'reports/nav065_staged_refit_v7_review.json'
    r = read(report_path)
    assert r['execution_integrity_pass'] and not r['scientific_pass']
    assert r['run_status'] == 'HELD_TRAIN_GATE' and r['development_scoring_rounds'] == 0
    assert read(LAUNCH / 'review.json') == r
    for name, digest in r['evidence']['output_hashes'].items():
        assert sha(Path(r['evidence']['output']) / name) == digest
    candidates = r['train_candidates']
    trials = '/'.join(str(c['trials']) for c in candidates)
    phase_pairs = '/'.join(str(c['phase_counts'].get('numeric', 0)) + '+' + str(c['phase_counts'].get('boundary', 0)) for c in candidates)
    initial_false, final_false = r['initial_train_false_feasible'], r['train_false_feasible']
    initial_sign = sum(c['initial_gate']['selected_sign_mismatches'] for c in candidates)
    final_sign = sum(c['selected_sign_mismatches'] for c in candidates)
    old_pass = sum(c['original_grid']['pass'] for c in candidates)
    summary = (f'用户确认v7方案后在tiger11/tmux完成一次独立两阶段critic回修，legacy主agent验收，无测试子agent。'
        f'复用972条train/372471步，新增回合0；六候选数值+边界步数分别{phase_pairs}，共{r["optimizer_trials"]}次更新，六actor不变、33项v2身份匹配。'
        f'起点误放行{initial_false}→{final_false}，所选边界错判{initial_sign}→{final_sign}；旧五档{old_pass}/6、完整train {r["train_pass_candidates"]}/6通过，状态HELD_TRAIN_GATE。'
        '按批准协议开发重评0，无新G4/部署包；原G4 FAIL、52/67、NAV-065=false、NAV-066未启动保持。'
        '本次额度已结束，不能自动重启或转移余量。')
    table = ['| 候选 | 数值+边界更新 | 起点误放行（前→后） | 新增reach相对MAE（前→数值段→末次） | 边界错判（前→后） | 完整train |',
             '| --- | --- | --- | --- | --- | --- |']
    for c in candidates:
        name = c['run_id'].replace('nav064_terminal_tail_v1_', '')
        first = c['initial_gate']
        numeric = next((s['train_gate'] for s in c['stage_checkpoints'] if s['stage'] == 'numeric'), first)
        mae = '→'.join(f'{g["additional_metrics"]["vh"]["relative_mae"]:.1%}' for g in (first, numeric, c))
        table.append(f'| {name} | {c["phase_counts"].get("numeric", 0)}+{c["phase_counts"].get("boundary", 0)} | {c["initial_false_feasible"]}→{c["all_start_confusion"].get("false_feasible", 0)} | {mae} | {first["selected_sign_mismatches"]}→{c["selected_sign_mismatches"]} | {"PASS" if c["pass"] else "FAIL"} |')
    details = '\n'.join(f'- {c["run_id"].replace("nav064_terminal_tail_v1_", "")}: ' + '; '.join(c['failures']) for c in candidates)
    doc = ('# NAV-065 v7实际结果与主agent复核\n\n' + summary + '\n\n' + '\n'.join(table) +
        '\n\n顺序为无校准seed0/1/2、完整方法seed0/1/2。相对MAE分母和旧/新分片保持原标准，门槛为35%；原始数据和每阶段模型保留。\n\n'
        '本轮取消了旧正确起点的过程单调保护，实际优化每步都执行，没有按旧保护回退。逐步日志区分误放行、误拒绝及主预算变化；完整train每步检查，未达到全部标准。'
        '因此不能再仅用“保护锁住了更新”解释本轮失败；本次有限步数和分阶段方法仍不足以让原数值与符号条件同时满足。这不证明固定目标不存在可拟合解，也不证明只需继续加步数。\n\n'
        '论文§4.2固定部署actor后重估价值、再做预算搜索的顺序保持；生产入口直接调用原v5 train_gate和原部署evaluate/G4。共同停止目标、beta=0、actor、v2/校准、地图、单位和分片均未改。'
        '本次未触发开发评估，因此不能推断开发性能变好或变差，也不影响NAV-064既有主预算名义策略结果；未证明物理闭环成功。\n\n'
        '各候选末次失败项：\n\n' + details + '\n\n'
        '主agent复核了批准pin/源码快照、33项v2、972条原始轨迹/372471步的J账本与共同停止真值、每个更新的阶段及额度、逐起点分类变化、阶段模型、六actor和CPU重载。'
        '79项准备基本测试证据与未变源码匹配，本轮没有为凑测试数再次训练。逐步模型张量只在阶段边界保存；中间日志的数值/算术和分类变化已核对，未重放优化器。\n\n'
        '后续先用现有train日志分析数值阶段和边界阶段的冲突、具体误判分量及数据表示；若需改变过程、网络或新增训练额度，另行形成可审阅方案。不能挑中间checkpoint冒充批准末次结果、放宽门槛、转用开发数据拟合或提前启动NAV-066。\n\n'
        '证据：[机器复核](../reports/nav065_staged_refit_v7_review.json)、[批准登记](../config/nav065_staged_refit_prereg_v7.json)、'
        '[方案](nav065_staged_refit_v7_20261118.md)。外部输出`/pub/data/caohy/work/camra_navigation/nav065_staged_refit_v7/`；'
        '启动/资源/源码快照/逐例复算`/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7/`。'
        '执行提交`3a4362a`，tmux `nav064-r14-repair-20261115:nav065-v7-run/nav065-v7-acceptance`；Python '
        '`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。\n')
    doc_path = NAV / 'docs/nav065_staged_refit_v7_review_20261118.md'
    patch(doc_path, doc, add=True)
    handoff = NAV / 'docs/implementation_handoff.md'
    text = handoff.read_text()
    heading = text.splitlines()[0] + '\n'
    patch(handoff, heading + '\n## 2026-11-18 NAV-065 v7结束：两阶段回修未通过完整train\n\n' + summary +
          '\n\n详见[v7结果](nav065_staged_refit_v7_review_20261118.md)和[主agent复核](../reports/nav065_staged_refit_v7_review.json)。'
          '已保存逐步分类变化和阶段模型；先诊断现有证据，不以中间改善选模型替换末次结果。下方未批准/待启动均为历史时点。\n' + text[len(heading):])
    readme = NAV / 'README.md'
    text = readme.read_text()
    heading = text.splitlines()[0] + '\n'
    patch(readme, heading + '\n2026-11-18最新：' + summary + '见[v7结果](docs/nav065_staged_refit_v7_review_20261118.md)。以下为历史时点。\n' + text[len(heading):])
    agents = ROOT / 'AGENTS.md'
    text = agents.read_text()
    anchor = '\n## 前置状态：v2 选择与历史校准复核（2026-11-09）\n'
    assert text.count(anchor) == 1
    patch(agents, text.replace(anchor, '\n### NAV-065最新补充（2026-11-18，两阶段回修v7已结束）\n\n' + summary +
        '79项基本测试证据与未变源码匹配，主agent逐步日志/阶段模型/原始轨迹验收通过不等于科学通过。见[结果](CaMRA/navigation/docs/nav065_staged_refit_v7_review_20261118.md)。上方v7未批准为历史时点。\n' + anchor))
    append({'timestamp': datetime.now().astimezone().isoformat(),
        'summary': 'NAV-065/1.3.1 v7执行收尾：两阶段离线回修仍未通过完整train，按协议阻断开发和下游',
        'actions': ['根据用户确认登记独立批准pin，现场核验并提交3a4362a后在新目录执行一次150数值+150边界的有界回修。',
                    '主agent按legacy复核源码/批准pin、33项v2、原始J账本及共同停止、每步阶段/额度/分类变化、阶段checkpoint和CPU重载；无测试子agent。',
                    '保留六组末次及阶段模型、全部日志/失败；同步导航README、AGENTS、交接及结果说明，旧状态/记录不改。'],
        'files_changed': ['CaMRA/navigation/reports/nav065_staged_refit_v7_review.json', str(doc_path.relative_to(ROOT)),
                          str(readme.relative_to(ROOT)), str(handoff.relative_to(ROOT)), 'AGENTS.md',
                          'doc/plan/20261109_camra_navigation_implementation.json', str(HERE.relative_to(ROOT)) + '/'],
        'commands': ['tiger11/tmux nav064-r14-repair-20261115:nav065-v7-run/nav065-v7-acceptance；cwd=' + str(NAV) + '；Python=' + sys.executable,
                     '<Python> -u scripts/refit_nav065_staged.py --registration config/nav065_staged_refit_prereg_v7.json',
                     '<Python> -u .worklog-testing/NAV-065/staged-v7-execution/await_review.py、close.py（绝对路径）；legacy validate_worklog.py --mode update'],
        'results': [summary, '原数值MAE/Pearson、零假可行/主预算识别及所选边界符号标准均保持；不是只要求平均误差。',
                    '数值阶段与边界阶段分别计数且无余量转移、过程无回退；原始972条标签/目标、六actor、v2/校准、地图/时序/单位/分片和参考/论文不变。',
                    '原v5 train_gate已在真实入口每步调用；六组全过后原evaluate/G4才可调用的链保持，本轮因train失败实际未调用开发。',
                    '局部执行完整性通过，NAV-065科学验收失败；不更新旧G4或生成部署包，NAV-066/067及物理后续仍受依赖阻塞。'],
        'evidence': [str(report_path.relative_to(ROOT)), str(LAUNCH / 'review.json'), str(LAUNCH / 'train_cases.json'),
                     str(LAUNCH / 'source_before.json'), 'CaMRA/navigation/config/nav065_staged_refit_prereg_v7.json',
                     'CaMRA/navigation/reports/nav065_staged_refit_v7_preparation_review.json'],
        'remaining_work': ['基于已有逐步日志和阶段模型诊断数值/符号冲突；需新方法或额度时先形成具体方案，不能重启、挑中间结果或扩大数据用途。',
                           '全部原train和开发G4通过并有同版部署包后才能完成NAV-065及交付NAV-066。'],
        'blockers': ['本轮末次完整train未全部通过；已批准v7尝试结束，无继续更新授权，原G4 FAIL保持。']})
    print(json.dumps({'documents_updated': True, 'nav065_status': False, 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
