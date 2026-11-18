"""Record v8's audited outcome without changing task definitions or old evidence."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8')
sys.path.insert(0, str(ROOT / '.worklog-testing/NAV-065/staged-v7-preparation'))
from worklog import append
spec = importlib.util.spec_from_file_location('v7_close_helpers', ROOT / '.worklog-testing/NAV-065/staged-v7-execution/close.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
patch, read, sha = helper.patch, helper.read, helper.sha

def main():
    assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
    report_path = NAV / 'reports/nav065_pilot_refit_v8_review.json'
    r = read(report_path)
    assert r == read(LAUNCH / 'review.json') and r['execution_integrity_pass']
    assert not r['scientific_nav065_pass'] and r['development_scoring_rounds'] == 0
    for name, digest in r['evidence']['output_hashes'].items():
        assert sha(Path(r['evidence']['output']) / name) == digest
    first, last = r['initial_gate'], r['final_gate']
    old_mae, new_mae = [g['additional_metrics']['vh']['relative_mae'] for g in (first, last)]
    old_corr, new_corr = [g['additional_metrics']['vh']['pearson'] for g in (first, last)]
    ff_before, ff_after = [r[k].get('false_feasible', 0) for k in ('initial_start_confusion', 'final_start_confusion')]
    fi_before, fi_after = [r[k].get('false_infeasible', 0) for k in ('initial_start_confusion', 'final_start_confusion')]
    numeric, boundary = r['phase_counts'].get('numeric', 0), r['phase_counts'].get('boundary', 0)
    passed = last['pass']
    summary = (f'用户确认v8后在tiger11/tmux完成一次固定full_s0单候选验证，legacy主agent验收，无测试子agent。'
        f'复用该候选162条train/{r["saved_train_steps"]}步，新增回合/开发评分0；实际数值{numeric}+边界{boundary}、共{r["optimizer_trials"]}更新。'
        f'临界数据到达相对MAE {old_mae:.2%}→{new_mae:.2%}（原线35%），Pearson {old_corr:.4f}→{new_corr:.4f}（原线0.70）；'
        f'起点误放行{ff_before}→{ff_after}、误拒绝{fi_before}→{fi_after}。'
        f'停止原因{r["stop_reason"]}，pilot完整train {"PASS" if passed else "FAIL"}；六actor及其余五模型不变、33项v2身份匹配。'
        '无新G4/部署包，原G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持；本次单次尝试已结束，不自动扩展或挪用阶段余量。')
    interpretation = ('本候选通过只支持该固定train集合，不代表六候选或开发放行；进一步扩展需单独有界协议。' if passed else
        '数值门槛未全部通过时禁止进入边界纠错；本次结果应按实际停止原因解释。整体平均误差下降与起点符号可靠性是不同要求，不能以单项改善代替完整验收。')
    doc = ('# NAV-065 v8实际结果与主agent复核\n\n' + summary + '\n\n'
        '| 项目 | 开始 | 末次 | 原要求 |\n| --- | --- | --- | --- |\n'
        f'| 临界数据reach相对MAE | {old_mae:.2%} | {new_mae:.2%} | ≤35% |\n'
        f'| 临界数据reach Pearson | {old_corr:.4f} | {new_corr:.4f} | ≥0.70 |\n'
        f'| 162起点错误放行 | {ff_before} | {ff_after} | 0 |\n'
        f'| 162起点错误拒绝 | {fi_before} | {fi_after} | 按原主预算/边界标准 |\n'
        f'| 所选边界符号错判 | {first["selected_sign_mismatches"]} | {last["selected_sign_mismatches"]} | 0 |\n\n'
        + interpretation + '\n\n'
        '本轮只改critic，固定actor没有更新，因此这些误判变化不能解释为机器狗导航动作退化。NAV-064既有主预算名义导航结果保留；本次没有新的开发、物理或预算搜索成绩。'
        '若本轮未进入边界阶段，也没有实际证据证明受限SGD纠错有效或无效。\n\n'
        '生产调用链保持：固定v7末次来源→原共同停止标签→原分层数值拟合/有条件边界纠错→每步原v5 train_gate→独立保存及CPU重载；本pilot无开发评估调用。'
        '原W=max(V_h,V_c−z/Z_REF)、beta=0、网络/obs-v7、J单位、数据用途及固定v2配套配置保持。\n\n'
        '主agent复核批准pin与快照、57项基本测试的对应源码hash、162条原始轨迹/J账本/共同停止、全部逐步起点算术/标签/分类变化、阶段切换/额度及阶段/末次模型CPU重载。'
        '其余五来源模型全部张量及六actor保持，33项v2资源hash匹配。中间日志已核对，未重放优化器；阶段外没有每步模型张量。执行完整性通过不等于科学通过。\n\n'
        '末次完整train失败项：\n\n' + '\n'.join('- ' + message for message in last['failures']) + '\n\n'
        '后续先依据本轮已有train证据分析数值拟合的剩余误差和边界表现，再提出必要的有界改动。不得自动追加步数、转移阶段余量、挑中间模型、降低门槛或提前进入NAV-066。\n\n'
        '证据：[主agent复核](../reports/nav065_pilot_refit_v8_review.json)、[批准登记](../config/nav065_pilot_refit_prereg_v8.json)、[原方案](nav065_pilot_refit_v8_20261118.md)。'
        '输出`/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_v8/`；资源/源码快照/原始复算`/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8/`；'
        'tmux `nav064-r14-repair-20261115:nav065-v8-run/nav065-v8-acceptance`，Python `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。\n')
    doc_path = NAV / 'docs/nav065_pilot_refit_v8_review_20261118.md'
    patch(doc_path, doc, add=True)
    for path, prefix in [(NAV / 'README.md', '2026-11-18最新：'),
                         (NAV / 'docs/implementation_handoff.md', '## 2026-11-18 NAV-065 v8结束：单候选验证结果\n\n')]:
        original = path.read_text()
        heading = original.splitlines()[0] + '\n'
        link = ('docs/' if path.name == 'README.md' else '') + 'nav065_pilot_refit_v8_review_20261118.md'
        patch(path, heading + '\n' + prefix + summary + '\n\n详见[v8结果](' + link + ')。以下准备/未批准记录保留为历史时点。\n' + original[len(heading):])
    agents = ROOT / 'AGENTS.md'
    original = agents.read_text()
    anchor = '\n## 前置状态：v2 选择与历史校准复核（2026-11-09）\n'
    assert original.count(anchor) == 1
    patch(agents, original.replace(anchor, '\n### NAV-065最新补充（2026-11-18，单候选回修v8已结束）\n\n' + summary +
        '执行完整性通过不等于科学通过；见[结果](CaMRA/navigation/docs/nav065_pilot_refit_v8_review_20261118.md)。上方v8未批准为历史时点。\n' + anchor))
    append({'timestamp': datetime.now().astimezone().isoformat(),
        'summary': 'NAV-065/1.3.1 v8执行收尾：单候选数值优先验证结束，保留实际门禁与下游阻塞',
        'actions': ['用户确认后登记独立批准pin，提交批准版本6759ff8，在新目录执行一次固定full_s0有界pilot。',
            '主agent按legacy复核原始162轨迹/J/共同停止、每步日志/起点算术/分类变化、数值阶段转移规则/额度、阶段及末次CPU重载和身份；无测试子agent。',
            '核对57基本检查源码hash、33项v2、全部源码快照及其余五模型/六actor；保留全部产物/失败并同步结果、README、AGENTS和交接。'],
        'files_changed': [str(report_path.relative_to(ROOT)), str(doc_path.relative_to(ROOT)), 'AGENTS.md',
            'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md',
            'doc/plan/20261109_camra_navigation_implementation.json', str(HERE.relative_to(ROOT)) + '/'],
        'commands': ['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v8-run/nav065-v8-acceptance；cwd=' + str(NAV) + '；Python=' + sys.executable,
            '<Python> -u scripts/refit_nav065_pilot.py --registration config/nav065_pilot_refit_prereg_v8.json',
            '<Python> -u .worklog-testing/NAV-065/pilot-v8-execution/await_review.py、close.py（绝对路径）；legacy validate_worklog.py --mode update'],
        'results': [summary, '本轮仍使用原目标/原v5 train_gate，数值未过不切换；单候选不授权开发/G4/部署，实际开发评分0。',
            '固定策略动作不变，本次critic预测变化不等于NAV-064策略训练退化；不推断未执行的边界SGD或开发/物理效果。',
            '局部执行与全局接口/来源一致性通过；NAV-065全部六候选/开发/G4科学标准仍未满足，保持false和原verification_result，旧失败及全部记录保留。'],
        'evidence': [str(report_path.relative_to(ROOT)), str(LAUNCH / 'review.json'), str(LAUNCH / 'train_cases.json'),
            str(LAUNCH / 'source_before.json'), 'CaMRA/navigation/config/nav065_pilot_refit_prereg_v8.json',
            'CaMRA/navigation/reports/nav065_pilot_refit_v8_preparation_review.json'],
        'remaining_work': ['根据既有train日志诊断剩余问题；需要新增优化/变更时另行具体评审，不重启本run或挪用边界余量。',
            '全部六候选原train及开发G4/部署包通过后才能完成NAV-065，NAV-066及物理下游保持阻塞。'],
        'blockers': ['本次批准单次pilot已结束，整体NAV-065尚未通过；原G4 FAIL保持，无自动扩展/续跑授权。']})
    print(json.dumps({'summary': summary, 'NAV-065': False, 'NAV-066_started': False}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
