from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PY = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX') and sys.executable == PY
load = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = load(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json')
states = {t['id']: t['status'] for t in plan['tasks']}
assert sum(states.values()) == 52
assert next(t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))['id'] == 'NAV-065'
with (HERE / 'validation-before.log').open('x') as out:
    subprocess.run([PY, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py', str(ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'), '--mode', 'update'], stdout=out, stderr=subprocess.STDOUT, check=True)
v2 = load(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
assert all(sha(p) == h for p,h in v2.items())
sources = [
    ('original', '/pub/data/caohy/work/camra_navigation/nav065_deployment_v1', NAV / 'reports/nav065_deployment_source_binding_v1.json'),
    ('refit_v1', '/pub/data/caohy/work/camra_navigation/nav065_refit_v1', Path('/pub/data/caohy/work/camra_navigation/nav065_refit_v1/fitted_candidates.json')),
    ('guarded_v4', '/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_v4', Path('/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_v4/fitted_candidates.json')),
]
ranking = []
evidence = {}
for version, root, manifest in sources:
    report_path = Path(root) / 'deployment_critic_validation_v2.json'
    report = load(report_path)
    identities = {c['run_id']:c for c in load(manifest)['candidates']}
    evidence[str(report_path)] = sha(report_path)
    evidence[str(manifest)] = sha(manifest)
    for row in report['candidates']:
        c = identities[row['run_id']]
        assert sha(c['checkpoint']) == c['checkpoint_sha256']
        counts = row['by_budget']
        primary = counts['1.15']['true_feasible']
        fp = sum(x['false_feasible'] for x in counts.values())
        fn = sum(x['false_infeasible'] for x in counts.values())
        error = sum(row['metrics'][role]['relative_mae'] for role in ('vc_norm','vh'))
        rank = [-primary, fp, fn, error, c['seed'], version]
        ranking.append({'version':version, 'identity':c, 'primary_correct':primary, 'dev_false_feasible':fp, 'dev_false_infeasible':fn, 'dev_combined_relative_mae':error, 'rank':rank})
best_per_run = [min([x for x in ranking if x['identity']['run_id'] == rid], key=lambda x:x['rank']) for rid in sorted({x['identity']['run_id'] for x in ranking})]
selected = [min([x for x in best_per_run if x['identity']['variant'] == method], key=lambda x:x['rank']) for method in ('uncalibrated','full')]
selection = {'rule':'Use existing development only: primary correct descending, false permits ascending, false rejections ascending, combined relative MAE ascending, seed then version as deterministic ties. No new scoring or independent results used.', 'user_override':'确认，用目前训练出来的最好的模型去检验；supersedes draft v7/latest/minimum-seed source recommendation, not thresholds or 90 episode cap.', 'ranked_existing_development':ranking, 'six_best_by_run':best_per_run, 'selected':selected, 'evidence':evidence}
with (HERE / 'selection.json').open('x') as out:
    json.dump(selection,out,ensure_ascii=False,indent=2)
baseline = {'timestamp':datetime.now().astimezone().isoformat(), 'plan':plan, 'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'v2':v2, 'protected':{str(p):sha(p) for p in list((NAV/'config').glob('*.json')) + list((NAV/'reports').glob('nav065*review.json'))}, 'host':socket.getfqdn(), 'python':sys.executable, 'tmux':os.environ['TMUX'], 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader'],text=True)}
with (HERE / 'baseline.json').open('x') as out:
    json.dump(baseline,out,ensure_ascii=False,indent=2)
record = {'timestamp':baseline['timestamp'], 'summary':'NAV-065新版协议激活开工：用户确认独立检验并要求按已有开发结果选最好模型，不训练', 'actions':['重新读取根AGENTS、legacy技能/执行及字段规则、计划全文与当前任务/依赖/历史、导航README和交接。','在tiger11/tmux核验绝对解释器/GPU、33项冻结v2和原始模型；完整读取既有original/refit-v1/v4开发报告，不执行新评分。','按用户最新指令覆盖草案v7/最小种子建议，按既有开发主档识别、误放行、误拒绝、平均误差排序并固定最优来源；新独立集尚未生成或查询。'], 'files_changed':['doc/plan/20261109_camra_navigation_implementation.json','.worklog-testing/NAV-065/best-decision-v2/'], 'commands':[PY+' -B -u '+str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav065-best-v2; cwd='+str(ROOT)], 'results':['本项服务RC-PPO固定actor后的双critic预算决策验证；NAV-064依赖有效，当前52/67、NAV-065=false、旧G4 FAIL。','授权范围为已确认v2数值/一档保护/新选择数据/最多90名义回合和用户要求的已有最优模型；更新0、物理回合0、final_*封存。','允许新增隔离导航规则/执行入口和协议、原位修订计划准备/独立决策验收并保留旧定义/状态证据；不改模型、原目标/共同停止/beta或冻结v2。','先完成生产规则、基本测试和来源/版本绑定后验收NAV-065准备，再按依赖启动NAV-066；不以准备通过冒充科学可用。'], 'evidence':[str(HERE/'baseline.json'),str(HERE/'selection.json'),str(HERE/'validation-before.log')], 'remaining_work':['按批准登记新版协议、实现实际预算调用链并基本验证，完成准备验收后才冻结新独立决策并执行最多90回合。'], 'blockers':[]}
anchor = '  "blockers": ["草案尚未激活；按legacy不确定决策规则及主计划，验收/来源/独立数据/资源的具体修改须最终确认，当前不执行新额度或依赖任务。"]\n}\n      ],\n   "test_reports": ['
new = anchor.replace('\n}\n      ],','\n},\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n      ],')
patch = '*** Begin Patch\n*** Update File: '+str(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')+'\n@@\n'+'\n'.join('-'+s for s in anchor.splitlines())+'\n'+'\n'.join('+'+s for s in new.splitlines())+'\n*** End Patch\n'
(HERE/'opening.patch').write_text(patch)
print(json.dumps({'host':baseline['host'],'v2_files':len(v2),'selected':[{'method':x['identity']['variant'],'seed':x['identity']['seed'],'version':x['version'],'false_permits':x['dev_false_feasible'],'false_rejections':x['dev_false_infeasible'],'rmae':x['dev_combined_relative_mae']} for x in selected]},ensure_ascii=False,indent=2))
