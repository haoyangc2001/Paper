"""Close NAV-067 only after the original allocation's saved audit has passed."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import review_saved as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAV = ROOT/'CaMRA/navigation'
PLAN = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE') == '%599'
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
review = R.load(HERE/'actual-full-review.json')
watch = R.load(HERE/'saved-review-watcher-exit.json')
assert watch['review_exit_code'] == 0
assert review['status'] == 'ACTUAL_CONTROLLED_AND_REPEAT_TRACE_REVIEW_PASS_PENDING_TASK_CLOSURE'
assert len(review['controlled']) == 6 and len(review['paired_repetitions']) == 6
assert not Path('/proc/2588869/cmdline').exists(), 'original coordinator has not exited'
assert not (R.V1.EXPECTED_ROOT/'RECOVERY_V2_HELD.json').exists()
proposal, execution, manifest, prereg, _ = R.V2.context(initial=False)
assert review['approval_sha256'] == R.sha(R.V2.APPROVAL)
assert review['proposal_sha256'] == R.sha(R.V2.PROPOSAL)
assert review['review_source_sha256'] == R.sha(HERE/'review_saved.py')
assert review['controlled'] == R.load(HERE/'actual-controlled-review.json')['controlled']
for artifact in review['saved_artifacts']:
    path = Path(artifact['path'])
    assert path.stat().st_size == artifact['bytes'] and R.sha(path) == artifact['sha256']
old_report = NAV/'reports/nav067_recovery_v3_execution_review.json'
old = R.load(old_report)
for entry in old['entries']:
    for path, digest in entry['artifacts'].items():
        assert R.sha(path) == digest, path
amendment = NAV/'reports/nav042_report_amendment_nav067_v1.json'
amend = R.load(amendment)
assert R.sha(ROOT/amend['old_report_retained']) == amend['old_report_sha256']
assert R.sha(old_report) == amend['actual_new_evidence']['six_original_corrected_pilot_sha256']
assert review['resources']['kit_starts_including_failure'] == 10
assert review['resources']['reserved_environment_ticks_including_failure'] == 57830
assert review['resources']['cumulative_gpu_wall_s'] <= 14400.
assert all(pair['runs_per_configuration'] == 2 and not pair['general_reliability_claim']
           for pair in review['paired_repetitions'])
for name in ('close_verified_task.py', 'review_saved.py', 'review_finished_map.py'):
    ast.parse((HERE/name).read_text(),filename=name)

def patch(text):
    subprocess.run(['apply_patch'],input=text,text=True,cwd=ROOT,check=True)

def add(path, content):
    assert not path.exists(), str(path)
    patch('*** Begin Patch\n*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in content.splitlines())+'*** End Patch\n')

def prepend(path, title, content):
    assert path.read_text().startswith(title+'\n\n')
    patch('*** Begin Patch\n*** Update File: '+str(path)+'\n@@\n '+title+'\n \n'+
          ''.join('+'+line+'\n' for line in (content+'\n\n').splitlines())+'*** End Patch\n')

timestamp = datetime.now().astimezone().isoformat()
criteria = [
    {'criterion':1,'status':'PASS','evidence':'Original three and repeated three runtime readbacks: reset-only events, clean friction/mass, zero reset perturbations, no noise/resampling/autoreset; 295 frozen identities verified.'},
    {'criterion':2,'status':'PASS','evidence':'Six original and six repeated saved traces pass genuine commands, 10/50/200 Hz, warmup/reset, terminal snapshots and mechanical-work recomputation; actual partial cycle retained.'},
    {'criterion':3,'status':'PASS','evidence':'Six actual controlled cases plus saved-state production replay: intermediate collision, dangerous history, turning footprint, simultaneous events, asynchronous termination/masking and native filtered contact. Written-pose fixtures explicitly separated from native response and navigation performance.'},
    {'criterion':4,'status':'PASS','evidence':'Independent original clean-claim amendment retained; this supplement records completed repetitions. Original reports and scientific FAIL unchanged; old/new candidate and contract changes preclude causal clean-only comparison.'},
    {'criterion':5,'status':'PASS','evidence':'Authorized six repeats complete; each of six configurations has denominator n=2. Descriptive paired time/work/position differences reported without general reliability or bitwise-determinism claims.'},
    {'criterion':6,'status':'PASS_ENGINEERING_SCOPE_ONLY','evidence':'Common-budget wiring evidence only. NAV-043 applicability and G5 remain unexecuted/unpassed; all timeouts and budget violations retained.'},
]
global_review = {
    'same_RC_PPO_mathematical_objective':True,
    'selected_actor_dual_critic_observation_action_budget_identity':manifest['candidates'],
    'frozen_controller_and_calibration_identity':manifest['binding'],
    'actual_production_chain':['run_nav067_remaining_recovery_v2.py',
        'run_nav067_remaining_validation.repetition -> nav067_simulation_v3.simulate',
        'clean_v3.IsaacSubstepAdapter -> clean_v2 policy50Hz/physics200Hz',
        'physical_v2.SubstepNavigation -> evaluation.recorder'],
    'same_45s_obs_v7_and_positive_joint_mechanical_work_J':True,
    'old_scientific_G4_and_budget_efficiency_FAIL_preserved':True,
    'no_final_data_access_no_training_no_W_queries':True,
    'NAV_043_and_G5_unexecuted':True,
    'reference_code_manuscripts_and_frozen_inputs_unchanged':True,
}
result_rows = []
for pair in review['paired_repetitions']:
    for trial_kind in ('baseline', 'repeat'):
        row = pair[trial_kind]
        result_rows.append({'level':pair['level'],'method':pair['method'],'trial_kind':trial_kind,
            'label':row['label'],'t_s':row['t_s'],'safety_y':row['safety_y'],
            'actual_work_J':row['actual_work_J'],'internal_model_cost_J':row['internal_model_cost_J'],
            'budget_J':row['budget_J'],'over_budget':row['actual_work_J']>row['budget_J'],
            'over_budget_fraction':row['actual_work_J']/row['budget_J']-1.,
            'actual_safe_timely_budget_success':row['label']=='goal_reached' and row['safety_y']==0.
                and row['actual_work_J']<=row['budget_J'],
            'performance_evidence':False})
assert len(result_rows) == 12
report_path = NAV/'reports/nav067_engineering_acceptance_v1.json'
report = {'schema':'camra-nav067-engineering-acceptance-v1','at':timestamp,'task':'NAV-067',
    'status':'PASS_ENGINEERING_INTEGRATION_NOT_NAVIGATION_PERFORMANCE',
    'acceptance_criteria':criteria,'global_review':global_review,
    'full_actual_audit_path':str((HERE/'actual-full-review.json').relative_to(ROOT)),
    'full_actual_audit_sha256':R.sha(HERE/'actual-full-review.json'),
    'original_pilot_report_sha256':R.sha(old_report),
    'historical_clean_amendment_path':str(amendment.relative_to(ROOT)),
    'historical_clean_amendment_sha256':R.sha(amendment),
    'clean_amendment_supplement':'Six same-configuration repeats now reviewed; original amendment/report remains immutable.',
    'actual_navigation_results':result_rows,
    'controlled_cases':review['controlled'],'paired_repetitions':review['paired_repetitions'],
    'resources':review['resources'],'raw_artifact_index':review['saved_artifacts'],
    'all_failures_retained':True,'NAV_067_status':True,'completed_plan_tasks':55,'total_plan_tasks':67,
    'NAV_043_executed':False,'G5_passed':False,'training_updates':0,'W_queries':0,
    'known_limits':['Common-budget k=1 wiring pilots are not W-selected physical performance.',
        'Six configurations with n=2 each do not establish general physical reliability.',
        'Old G4 FAIL, independent budget-efficiency FAIL and accepted limitation remain separate.',
        'NAV-043 needs its own applicability evidence and registered resources; this recovery approval excludes it.',
        'Controlled written-pose cases and native contact are not natural navigation success episodes.'],
    'remote_push':False}
add(report_path,json.dumps(report,ensure_ascii=False,indent=2)+'\n')
doc_path = NAV/'docs/nav067_engineering_acceptance_20261119.md'
lines = [
    'NAV-067已通过工程接入验收，计划55/67，下一项NAV-043尚未执行。固定v2、新版两候选、45秒/obs-v7和RC-PPO原理保持，原科学FAIL保留。',
    '',
    '六项实际受控检查通过：中间碰撞、危险历史、转向扫掠、同刻事件、异步结束和原生接触。前五项为人为位姿写入后的实际读回，第六项是初次放置后的原生接触响应；均不计作自然导航成功。',
    '',
    '原六条pilot和本次六条重复均完成命令真实观测、10/50/200Hz、复位、终止、部分周期及机械功复算。每种配置两次，只报告下表差异；不宣称未知场景稳定或物理逐位一致。',
    '',
    '| 地图 | 方法 | 原/重复结束 | 时间差(s) | 机械功差(J) | 最大位置差(m) |',
    '| --- | --- | --- | ---: | ---: | ---: |']
for pair in review['paired_repetitions']:
    delta = pair['delta_repeat_minus_baseline']
    lines.append('| '+pair['level']+' | '+pair['method']+' | '+pair['baseline']['label']+'/'+pair['repeat']['label']+
        ' | '+format(delta['t_s'],'.9g')+' | '+format(delta['actual_work_J'],'.9g')+
        ' | '+format(pair['position_difference_max_m'],'.9g')+' |')
lines += ['', '这12条是共同预算接线记录。到达、超时、安全历史和超预算逐条保留在报告中；工程验收不代表预算内导航性能通过，也不替NAV-043/G5放行。',
    '', '旧NAV-042的clean声明修正见独立amendment，本次补充重复已完成；旧报告/失败/模型均不覆盖，旧新差异不能全部归因于关闭扰动。',
    '', '本次恢复累计启动'+str(review['resources']['kit_starts_including_failure'])+'次应用（含原失败），单GPU墙钟'+format(review['resources']['cumulative_gpu_wall_s'],'.3f')+'秒，环境子步预留57830，未超10应用/4小时/60000上限；预留不等于实际消耗。',
    '', '后续应先检查实际部署策略的跟踪、45秒时限、能量前缀和原裕度适用性。当前恢复授权不含NAV-043新执行，不自动增加物理回合、训练或预算查询。',
    '', '[完整验收和结果](../reports/nav067_engineering_acceptance_v1.json)；原始大轨迹保留在登记实验目录，Git保存校验索引。']
add(doc_path,'\n'.join(lines)+'\n')

raw = PLAN.read_bytes()
before = json.loads(raw)
task = next(t for t in before['tasks'] if t['id']=='NAV-067')
assert not task['status'] and sum(t['status'] for t in before['tasks'])==54
record = {'timestamp':timestamp,'summary':'NAV-067逐条与全局验收完成：真实clean/200Hz事件/6受控/6同配重复通过，仅工程接入结案',
    'actions':['审阅原协调器终态、既有只读审阅器全量报告及原始结果；核验6实际受控、原6加新6导航轨迹和每配置n=2差异，不凭退出码放行。',
        '复核295项保护输入及全部封存源/证据/旧六轨迹hash；原数学目标、固定v2/校准/两模型/45秒obs-v7/200-50-10Hz/J均保持。',
        '补充独立clean声明amendment的最终重复结果，不覆盖旧报告；逐条登记工程验收和全局实际调用链/接口版本及下游限制。',
        '只将NAV-067标true，NAV-043和G5保持未执行/未通过；提交实际证据和索引，不推送。'],
    'files_changed':[str(report_path.relative_to(ROOT)),str(doc_path.relative_to(ROOT)),str(PLAN.relative_to(ROOT)),
        'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/README.md',str(HERE.relative_to(ROOT))+'/'],
    'commands':['tiger11/tmux nav064-r14-repair-20261115:nav067-evidence-close %599；cwd='+str(ROOT)+'；CUDA_VISIBLE_DEVICES=6 '+sys.executable+' -B '+str(Path(__file__)),
        'Original read-only watcher: '+sys.executable+' -B '+str(HERE/'review_saved.py')+' --full',
        'legacy validate_worklog.py <current NAV plan> --mode update; explicit local git commit, no push'],
    'results':['六项验收及全局一致性通过；NAV-067 true、55/67；这是工程接入，不是方法性能或G5通过。',
        '原六pilot加六重复完整保存，全部业务失败和超支如实记录，每配置两次差异只作描述；原科学FAIL不改。',
        '批准恢复含原失败共10应用，墙钟和预留环境步均在上限内；训练/更新/W查询/final访问/NAV-043执行0。'],
    'evidence':[str(report_path.relative_to(ROOT)),str((HERE/'actual-full-review.json').relative_to(ROOT)),str((HERE/'saved-review-watcher-exit.json').relative_to(ROOT)),
        str(amendment.relative_to(ROOT)),str(R.V1.EXPECTED_ROOT/'RECOVERY_V2_FINISHED.json')],
    'remaining_work':['本项工程缺口关闭；下一项NAV-043按45秒新版候选/预算规则和自身适用性/配额准备，未获其执行额度不运行。'],
    'blockers':[]}
old = '\n'.join('        '+line for line in json.dumps(task['work_records'][-1],ensure_ascii=False,indent=2).splitlines())
new = '\n'.join('        '+line for line in json.dumps(record,ensure_ascii=False,indent=2).splitlines())
assert PLAN.read_text().count(old)==1
patch('*** Begin Patch\n*** Update File: '+str(PLAN)+'\n@@\n'+''.join('-'+line+'\n' for line in old.splitlines())+''.join('+'+line+'\n' for line in (old+',\n'+new).splitlines())+'*** End Patch\n')
verification = 'PASS_ENGINEERING_INTEGRATION_NOT_NAVIGATION_PERFORMANCE; '+str(report_path.relative_to(ROOT))+' SHA256 '+R.sha(report_path)+'; six criteria and global chain reviewed; n=2 per configuration; NAV-043/G5 pending; historical scientific FAIL retained.'
patch('*** Begin Patch\n*** Update File: '+str(PLAN)+'\n@@\n       "id": "NAV-067",\n       "phase": "预算与闭环",\n       "goal": "修正clean物理配置与子步事件采集，用新版候选重验在线接入",\n-      "status": false,\n+      "status": true,\n@@\n-      "verification_result": "",\n-      "planned_outputs": [\n-        "scripts/evaluate_navigation.py、src/camra_navigation/envs/physical.py及隔离clean配置",\n+      "verification_result": '+json.dumps(verification,ensure_ascii=False)+',\n+      "planned_outputs": [\n+        "scripts/evaluate_navigation.py、src/camra_navigation/envs/physical.py及隔离clean配置",\n*** End Patch\n')
if raw.count(b'\r\n') and raw.count(b'\r\n') == raw.count(b'\n'):
    changed = PLAN.read_bytes().replace(b'\r\n',b'\n')
    PLAN.write_bytes(changed.replace(b'\n',b'\r\n'))
after = R.load(PLAN)
for a,b in zip(before['tasks'],after['tasks']):
    if a['id']=='NAV-067':
        assert a['work_records'] == b['work_records'][:-1]
        assert {k:v for k,v in a.items() if k not in ('work_records','status','verification_result')} == {k:v for k,v in b.items() if k not in ('work_records','status','verification_result')}
        assert b['status'] and b['verification_result']==verification
    else:
        assert a==b
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in after.items() if k!='tasks'}
assert sum(t['status'] for t in after['tasks'])==55
states={t['id']:t['status'] for t in after['tasks']}
assert next(t['id'] for t in after['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))=='NAV-043'
brief = 'NAV-067已逐条通过工程接入验收，计划1.4.1、55/67；6实际受控及原6加新6导航保存轨迹复核通过，每配置n=2只作描述。不是方法性能或G5通过，原科学FAIL、超时/超支和全部旧失败保留。下一项NAV-043尚未执行，当前恢复批准不含其新额度，不自动加跑。见[本次验收](CaMRA/navigation/docs/nav067_engineering_acceptance_20261119.md)。下方运行中/false为历史时点。'
prepend(ROOT/'AGENTS.md','# Agent 接手说明','## 最新结案（2026-11-19，NAV-067工程接入通过）\n\n'+brief)
prepend(NAV/'docs/implementation_handoff.md','# 上层导航实施交接','## 2026-11-19 NAV-067工程接入验收完成，下一项NAV-043未执行\n\n'+brief.replace('CaMRA/navigation/docs/nav067_engineering_acceptance_20261119.md','nav067_engineering_acceptance_20261119.md'))
prepend(NAV/'README.md','# 上层最小成本到达避障导航器',brief.replace('CaMRA/navigation/docs/nav067_engineering_acceptance_20261119.md','docs/nav067_engineering_acceptance_20261119.md'))
validated=subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update'],cwd=ROOT,capture_output=True,text=True)
assert validated.returncode==0,validated.stdout+validated.stderr
add(HERE/'closure-verification.json',json.dumps({'at':timestamp,'status':'NAV067_LOCAL_AND_GLOBAL_ACCEPTANCE_PASS',
    'all_prior_records_preserved':True,'only_task_status_change':'NAV-067 false -> true','completed':55,
    'next_eligible':'NAV-043','NAV_043_executed':False,'scientific_FAIL_retained':True,
    'validator':validated.stdout,'new_simulator_applications':0,
    'report_sha256':R.sha(report_path),'review_sha256':R.sha(HERE/'actual-full-review.json')},ensure_ascii=False,indent=2)+'\n')
paths=[str(report_path.relative_to(ROOT)),str(doc_path.relative_to(ROOT)),str(PLAN.relative_to(ROOT)),
    'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/README.md',
    *[str((HERE/name).relative_to(ROOT)) for name in ('close_verified_task.py','actual-full-review.json',
        'saved-review-watcher-exit.json','closure-verification.json')]]
assert not subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','test(nav): accept clean substep integration with repeated trace evidence'],cwd=ROOT,check=True)
print(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'NAV_067':True,'completed':55,'next':'NAV-043','NAV_043_executed':False},ensure_ascii=False),flush=True)
