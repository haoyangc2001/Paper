"""Document the old clean claim's limits; never runs a simulator."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'src'),str(ROOT/'CaMRA/calibration/src')]
from nav067_sources import load, sha, write_new
import run_nav067_remaining_recovery_v2 as V2

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def patch(content):
    subprocess.run(['apply_patch'],input=content,text=True,cwd=ROOT,check=True)

assert os.environ.get('TMUX_PANE')=='%599'
V2.context(initial=False)
plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan_text=plan_path.read_text(); plan=json.loads(plan_text)
task=next(t for t in plan['tasks'] if t['id']=='NAV-067')
assert not task['status'] and sum(t['status'] for t in plan['tasks'])==54
old_path=NAV/'reports/nav042_online_loop.json'
pilot_path=NAV/'reports/nav067_recovery_v3_execution_review.json'
old=load(old_path); pilot=load(pilot_path)
source_paths=[NAV/'scripts/evaluate_navigation.py',NAV/'src/camra_navigation/envs/physical.py',
    NAV/'src/camra_navigation/envs/clean_v2.py',NAV/'src/camra_navigation/envs/clean_v3.py',
    NAV/'src/camra_navigation/envs/physical_v2.py',NAV/'scripts/nav067_simulation_v3.py',
    old_path,pilot_path,HERE/'actual-controlled-review.json',
    *[NAV/'reports'/f'nav042_{level}.json' for level in ('L1_empty','L2_single_detour','L3_multi_path')]]
before={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
assert len(old['episodes'])==6 and len(pilot['descriptive_physical_outcomes'])==6
assert old['limitations'][1]=='no perturbation profile; clean-physics only'
trace_checks=[]
for trace in old['trace_files']:
    assert sha(trace['path'])==trace['sha256'],trace['path']
    trace_checks.append(trace)
controlled=load(HERE/'actual-controlled-review.json')
assert len(controlled['controlled'])==6
report_path=NAV/'reports/nav042_report_amendment_nav067_v1.json'
report={'schema':'camra-nav042-clean-claim-amendment-nav067-v1','at':now(),'discovered_in_task':'NAV-067',
    'correction_id':'NAV-067-CLEAN-CLAIM-AMENDMENT-V1','status':'AMENDMENT_DOCUMENTED_NAV067_REPETITIONS_PENDING',
    'old_report_retained':str(old_path.relative_to(ROOT)),'old_report_sha256':sha(old_path),
    'old_claim':'no perturbation profile; clean-physics only',
    'corrected_interpretation':'The old runner disables observation corruption but does not establish that inherited startup/reset/interval randomization is disabled. Its saved wiring report is not sufficient evidence of the current clean profile or complete 200 Hz events.',
    'specific_source_findings':[
        {'path':'scripts/evaluate_navigation.py','lines':[266,271,299,302],
            'finding':'Constructs the training environment, sets seed and observation corruption, then gym.make. No complete event/material/mass/command-resampling clean override or matching runtime readback is established.'},
        {'path':'scripts/evaluate_navigation.py','lines':[133,139,142,148],
            'finding':'Calls gym/env.step and reads fall state after the 50 Hz policy step. This does not give the new event monitor an actual pose at every 200 Hz substep.'},
        {'path':'src/camra_navigation/envs/physical.py','lines':[494,514,532],
            'finding':'Allows absent command observation to fall back to applied rows and updates navigation geometry after the 10 Hz interval. The old successful command comparisons do not prove a missing-observation rejection path or complete 200 Hz event coverage.'}],
    'what_is_not_claimed':'No retrospective estimate of how much randomization occurred or which old trajectory was changed by it. Missing actual evidence is not filled with expected values.',
    'corrected_production_chain':['scripts/evaluate_navigation_v2.py','scripts/nav067_simulation_v3.py',
        'src/camra_navigation/envs/clean_v3.py -> clean_v2.py IsaacSubstepAdapter',
        'src/camra_navigation/envs/physical_v2.py SubstepNavigation',
        'evaluation.recorder offline energy recomputation'],
    'actual_new_evidence':{'six_original_corrected_pilot_report':str(pilot_path.relative_to(ROOT)),
        'six_original_corrected_pilot_sha256':sha(pilot_path),'pilot_runtime_reviews':pilot['main_agent_saved_runtime_reviews'],
        'pilot_trace_count':pilot['trace_audited_episodes'],'controlled_cases':controlled['controlled'],
        'controlled_kind':'C01-C05 deliberate fixture pose writes with actual post-physics readback; C06 native filtered robot-obstacle contact after initial placement.',
        'repetitions':'Still running in the original approved allocation; not part of a completed repetition claim in this amendment.'},
    'old_six_episodes':[{'level':e['level'],'method':e['method'],'label':e['label'],
        'candidate_id':e['candidate_id'],'actor_sha256':e['actor_sha256'],'time_s':e['t_s'],
        'measured_joint_work_J':e['measured_J'],'budget_J':e['z0_J']} for e in old['episodes']],
    'new_corrected_pilot_six_episodes':pilot['descriptive_physical_outcomes'],
    'comparison_limits':['Old and new candidates/actor identities differ.',
        'The new candidates use the original frozen 45 second obs-v7 task, revised budget/time/terminal contracts and separately registered common-budget references.',
        'New runtime clean settings and event/partial-cycle accounting differ from the old runner.',
        'Old versus new differences are not same-configuration repetition and cannot all be causally attributed to clean settings.',
        'The new six pilot episodes all exceed the common budget; full reaches L1/L3 and times out on L2, uncalibrated times out on all three. These are wiring outcomes, not W-selected physical performance.'],
    'affected_tasks':['NAV-042 historical interpretation','NAV-067','NAV-043','NAV-044','NAV-045','NAV-046','NAV-047','NAV-048'],
    'affected_gates':['G5 future applicability/physical release','G6 future final audit'],
    'status_disposition':'Keep original NAV-042 reports/status as historical wiring evidence; the rebased chain consumes NAV-067 actual clean evidence. NAV-067 stays false until its remaining criteria are verified. NAV-043/G5 have not been executed or passed.',
    'historical_FAIL_preserved':True,'original_report_and_source_files_unchanged':True,
    'source_hashes':before,'old_trace_identity_checks':trace_checks,'training_updates':0,'W_queries':0,
    'new_simulator_applications':0,'new_physical_episodes':0,'final_data_access':False}
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
assert not report_path.exists()
patch('*** Begin Patch\n*** Add File: '+str(report_path)+'\n'+''.join('+'+s+'\n' for s in text.splitlines())+'*** End Patch\n')
doc_path=NAV/'docs/nav042_clean_claim_amendment_20261119.md'
doc='''旧NAV-042报告中的“clean-physics only”需要收窄解释。旧入口关闭了观测噪声，但没有完整证明继承的训练随机化、摩擦/质量、复位和命令重采样都已关闭；事件读取也不足以证明每个200Hz子步均被正确处理。旧六回合仍保留为当时的接线证据，不能作为本轮clean或全程安全证据。

NAV-067新版入口已保存实际event manager、摩擦/质量/地面参数、复位、命令观测和200Hz子步记录。此前修正pilot六回合及本次六项实际受控检查均有核验；同配置导航重复仍在原批准分配中执行，所以NAV-067尚未结案。

旧、新pilot的actor、观测与预算/终止契约也有变化，不能把结果差异都归因于关闭扰动。两批也不构成同配置重复。新的共同预算pilot仍全部超预算；完整方法到达L1/L3、L2超时，无校准三图超时，不将接线通过写成导航性能通过。

本补充不改写旧报告、历史失败或冻结v2。NAV-043仍要验证新版策略的物理可用性和原裕度适用性，G5/G6没有因此通过。具体来源/hash、逐条勘误、旧新各六条结果及下游影响见[机器补充报告](../reports/nav042_report_amendment_nav067_v1.json)。本轮没有增加训练、查询或物理执行。
'''
assert not doc_path.exists()
patch('*** Begin Patch\n*** Add File: '+str(doc_path)+'\n'+''.join('+'+s+'\n' for s in doc.splitlines())+'*** End Patch\n')
record={'timestamp':now(),'summary':'NAV-067补齐旧NAV-042 clean声明的独立amendment：不覆盖原报告，重复仍在运行',
    'actions':['逐项核对旧入口、六原始trace身份及新版clean/子步生产链，明确旧报告只能支持当时接线，不能支持本轮clean/200Hz全程事件结论。',
        '新增独立JSON与通俗说明，绑定原报告/hash、实际新版六pilot/六受控证据、版本差异及受影响任务/门禁。'],
    'files_changed':[str(report_path.relative_to(ROOT)),str(doc_path.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),str(plan_path.relative_to(ROOT))],
    'commands':['tiger11/tmux nav064-r14-repair-20261115:nav067-evidence-close %599；'+sys.executable+' -B '+str(Path(__file__))],
    'results':['amendment按原NAV-067第4条验收补齐来源与解释；原报告/六trace/冻结v2未改，旧科学FAIL保持。',
        '旧新差异包含actor/观测/预算与终止契约，不都归因于clean；不替代同配置重复。',
        '六导航重复仍在执行，NAV-067=false、54/67，NAV-043/G5未执行；新增物理/训练/W查询0。'],
    'evidence':[str(report_path.relative_to(ROOT)),str(doc_path.relative_to(ROOT)),str(pilot_path.relative_to(ROOT)),
        str((HERE/'actual-controlled-review.json').relative_to(ROOT))],
    'remaining_work':['同一分配六导航重复结束后，审阅完整原/新12回合和偏差，完成局部及全局双重验收后再判NAV-067。'],
    'blockers':[]}
old_record='\n'.join('        '+s for s in json.dumps(task['work_records'][-1],ensure_ascii=False,indent=2).splitlines())
new_record='\n'.join('        '+s for s in json.dumps(record,ensure_ascii=False,indent=2).splitlines())
assert plan_text.count(old_record)==1
patch('*** Begin Patch\n*** Update File: '+str(plan_path)+'\n@@\n'+''.join('-'+s+'\n' for s in old_record.splitlines())+''.join('+'+s+'\n' for s in (old_record+',\n'+new_record).splitlines())+'*** End Patch\n')
after=load(plan_path)
for a,b in zip(plan['tasks'],after['tasks']):
    if a['id']=='NAV-067':
        assert b['work_records'][:-1]==a['work_records']
        assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    else:
        assert a==b
assert before=={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
V2.context(initial=False)
validator=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update']
subprocess.run(validator,cwd=ROOT,check=True)
write_new(HERE/'clean-amendment-verification.json',{'at':now(),'status':'AMENDMENT_SOURCES_AND_HISTORY_PASS_NAV067_STILL_FALSE',
    'old_source_and_report_hashes_unchanged':True,'old_trace_hashes_verified':len(trace_checks),
    'all_prior_tasks_and_records_preserved':True,'plan_status_changes':0,'new_simulator_applications':0,
    'amendment_sha256':sha(report_path)})
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
paths=[str(p.relative_to(ROOT)) for p in (report_path,doc_path,plan_path,Path(__file__),HERE/'clean-amendment-verification.json')]
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','docs(nav): amend historical clean claim with actual evidence limits'],cwd=ROOT,check=True)
print('CLEAN_AMENDMENT_COMMITTED_NO_NEW_PHYSICS',flush=True)
