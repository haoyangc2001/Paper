"""Record terminal evidence without rerunning any simulator or altering a gate."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

HERE = Path(__file__).resolve().parent
PLAN = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
REPORT = NAV/'reports/nav067_recovery_v3_execution_review.json'
assert os.environ.get('TMUX_PANE'), 'Review must execute in a recorded tmux pane.'
review_tmux = subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],'-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip()
assert review_tmux.startswith('nav064-r14-repair-20261115:'), review_tmux
assert not REPORT.exists()
started = json.loads((HERE/'COORDINATOR_STARTED.json').read_bytes())
assert not Path('/proc/'+str(started['pid'])).exists(), 'Coordinator remains live; observe the same run.'
finished = json.loads((HERE/'FINISHED.json').read_bytes()) if (HERE/'FINISHED.json').exists() else None
held = json.loads((HERE/'HELD.json').read_bytes()) if (HERE/'HELD.json').exists() else None
assert (finished is None) != (held is None), 'Exactly one authoritative terminal outcome is required.'
manifest, _, _ = read_manifest()
baseline = json.loads((HERE/'before-plan.json').read_bytes())
current = json.loads(PLAN.read_bytes())
state = {t['id']:t['status'] for t in current['tasks']}
assert sum(state.values()) == 54 and not state['NAV-067'] and not state['NAV-043']
audits, entries = [], []
for slot in json.loads((NAV/'config/nav067_startup_recovery_v3.json').read_bytes())['slots']:
    level = slot['level']
    launch = HERE/('launch-'+level+'.json')
    if not launch.exists(): continue
    launch_value = json.loads(launch.read_bytes())
    assert not Path('/proc/'+str(launch_value['pid'])).exists()
    out = OUTPUT/slot['relative_output']
    files = {str(p):sha(p) for p in out.rglob('*') if p.is_file()}
    entry = {'level':level,'launch':str(launch),'launch_sha256':sha(launch),
        'exit':json.loads((HERE/('exit-'+level+'.json')).read_bytes()),
        'artifacts':files,'entry_ERROR':None}
    if (out/'ERROR.json').exists(): entry['entry_ERROR'] = json.loads((out/'ERROR.json').read_bytes())
    entries.append(entry)
    audit = HERE/('audit-'+level+'.json')
    if audit.exists(): audits.append(json.loads(audit.read_bytes()))
episodes = [r for a in audits for r in a['episodes']]
actual_results = []
for entry in entries:
    for name in entry['artifacts']:
        if name.endswith('/result.json'): actual_results.append(json.loads(Path(name).read_bytes()))
saved_reviews = []
for entry in entries:
    level = entry['level']
    attempt = 'attempt2' if level == 'L1_empty' else 'direct'
    for method in ('uncalibrated','full'):
        trial = OUTPUT/level
        if attempt != 'direct': trial = trial/attempt
        trial = trial/method
        if str(trial/'result.json') not in entry['artifacts']: continue
        review_path = HERE/('main-agent-'+level+'-'+method+'.json')
        if not review_path.exists():
            log_path = HERE/('main-agent-'+level+'-'+method+'.log')
            command = [sys.executable,'-B',str(HERE/'review_saved_episode.py'),level,attempt,method]
            with log_path.open('x') as stream:
                checked = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            if checked.returncode != 0:
                write_new(review_path,{'status':'FAILED_SAVED_EPISODE_TRACE_REVIEW',
                    'level':level,'method':method,'trial':str(trial),'returncode':checked.returncode,
                    'log':str(log_path),'new_simulation_starts':0,'training_updates':0,'W_queries':0})
        value = json.loads(review_path.read_bytes())
        assert value['level'] == level and value['method'] == method and value['trial'] == str(trial)
        saved_reviews.append(value)
assert len(saved_reviews) == len(actual_results), 'Every saved result requires its own offline review.'
review_failures = [r for r in saved_reviews if r['status'] != 'PASS_SAVED_EPISODE_TRACE_ONLY']
runtime_reviews = []
for entry in entries:
    runtime_path = next((p for p in entry['artifacts'] if p.endswith('/runtime.json')),None)
    if runtime_path is None: continue
    level = entry['level']
    review_path = HERE/('runtime-review-'+level+'.json')
    if not review_path.exists():
        with (HERE/('runtime-review-'+level+'.log')).open('x') as stream:
            checked = subprocess.run([sys.executable,'-B',str(HERE/'review_runtime.py'),level],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if checked.returncode != 0:
            write_new(review_path,{'status':'FAILED_SAVED_RUNTIME_READBACK_REVIEW','level':level,
                'runtime':runtime_path,'returncode':checked.returncode})
    value = json.loads(review_path.read_bytes())
    assert value['level'] == level and value['runtime'] == runtime_path
    runtime_reviews.append(value)
runtime_failures = [r for r in runtime_reviews if r['status'] != 'PASS_SAVED_RUNTIME_READBACK_ONLY']
observations = [{'map_id':r['map_id'],'method':r['method'],'budget_mode':r['budget_mode'],
    'label':r['label'],'duration_s':r['t_s'],'safety_y':r['safety_y'],
    'terminal_goal_distance_m':r['terminal']['goal_distance_m'],
    'common_budget_J':r['z0_J'],'actual_joint_work_J':r['measured_joint_work_J'],
    'actual_budget_overrun_J':max(0.,r['measured_joint_work_J']-r['z0_J']),
    'actual_budget_overrun_fraction':max(0.,r['measured_joint_work_J']/r['z0_J']-1.),
    'internal_model_cost_J':r['internal_model_cost_J'],
    'internal_safe_and_budget_feasible':r['internal_safe_and_budget_feasible'],
    'terminal_events':r['terminal']['events'],
    'selected_budget_performance_evidence':False} for r in actual_results]
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
report = {'task':'NAV-067','timestamp':now,
    'status':('HELD_SAVED_RUNTIME_REVIEW' if runtime_failures else 'HELD_SAVED_TRACE_REVIEW' if review_failures else
        'PILOT_TRACE_PASS_NAV067_INCOMPLETE' if finished else 'HELD_RECOVERY_ATTEMPT'),
    'run_git_head':started['git_head'],'amendment_sha256':started['amendment_sha256'],
    'manifest_sha256':sha(OUTPUT/'manifest.json'),'protected_hashes_checked':manifest['binding']['protected_hashes_checked'],
    'entries':entries,'trace_audited_episodes':len(episodes),'saved_episode_results':len(actual_results),
    'episodes':episodes,'held':held,'training_updates':0,'W_queries':0,'automatic_retries':0,
    'descriptive_physical_outcomes':observations,
    'main_agent_saved_episode_reviews':saved_reviews,
    'main_agent_saved_episode_review_source_sha256':sha(HERE/'review_saved_episode.py'),
    'main_agent_saved_episode_review_failures':review_failures,
    'main_agent_saved_runtime_reviews':runtime_reviews,
    'main_agent_saved_runtime_review_source_sha256':sha(HERE/'review_runtime.py'),
    'main_agent_saved_runtime_review_failures':runtime_failures,
    'performance_evidence':False,'NAV_067_status':False,'NAV_043_started':False,
    'criterion_disposition':[
        {'criterion':1,'state':('saved runtime readback review failed' if runtime_failures else
            'saved config/material/mass/event/collider pose-enabled readbacks reviewed; controlled physical responses pending')},
        {'criterion':2,'state':('saved-episode offline review failed' if review_failures else
            'six-episode trace audit passed' if len(episodes)==6 else 'incomplete')},
        {'criterion':3,'state':'basic counterexamples only; actual controlled cases unallocated'},
        {'criterion':4,'state':'this independent review preserves old clean/pilot reports and all failures'},
        {'criterion':5,'state':'same-configuration repeatability unallocated'},
        {'criterion':6,'state':'physical applicability and method performance remain NAV-043 and later'}],
    'limitations':['Integration common budgets are not W-selected budgets or independent physical performance trials.',
        'Controlled actual cases and same-configuration repetitions were not part of this confirmation.',
        'No actual-case or repeatability evidence is inferred from passing basic tests.',
        'Old scientific G4/efficiency failures and original L1 attempt1 remain unchanged.']}
write_new(REPORT,report)
summary = ('NAV-067已保存证据的离线核验未通过：保留原始证据，未增加物理试验' if review_failures or runtime_failures else
           'NAV-067一次恢复结束：6回合接线轨迹核验完成，受控/重复验收尚未完成' if finished
           else 'NAV-067一次恢复已停止：保留真实运行故障，未自动重试或继续后续地图')
task = next(t for t in current['tasks'] if t['id']=='NAV-067')
task['work_records'].append({'timestamp':now,'summary':summary,
    'actions':['按用户确认执行v3批准登记及v4生产入口；先完成本地阶段提交再启动，在原tiger11/tmux %587运行。',
        '逐level核对真实runtime、退出/ERROR/FINISHED、命令观测、时钟、子步、复位及机械功；基础设施或轨迹异常即停，不以进程退出0放行。',
        '主agent对每个已保存回合运行离线原始轨迹核验，已有核验不重跑；新增仿真0，失败单列且不覆盖原始数据。',
        '复核295项保护身份、全部已有任务字段和历史记录；保留原attempt1及PROPOSED提案，新增独立实际审阅报告。',
        '首次提交被Git忽略日志规则拦截，没有启动仿真；保留失败日志后以明确路径纳入证据并成功提交。'],
    'files_changed':[str(REPORT.relative_to(ROOT)),str(PLAN.relative_to(ROOT)),'CaMRA/navigation/docs/implementation_handoff.md',str(HERE.relative_to(ROOT))],
    'commands':['tiger11/tmux nav064-r14-repair-20261115:nav067-clean-v2 @587/%587; absolute Python3.11.15 -B recovery-v3-execution/launch.py',
        'Terminal review tmux: '+review_tmux+'; absolute interpreter '+sys.executable,
        '同环境finish_review.py与legacy validate_worklog.py --mode update'],
    'results':['真实启动地图数'+str(len(entries))+'，保存回合结果'+str(len(actual_results))+'，接线trace核验回合'+str(len(episodes))+'；'+report['status'],
        '主agent已保存回合离线核验'+str(len(saved_reviews))+'项，失败'+str(len(review_failures))+'项。',
        '主agent运行时readback核验'+str(len(runtime_reviews))+'项，失败'+str(len(runtime_failures))+'项；不替代实际碰撞受控反例。',
        '描述性业务结果逐回合保存：到达/超时、安全历史、共同预算、实际机械功及超支；不作为W所选预算的性能验收。',
        '新增训练/W查询/自动重试均0；固定v2/模型/45秒/obs-v7/RC-PPO目标/旧校准及原清单保持。',
        'NAV-067保持false、54/67；本次不替代真实受控反例、同配置重复性或NAV-043闭环适用性。'],
    'evidence':[str(REPORT.relative_to(ROOT)),str(HERE.relative_to(ROOT))+'/COORDINATOR_STARTED.json',
        str(HERE.relative_to(ROOT))+'/FINISHED.json' if finished else str(HERE.relative_to(ROOT))+'/HELD.json'],
    'remaining_work':['按实际错误完成必要诊断/回修，任何后续物理重试仍受本轮失败策略约束。'] if held or review_failures or runtime_failures else ['明确真实受控反例及同配置重复性的独立额度，逐条完成NAV-067验收后再进入NAV-043。'],
    'blockers':(['已保存回合离线核验存在失败，须核对证据。'] if review_failures else [])+
        (['运行时readback核验存在失败，须核对证据。'] if runtime_failures else [])+
        (['本次恢复已按失败策略停止；不得自动重试。'] if held else [])+
        ['真实受控反例与同配置重复无独立额度。']})
# Preserve the original byte layout, changing only the current task's appended records.
text = (HERE/'before-plan.json').read_bytes().decode('utf-8')
offset = text.index('"id": "NAV-067"')
opening = text.index('[',text.index('"work_records"',offset))
old_records, length = json.JSONDecoder().raw_decode(text[opening:])
closing = opening+length-1
assert old_records == next(t for t in baseline['tasks'] if t['id']=='NAV-067')['work_records']
additional = task['work_records'][len(old_records):]
newline = '\r\n' if '\r\n' in text else '\n'
body = (','+newline).join(newline.join('        '+line for line in json.dumps(r,ensure_ascii=False,indent=2).splitlines()) for r in additional)
restored = text[:closing].rstrip(' \t\r\n')+','+newline+body+newline+'      '+text[closing:]
assert json.loads(restored) == current
PLAN.write_bytes(restored.encode('utf-8'))
for a,b in zip(baseline['tasks'],current['tasks']):
    assert {k:v for k,v in a.items() if k!='work_records'} == {k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
assert {k:v for k,v in baseline.items() if k!='tasks'} == {k:v for k,v in current.items() if k!='tasks'}
handoff = NAV/'docs/implementation_handoff.md'
old = handoff.read_text()
outcome_table = '| 地图 | 方法 | 结束原因 | 时间(s) | 安全历史y | 实际机械功(J) | 共同预算(J) | 实际超支 |\n| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
for r in observations:
    outcome_table += ('| '+r['map_id']+' | '+r['method']+' | '+r['label']+' | '+format(r['duration_s'],'.3f')+
        ' | '+str(r['safety_y'])+' | '+format(r['actual_joint_work_J'],'.3f')+' | '+format(r['common_budget_J'],'.3f')+
        ' | '+format(r['actual_budget_overrun_fraction'],'.1%')+' |\n')
note = ('## '+now[:10]+' NAV-067已批准一次恢复的实际结果\n\n'+summary+'。本次保存'+str(len(actual_results))+'个实际回合结果，'+str(len(episodes))+'个通过接线轨迹核验。状态`'+report['status']+'`，不以退出码或基本测试代替实际证据；故障及原L1 attempt1全部保留。批准登记、原manifest和295项保护身份匹配，无训练/W查询/自动重试。\n\n'
    +outcome_table+'\n以上为共同预算接线pilot，非W所选预算的正式物理性能；超时/超支保留，不能写成安全可行成功，也不能据此证明所有策略均无解。\n\n'+
    '实际审阅见[本轮报告](../reports/nav067_recovery_v3_execution_review.json)和`.worklog-testing/NAV-067/recovery-v3-execution/`。NAV-067仍false、54/67；真实受控反例及同配置重复未获额外额度，NAV-043未执行。以下为此前时点。\n\n')
assert old.startswith('# 上层导航实施交接\n')
handoff.write_text(old.replace('# 上层导航实施交接\n','# 上层导航实施交接\n\n'+note,1))
command = [sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update']
check = subprocess.run(command,capture_output=True,text=True)
with (HERE/'final-plan-check.log').open('x') as stream: stream.write(check.stdout+check.stderr)
assert check.returncode == 0, check.stdout+check.stderr
print(json.dumps({'status':report['status'],'saved_results':len(actual_results),'trace_audited_episodes':len(episodes),
    'report':str(REPORT),'held':held,'episodes':episodes},ensure_ascii=False,indent=2),flush=True)
