from datetime import datetime
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PLAN = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
PY = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
load = lambda p:json.loads(Path(p).read_bytes())
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX') and sys.executable==PY
before = PLAN.read_text()
plan = json.loads(before)
tasks = {t['id']:t for t in plan['tasks']}
assert plan['document']['version']=='1.4.0' and sum(t['status'] for t in tasks.values())==53
assert tasks['NAV-065']['status'] and not tasks['NAV-066']['status'] and not tasks['NAV-067']['status']
with (HERE/'validation-before.txt').open('x') as out:
 subprocess.run([PY,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update'],stdout=out,stderr=subprocess.STDOUT,check=True)
regpath = NAV/'config/nav066_decision_prereg_v2.json'
reportpath = NAV/'reports/nav066_budget_decision_validation_v2.json'
reg, report = load(regpath), load(reportpath)
audit = load(ROOT/'.worklog-testing/NAV-065/best-decision-v2/final-audit.json')
assert audit['integrity_pass'] and not audit['scientific_gate'] and not report['pass']
assert report['registration_pin']==sha(regpath)
assert all(sha(p)==h for p,h in audit['source_hashes'].items())
assert all(sha(NAV/p)==h for p,h in reg['implementation_hashes'].items())
assert all(sha(p)==h for p,h in reg['protected_hashes'].items())
assert all(sha(c['checkpoint'])==c['checkpoint_sha256'] for c in reg['six_candidates'])
assert report['grid_episodes']==90 and report['independent_decisions']==18 and report['steps']==33531
assert report['updates']==0 and report['physical_episodes']==0 and not report['final_results_opened']
unc,full = (report['methods'][m] for m in ('uncalibrated','full'))
assert unc['pass'] and not full['pass']
for s in (unc,full):
 assert s['accepted']==s['correctly_accepted']==9 and s['accepted_failures']==s['rejected']==0
 assert s['feasible_map_recall']==1 and all(n==3 for n in s['level_accepted'].values())
assert {k for k,v in full['checks'].items() if not v}=={'budget_efficiency'}
failed = [r for r in full['details'] if not r['efficiency_pass']]
assert len(failed)==1 and failed[0]['map_id']=='l1_empty-s917001'
case = failed[0]
assert case['actual_min_index']==2 and case['selected_index']==4 and case['overspend_J']==0
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def write_new(p,value):
 with p.open('x') as out:
  json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False)
  out.write('\n')

write_new(HERE/'baseline.json',{'plan':plan,'plan_sha256':sha(PLAN),'head':head,'timestamp':now,
 'protected_hashes':reg['protected_hashes'],'implementation_hashes':reg['implementation_hashes'],
 'original_evidence_hashes':{str(reportpath):sha(reportpath),str(regpath):sha(regpath),**audit['source_hashes']}})
disposition = {
 'schema':'camra-navigation-stage-accepted-limitation-v1','created_at':now,'task':'NAV-066',
 'disposition':'ACCEPTED_WITH_LIMITATION','scientific_verdict':'FAIL_BUDGET_EFFICIENCY',
 'post_result_user_decision':True,
 'authorization':{'user_instruction':'那就按照你说的做，看一下后续任务都是什么 通俗简洁作答',
 'accepted_recommendation':'接受已知预算效率局限，允许研究流程进入下一阶段；原协议效率FAIL及案例保留，不写成全部指标达标或整个项目完成。'},
 'immutable_preregistration':{'path':str(regpath.relative_to(ROOT)),'sha256':sha(regpath)},
 'immutable_result':{'path':str(reportpath.relative_to(ROOT)),'sha256':sha(reportpath),'scientific_pass':False},
 'accepted_case':{**case,'allowed_one_step_budget_J':case['budget_J']/1.3*1.15,
  'extra_allocation_above_one_step_limit_J':case['budget_J']/1.3*.15,
  'explanation':'仅接受该图预算过保守：所选1.30、真实网格最低1.00，原一档效率上限1.15。多分配预算不是同额实际额外消耗。'},
 'required_measured_facts':{'each_method_selected_successes':'9/9','accepted_failures':0,'rejected':0,
  'feasible_map_recall':1.0,'each_level_successes':'3/3','uncalibrated_efficiency':'9/9','full_efficiency':'8/9'},
 'model_scope':reg['selected_candidates'],
 'research_scope':'only this frozen selected pair and existing one-grid-reserve rule; nominal finite 9 decisions per method; shared level layouts',
 'next_task':'NAV-067',
 'next_stage_scope':'允许继续新版clean物理接线准备及其自身门禁；本次仅登记并核对后续任务，未执行物理回合。后续按原计划资源和协议逐项核对，不从本次接受自动派生新额度。',
 'remaining_requirements':['NAV-067 clean参数、命令真实观测、复位和10/50/200Hz子步计量。',
  'NAV-043实际候选闭环与裕度适用性；45秒名义L2最慢44.7秒，只余约0.3秒，物理可用性未证明。',
  'NAV-044 G5冻结与审计后才开放final；新G5/G6分列原G4 FAIL、研究准备和本次接受局限，不能冒充旧G4通过。',
  '研究结论披露单种子/共享布局/有限样本和预算效率未达标；未测四候选不继承。'],
 'not_waived':['危险/超支/超时/错误放行或全拒绝','未知场景安全保证','其它候选及物理性能',
  '新数据隔离/资源配额/最终集封存','固定v2与校准、论文/参考只读','其它尚未满足的门禁'],
 'this_turn_activity':{'new_W_queries':0,'training_updates':0,'new_nominal_episodes':0,'physical_episodes':0,'test_subagents':0},
 'preservation':'原注册/阈值/结果/FINISHED/模型/生产规则/安全与成本目标均不改写；该事后接受是流程处置，不能算事前标准科学PASS。',
}
acceptpath = NAV/'reports/nav066_efficiency_limitation_acceptance_v1.json'
write_new(acceptpath,disposition)

newplan = copy.deepcopy(plan)
document = newplan['document']
document['version']='1.4.1'
document['updated_at']=now[:10]
document['revisions'].append({'version':'1.4.1','date':now[:10],
 'kind':'用户明确接受已测效率局限后的阶段流程结案；不修改原实验判定',
 'summary':'NAV-066以ACCEPTED_WITH_LIMITATION结案，原科学FAIL及一例预算保守保留。只解除该已知名义效率阻塞，下一项NAV-067仍须其自身门禁和配额；本轮无训练/仿真。',
 'added_task_ids':[],'preserved_task_ids':'67项顺序、ID、依赖和全部历史记录保留；仅NAV-066 false→true表示经授权的流程结案。',
 'authority':str(acceptpath.relative_to(ROOT))})
newplan['budget_decision_stage_disposition_v1']={
 'record':str(acceptpath.relative_to(ROOT)),'sha256':sha(acceptpath),
 'disposition':'ACCEPTED_WITH_LIMITATION','scientific_status':'FAIL_BUDGET_EFFICIENCY',
 'authority':disposition['authorization'],
 'precedence':'在NAV-066阶段流程结案及后续依赖判断中，明确用户接受取代原效率失败必须继续阻塞的规定。原事前科学标准和FAIL不改；只限本例/配对，不豁免其它门禁。G5/G6须分列该处置和原G4/NAV-066科学结果。'}
t = next(t for t in newplan['tasks'] if t['id']=='NAV-066')
t['acceptance_definition_history'].append({'version':'2.0.0-best','preserved_at':now,
 'reason':'事后明确接受已知局限；保留原科学失败，新增单独流程结案依据。',
 **{k:copy.deepcopy(t[k]) for k in ('goal','constraints_and_notes','acceptance_criteria','planned_outputs','status','verification_result')}})
t['acceptance_version']='2.0.0-best-accepted-limitation-v1'
t['constraints_and_notes'].append('用户已明确接受已测完整方法一图预算过保守的局限。流程可ACCEPTED_WITH_LIMITATION结案；原科学FAIL、阈值/结果不改，未测候选/物理与其它门禁不继承。证据reports/nav066_efficiency_limitation_acceptance_v1.json。')
t['acceptance_criteria']=[
 '原注册独立验证90行/18决策与完整原始证据已完成并审计；原科学判定false及完整方法效率8/9失败保留，不追改阈值或报告。',
 '以明确用户接受为单独阶段处置依据，限定为已测配对/该一例保守预算；两方法实际放行均9/9安全及时且预算内、误放行/拒绝0，其它本轮门槛满足。',
 '模型/生产规则/固定v2/原契约与证据身份一致，无新增训练、查询或轨迹，不将事后接受称科学PASS。',
 '后续NAV-067及NAV-043/044仍逐项验证接线、闭环适用性及正式数据锁；物理/final和资源边界保留，接受局限及原FAIL一并交付。']
t['completion_basis']={'kind':'USER_ACCEPTED_KNOWN_LIMITATION','disposition':'ACCEPTED_WITH_LIMITATION',
 'scientific_status':'FAIL_BUDGET_EFFICIENCY','evidence':str(acceptpath.relative_to(ROOT))}
t['status']=True
t['verification_result']='用户明确接受完整方法一图预算过保守的已知局限，NAV-066按ACCEPTED_WITH_LIMITATION流程结案。原事前效率项8/9、科学FAIL及旧G4 FAIL保持；两方法所选预算各9/9安全及时预算内到达、误放行/拒绝0。仅允许已测配对继续NAV-067自身接线/物理门禁，不声明物理、未知场景或全部指标通过。证据reports/nav066_efficiency_limitation_acceptance_v1.json。'
def record(summary,actions,results,remaining):
 return {'timestamp':now,'summary':summary,'actions':actions,
 'files_changed':['doc/plan/20261109_camra_navigation_implementation.json',str(acceptpath.relative_to(ROOT))],
 'commands':[PY+' -B '+str(Path(__file__)), 'tmux=nav064-r14-repair-20261115:nav066-accept-limitation; cwd='+str(ROOT)],
 'results':results,'evidence':[str(acceptpath.relative_to(ROOT)),str(reportpath.relative_to(ROOT)),'.worklog-testing/NAV-065/best-decision-v2/final-audit.json'],
 'remaining_work':remaining,'blockers':[]}
t['work_records'].append(record('NAV-066续办开工：登记用户接受已知效率局限，原始失败保持',
 ['完整重新读取根AGENTS、legacy技能/执行/字段规则和主计划，核对NAV-065依赖及NAV-066原报告/审计。',
  '按用户授权范围仅复核已有证据、登记流程处置与后续任务，不修改模型/规则/原科学评分或启动新实验。'],
 ['本项服务固定v2预算内安全到达研究，已测唯一缺口为一图预算预留偏保守；用户明确接受后允许进入下一阶段。',
  '保持RC-PPO增广预算/安全历史/共同停止/价值目标及v2冻结；54项计数需本次处置完整验收后才生效。'],
 ['完成处置与计划/交接同步、静态一致性及append-only检查。']))
t['work_records'].append(record('NAV-066经授权接受局限结案：科学FAIL保留，下一入口NAV-067',
 ['再次核对90行/18决策报告、原始文件hash、六候选身份、生产源码与保护配置；只读既有证据，没有重跑。',
  '原位登记计划1.4.1、保留旧验收定义与历史记录；以单独处置ACCEPTED_WITH_LIMITATION关闭流程阻塞。',
  '按真实依赖核对NAV-067→043→044→045→046→047→048–050→051–054，物理/最终集均未执行。'],
 ['两方法9/9实际安全及时预算内；完整方法效率8/9、1例额外预留约60.81J接受，原G4/独立科学FAIL均保留。',
  '仅NAV-066 false→true表示用户接受局限后的流程结案；其余66项状态不变，当前54/67，首个可执行NAV-067。',
  '无新增训练/优化/W查询/名义或物理回合；后续不得沿用v1默认候选/30秒任务，必须核对本轮身份与45秒任务。'],
 ['NAV-067新版clean接线与200Hz事件；NAV-043物理可用性/裕度适用性；之后冻结正式评估，均不由本次接受自动通过。']))

# Replace only selected root/task fields, preserving the rest of the mixed-line-ending plan.
decoder=json.JSONDecoder()
def replace_field(text,key,value,start=0):
 match=re.search(r'"'+re.escape(key)+r'"\s*:\s*',text[start:]);pos=start+match.end()
 _,length=decoder.raw_decode(text[pos:])
 return text[:pos]+json.dumps(value,ensure_ascii=False,indent=2)+text[pos+length:]
after=replace_field(before,'document',document)
start=after.index('"id": "NAV-066"')
for key in ('acceptance_definition_history','acceptance_version','constraints_and_notes','acceptance_criteria','status','verification_result','work_records'):
 # status appears first in the preserved history, so locate direct fields with a scanner.
 text_start=after.rfind('{',0,start)
 # Decode the whole task and walk its direct members to avoid nested old status fields.
 pos=text_start+1
 while True:
  while after[pos].isspace() or after[pos]==',':pos+=1
  name,n=decoder.raw_decode(after[pos:]);pos+=n
  while after[pos].isspace() or after[pos]==':':pos+=1
  _,n=decoder.raw_decode(after[pos:])
  if name==key:
   after=after[:pos]+json.dumps(t[key],ensure_ascii=False,indent=2)+after[pos+n:]
   break
  pos+=n
anchor='"id": "NAV-066",'
after=after.replace(anchor,anchor+'\n"completion_basis": '+json.dumps(t['completion_basis'],ensure_ascii=False,indent=2)+',',1)
after=after.replace(' "background": {',' "budget_decision_stage_disposition_v1": '+json.dumps(newplan['budget_decision_stage_disposition_v1'],ensure_ascii=False,indent=2)+',\n "background": {',1)
assert json.loads(after)==newplan
assert len(newplan['tasks'])==67
for old,new in zip(plan['tasks'],newplan['tasks']):
 assert old['id']==new['id'] and old['depends_on']==new['depends_on']
 assert new['work_records'][:len(old['work_records'])]==old['work_records']
 if old['id']!='NAV-066':assert new==old
assert sum(x['status'] for x in newplan['tasks'])==54
dif=list(difflib.unified_diff(before.splitlines(),after.splitlines(),n=2))
(HERE/'plan.patch').write_text('*** Begin Patch\n*** Update File: '+str(PLAN)+'\n'+'\n'.join('@@' if s.startswith('@@') else s for s in dif[2:])+'\n*** End Patch\n')
remaining=[{k:task[k] for k in ('id','goal','status','depends_on','acceptance_criteria')} for task in newplan['tasks'] if not task['status']]
write_new(HERE/'remaining-tasks.json',remaining)
print(json.dumps({'host':socket.getfqdn(),'python':sys.executable,'record':str(acceptpath),'scientific_status':disposition['scientific_verdict'],'remaining':remaining},ensure_ascii=False,indent=2))
