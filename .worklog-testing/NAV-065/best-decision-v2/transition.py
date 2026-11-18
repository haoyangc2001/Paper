from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
HERE=Path(__file__).resolve().parent
PLAN=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
OUT=Path('/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best')
PY='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
load=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
decoder=json.JSONDecoder()
phase=sys.argv[1]
before=PLAN.read_text()
plan=json.loads(before)
tasks={t['id']:t for t in plan['tasks']}
reg=load(NAV/'config/nav066_decision_prereg_v2.json')
pin=sha(NAV/'config/nav066_decision_prereg_v2.json')
assert all(sha(p)==h for p,h in reg['protected_hashes'].items())
assert all(sha(NAV/p)==h for p,h in reg['implementation_hashes'].items())
assert len(plan['tasks'])==67
assert [t['id'] for t in plan['tasks']]==[t['id'] for t in load(HERE/'baseline.json')['plan']['tasks']]
def record(summary,actions,results,evidence,remaining=[],blockers=[]):
 return {'timestamp':datetime.now().astimezone().isoformat(),'summary':summary,'actions':actions,
 'files_changed':['doc/plan/20261109_camra_navigation_implementation.json'],
 'commands':[PY+' -B '+str(Path(__file__))+' '+phase,'tmux=nav064-r14-repair-20261115; cwd='+str(ROOT)],
 'results':results,'evidence':evidence,'remaining_work':remaining,'blockers':blockers}

changes={}
records={}
if phase=='ready':
 ready=load(OUT/'readiness.json')
 manifest=load(OUT/'manifest.json')
 assert ready['registration_pin']==pin and ready['manifest_sha256']==sha(OUT/'manifest.json')
 assert ready['new_rollouts']==0 and ready['actor_or_critic_updates']==0
 assert len(manifest['maps'])==9 and all(m['admission']['admitted'] for m in manifest['maps'])
 assert not (OUT/'decisions.json').exists() and not (OUT/'episodes.jsonl').exists()
 assert 'Ran 13 tests' in (HERE/'basic-tests.log').read_text() and (HERE/'basic-tests.log').read_text().strip().endswith('OK')
 assert not tasks['NAV-065']['status'] and tasks['NAV-064']['status']
 oldbase=load(HERE/'baseline.json')['plan']
 assert all(tasks[t['id']]['work_records'][:len(t['work_records'])]==t['work_records'] for t in oldbase['tasks'])
 review={'status':'PROVISIONAL_RESEARCH_READY','scientific_validation':False,'old_G4':'FAIL',
  'registration_pin':pin,'source_binding':'reports/nav065_best_existing_model_binding_v2.json',
  'historical_fit_disclosure':'原original/refit-v1/v4全量训练/开发指标和失败继续由原始报告记录；此次按已暴露development择优不是拟合已清零。后续v5-v11仅离线/单候选诊断，不以无匹配开发证据替换所测配对。',
  'historical_reports':reg['historical_fit_reports'],'test_count':13,'test_log_sha256':sha(HERE/'basic-tests.log'),
  'implementation_files':len(reg['implementation_hashes']),'protected_files':len(reg['protected_hashes']),
  'new_maps':9,'initial_W_queries':0,'new_episodes':0,'updates':0,
  'cpu_reload':'prepare phase loaded every six_candidates checkpoint and verified policy/critic_cost/critic_reach tensor digests; selected pair alone is later used',
  'production_chain':'run_nav066_decision_validation.decide -> select_guarded_budget(query) -> registered CPU dual critics; evaluate -> same frozen actor -> rollout -> reconstruct_truth -> summarize_decisions',
  'manifest_sha256':sha(OUT/'manifest.json'),'readiness_sha256':sha(OUT/'readiness.json'),
  'execution':'tiger11; tmux nav064-r14-repair-20261115:nav065-ready-v2; absolute Python3.11.15',
  'limits':reg['limits'],'horizon_s':45,'max_nominal_steps':40500,
  'limitations':reg['limitations']}
 (NAV/'reports/nav065_budget_decision_readiness_v2.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 records['NAV-065']=record('NAV-065新版准备验收完成：保留旧G4 FAIL，仅研究准备通过，不宣称价值可靠',
 ['依据用户确认原位激活1.4.0与新版定义，旧标准另存acceptance_definition_history，67任务/依赖/历史保留。',
  '按已有18份开发候选结果固定六身份和最优配对；CPU重载六组并核对张量，33项v2及原契约/源码锁匹配。',
  '保护规则13项基础测试通过；9图几何/45秒准入、历史种子/实例及元数据暴露审计通过，派生参考成本已冻结。',
  '纠正草案30秒笔误为模型原冻结45秒，90回合上限不变，导出40500步上限；不改模型或目标。'],
 ['NAV-065=true只表示PROVISIONAL_RESEARCH_READY；旧G4 FAIL与拟合错误保留。','截至收尾新W查询0、轨迹0、更新0；NAV-066独立验证未开始。'],
 ['CaMRA/navigation/reports/nav065_budget_decision_readiness_v2.json',str(OUT/'readiness.json'),str(OUT/'manifest.json'),str(HERE/'basic-tests.log')],
 ['完成所测配对的NAV-066实际独立决策验收；其它四候选和物理性能不继承。'])
 changes['NAV-065']={'status':True,'verification_result':'v2.0.0-best准备验收通过；六身份与CPU张量核验、13基础测试、9独立任务实例准入和协议冻结完成。PROVISIONAL_RESEARCH_READY；旧G4 FAIL，未声明价值无误或科学通过。证据reports/nav065_budget_decision_readiness_v2.json。'}
elif phase=='open':
 assert tasks['NAV-065']['status'] and not tasks['NAV-066']['status']
 assert next(t['id'] for t in plan['tasks'] if not t['status'] and all(tasks[d]['status'] for d in t['depends_on']))=='NAV-066'
 assert not (OUT/'decisions.json').exists() and not (OUT/'episodes.jsonl').exists()
 records['NAV-066']=record('NAV-066开工：冻结最优现有配对的五档价值决策，之后独立执行90名义回合',
 ['切换任务前重新完整读取根AGENTS和计划全文；核对NAV-065准备及全部上游来源有效。',
  '绑定已批准登记pin、9图清单、两方法各一个最优现有模型；沿用W/beta/原目标和45秒任务。'],
 ['本项作用是检验实际预算决策能否安全及时预算内到达，同时限制拒绝和预算浪费。',
  '只允许90查询/90名义回合，更新0/物理0/final封存；先封存全部18决策，再执行轨迹，选中行复用。',
  '允许输出新验证日志和审阅；不改模型、规则、阈值、地图或v2；失败保持false并原样记录。'],
 ['CaMRA/navigation/config/nav066_decision_prereg_v2.json','CaMRA/navigation/reports/nav065_budget_decision_readiness_v2.json',str(OUT/'manifest.json')],
 ['执行封存决策和90行独立验证，按每方法安全/召回/拒绝/效率门槛验收。'])
elif phase=='close':
 report=load(OUT/'decision_validation.json')
 audit=load(HERE/'final-audit.json')
 assert audit['integrity_pass'] and report['registration_pin']==pin
 passed=report['pass']
 result={m:{k:v for k,v in s.items() if k!='details'} for m,s in report['methods'].items()}
 records['NAV-066']=record('NAV-066独立验证结束：'+('有限名义决策门禁PASS' if passed else '实际决策门禁FAIL，保留结果并停止下游'),
 ['封存18决策及90W查询后执行90条独立名义轨迹；主agent复核原始轨迹/J账本/不可逆安全/时间/oracle及网格效率。',
  '审计模型/规则/v2散列及历史记录，未训练、未改阈值/模型、未派测试子agent、未访问final或执行物理回合。'],
 [json.dumps(result,ensure_ascii=False),'总步数='+str(report['steps'])+'，旧G4 FAIL保留；这只是所测单种子配对的有限名义验证。'],
 ['CaMRA/navigation/reports/nav066_budget_decision_validation_v2.json','CaMRA/navigation/docs/nav066_budget_decision_review_v2_20261118.md',str(OUT/'episodes.jsonl'),str(HERE/'final-audit.json')],
 ['下游NAV-067/043及物理验收仍须单独完成。'] if passed else ['先评审本次固定规则的具体拒绝/错误/效率原因，任何实质新规则需新版本和独立数据；不默认重训或重跑。'],
 [] if passed else ['独立实际决策未同时满足各方法全部预登记门槛，NAV-066保持false，下游不执行。'])
 changes['NAV-066']={'status':bool(passed),'verification_result':('v2.0.0-best有限名义实际决策门禁通过；详见reports/nav066_budget_decision_validation_v2.json。只覆盖所测配对。' if passed else '')}
else:raise ValueError(phase)

after=before
for tid,record_ in records.items():
 start=after.index('"id": "'+tid+'"')
 for key,val in changes.get(tid,{}).items():
  match=re.search('"'+key+'"\\s*:\\s*',after[start:]);pos=start+match.end()
  _,length=decoder.raw_decode(after[pos:])
  after=after[:pos]+json.dumps(val,ensure_ascii=False)+after[pos+length:]
 match=re.search('"work_records"\\s*:\\s*',after[start:]);pos=start+match.end()
 arr,length=decoder.raw_decode(after[pos:]);end=pos+length-1
 after=after[:end]+(',' if arr else '')+'\n'+json.dumps(record_,ensure_ascii=False,indent=2)+'\n'+after[end:]
parsed=json.loads(after)
for old,new in zip(plan['tasks'],parsed['tasks']):
 assert old['work_records']==new['work_records'][:len(old['work_records'])]
diff=list(difflib.unified_diff(before.splitlines(),after.splitlines(),n=2))
(HERE/(phase+'.patch')).write_text('*** Begin Patch\n*** Update File: '+str(PLAN)+'\n'+'\n'.join('@@' if s.startswith('@@') else s for s in diff[2:])+'\n*** End Patch\n')
print('transition prepared',phase,changes,flush=True)
