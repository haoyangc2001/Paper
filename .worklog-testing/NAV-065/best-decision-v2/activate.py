from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PLAN = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p:json.loads(Path(p).read_bytes())
now = datetime.now().astimezone().isoformat()
baseline = load(HERE/'baseline.json')
selection = load(HERE/'selection.json')
plan = load(PLAN)
tasks = {t['id']:t for t in plan['tasks']}
assert not tasks['NAV-065']['status'] and not tasks['NAV-066']['status']
(HERE/'plan-full-read.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
print(json.dumps({k:plan[k] for k in ('execution_alignment_policy','uncertainty_policy')},ensure_ascii=False,indent=2))
print(json.dumps({i:{k:v for k,v in tasks[i].items() if k not in ('work_records','test_reports','acceptance_checks')} for i in ('NAV-065','NAV-066')},ensure_ascii=False,indent=2))

def write_new(p,obj):
    with p.open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

implementation = {str(p.relative_to(NAV)):sha(p) for folder in ('src','scripts') for p in (NAV/folder).rglob('*.py')}
implementation['tests/test_guarded_budget.py'] = sha(NAV/'tests/test_guarded_budget.py')
reg = {
 'schema':'camra-budget-decision-approved-prereg-v2','status':'APPROVED','version':'2.0.0-best',
 'authorization_reference':'用户确认新版预算决策协议，最新指令：确认，用目前训练出来的最好的模型去检验。既有同机tiger11/tmux例外和本地提交授权持续有效；不派专用测试子agent。',
 'draft_sha256':sha(NAV/'config/nav065_budget_decision_protocol_v2.json'),
 'draft_preserved':True,'limits':{'episodes':90,'optimizer_updates':0,'physical_episodes':0,'queries':90},
 'horizon_s':45.0,'max_steps_per_episode':450,'max_nominal_steps':40500,
 'horizon_correction':{'draft_typo_seconds':30,'draft_typo_max_steps':27000,'actual_frozen_seconds':45,'actual_max_steps':40500,
  'authority':'config/training_loop_prereg_terminal_tail_v1.json environment.t_max_s=45; original frozen source wins. Prior draft misreported this as unchanged=30. This is recorded before any new map/model query/trajectory. Episodes remain 90, no task/model change.'},
 'six_candidates':[r['identity'] for r in selection['six_best_by_run']],
 'selected_candidates':[r['identity'] for r in selection['selected']],
 'selection_rule':selection['rule'],'selection_override':selection['user_override'],
 'selection_evidence_sha256':sha(HERE/'selection.json'),
 'historical_fit_reports':selection['evidence'],
 'implementation_hashes':implementation,
 'protected_hashes':{**baseline['protected'],**baseline['v2']},
 'map_seeds':[917001,917002,917003],
 'levels':['L1_empty','L2_single_detour','L3_multi_path'],
 'output_directory':'/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best',
 'decision_rule':{'module':'camra_navigation.budget.guarded.select_guarded_budget','grid':[.70,.85,1.,1.15,1.30],
  'W':'max(V_h,V_c-z0_J/550)','beta':0,'reserve_steps':1,'reserve_k':.15,
  'reject':['nonfinite','exact W=0','feasible then infeasible','no feasible budget','no room for one reserve step'],
  'seal':'all 18 decisions and 90 initial W queries before any rollout; labels never enter decision'},
 'acceptance':{'per_method':True,'accepted_failures_max':0,'overspend_allowance_J':0,
  'feasible_map_recall_min':.8,'accepted_maps_min':6,'accepted_per_level_min':2,
  'extra_grid_steps_above_actual_min_max':1,
  'success':'45s task, actual irreversible safety history=0, goal_reached, no task failure, cost_model_J <= z0_J; original labels and floating comparisons'},
 'data_isolation':{'new_derivative_split':'budget_selection_decision_v2','old_selection_exposed':True,
  'final_results_opened':False,'policy_selection_after_access':False,'feedback_to_training':False,
  'layouts_shared_with_prior_levels':True,'claim':'unseen task instances only; not unseen layouts or IID guarantee'},
 'failure_policy':'No automatic retries, seed replacement, extra queries/rollouts, threshold changes or further training. Business failures preserved. Infrastructure fault holds the run with attempt journal.',
 'plan_revision':'1.4.0; NAV-065 readiness only, NAV-066 measured finite decision gate. Old G4 FAIL and all historical criteria/results preserved; only selected pair gets measured scope.',
 'limitations':['No probability guarantee for unknown inputs; at most 9 decisions per method, not 45/90 independent decisions.',
  'Nominal internal cost is not physical joint work; fixed v2 calibration applicability at 45s still requires NAV-043 before physical scientific claims.',
  'Best selection uses reused/exposed development; independent validation cannot support unrestricted multi-seed method performance.'],
}
regpath = NAV/'config/nav066_decision_prereg_v2.json'
write_new(regpath,reg)
regpath.with_suffix('.json.sha256').write_text(sha(regpath)+'\n')
write_new(NAV/'reports/nav065_best_existing_model_binding_v2.json',selection)

# Preserve every task and historical record; edit only the approved definitions.
before = PLAN.read_text()
newplan = json.loads(before)
document = newplan['document']
document['version']='1.4.0'
document['updated_at']='2026-11-18'
document['revisions'].append({'version':'1.4.0','date':'2026-11-18',
 'kind':'用户确认的验收复审与最优现有模型独立检验，不训练',
 'summary':'分离拟合诊断、NAV-065准备与NAV-066实际预算决策验收；保留旧G4 FAIL及历史标准。按已暴露development选最优现有配对、45秒冻结任务和90名义回合上限。',
 'added_task_ids':[],'preserved_task_ids':'67项顺序、ID、依赖及全部历史记录保留；状态仅在逐项实际验收后更新。'})
amendment = {'registration':'CaMRA/navigation/config/nav066_decision_prereg_v2.json','registration_pin':sha(regpath),
 'authority':reg['authorization_reference'],
 'precedence':'本次已批准的有限研究验收仅覆盖NAV-065/NAV-066。旧NAV-061、G4及NAV-065的全train零误放行/35%/0.70等作为历史拟合诊断完整保留，不再作为本轮准备或实际预算决策的前置硬门槛；不据此宣称价值预测无误。其它全局约束及下游独立门禁继续适用。',
 'horizon_correction':reg['horizon_correction'],'old_G4':'FAIL','downstream_scope':'only selected uncalibrated seed2 guarded-v4 + full seed0 original; no physical or other-seed inheritance'}
newplan['budget_decision_acceptance_revision_v2']=amendment
ntasks={t['id']:t for t in newplan['tasks']}
for i in ('NAV-065','NAV-066'):
 t=ntasks[i]
 t['acceptance_definition_history']=[{'version':'1.3.1','preserved_at':now,**{k:t[k] for k in ('goal','constraints_and_notes','acceptance_criteria','planned_outputs')}}]
 t['acceptance_version']='2.0.0-best'
t=ntasks['NAV-065']
t['goal']='冻结最优现有候选及预算规则，披露拟合局限，完成独立决策验证的研究准备'
t['constraints_and_notes']=[
 '按用户最新指令只用已有development择优；六身份/原始报告绑定，不训练、不新增开发评分，模型和v2冻结。',
 '旧G4 FAIL、原35%/0.70/零误放行标准及全部失败保留为历史诊断，不复用为本轮科学通过。',
 '生产路径用双critic W=max(V_h,V_c-z/550)、beta=0及一档保护；不改预算增广、安全历史、共同停止或actor。',
 '保持原模型45秒task；旧草案30秒系笔误，90回合额度不增加。最多40500名义步；没有物理回合。',
 '准备通过只为PROVISIONAL_RESEARCH_READY，实际可用性属于NAV-066；未测候选及物理下游不能继承。']
t['acceptance_criteria']=[
 '绑定六组已有候选、其来源与历史拟合/开发报告；严格CPU重载六组actor/双critic并验证张量hash；明确所测最优配对及未测范围。',
 '原始拟合错误不隐去：保留历史训练/开发门禁和失败、披露目标尺度/误差/符号诊断；不得声称全训练点无误。',
 '实际独立验证入口调用已冻结生产保护规则和同一双critic，基本测试覆盖留量/异常/非单调/全部拒绝/预算效率及缺失数据，J与z/550和版本一致。',
 '批准登记、实现hash、模型hash、数据/资源边界和新9图派生清单已冻结；几何/45秒准入与历史种子/任务实例互斥审计完成，未访问新结果。',
 'v2/校准及原来源锁未变；模型无更新，尚无新轨迹；只产生研究准备证据，不覆盖旧G4。']
t['planned_outputs']=['config/nav066_decision_prereg_v2.json及.sha256','reports/nav065_best_existing_model_binding_v2.json','src/camra_navigation/budget/guarded.py','scripts/run_nav066_decision_validation.py','tests/test_guarded_budget.py','NAV-065 readiness.json']
t=ntasks['NAV-066']
t['goal']='以冻结最优配对和一档保护规则独立验证实际预算决策'
t['constraints_and_notes']=[
 '依赖新版NAV-065研究准备；同一actor/双critic与已批准登记，先封存18个决策/90个W查询，再执行至多90条名义轨迹。',
 '5档乘以按方法一致的政策无关E_ref；初始J预算和z/550分字段；拒绝无fallback、不延长区间、不用标签择优。',
 '每方法9图45网格行；选中行先执行且复用网格行。未选/拒绝图的诊断轨迹不影响封存决策。',
 '只检验未见任务实例，level布局共享；final_*封存。结果不反馈训练、选择模型或调整规则；失败保留，不自动重跑。',
 '旧G4不改写；按方法分别判定，不混合掩盖失败；只限已测单种子名义配对，不进入最终性能分母。']
t['acceptance_criteria']=[
 '生产调用链、模型/规则/数据身份、18决策及90查询封存可复算；90网格行均有原始轨迹、安全历史、成本J和独立oracle核验。',
 '每方法全部放行均45秒内安全且预算内到达，放行失败0、允许超支0J；按原标签判定，不要求全部原始W预测正确。',
 '每方法至少6/9图放行且每level至少2/3；对有任一网格真实可行图的正确放行召回至少80%，无可行图不通过。',
 '每放行图所选索引不高于该冻结actor真实网格最低可行索引+1；完整报告拒绝率、误拒绝、原W混淆、超支及预算利用率。',
 '无训练、无反馈调参、无物理回合、无final访问；90条及40500步上限内，业务失败完整保留；任何一方法未达本轮门槛则false。']
t['planned_outputs']=['config/nav066_decision_prereg_v2.json','reports/nav066_budget_decision_validation_v2.json','docs/nav066_budget_decision_review_v2_20261118.md','外部独立目录manifest/decisions/episodes/decision_validation']

# Locate JSON field boundaries rather than reformatting the 13k-line history.
decoder=json.JSONDecoder()
def replace_field(text,key,value,start=0):
 match=re.search(r'"'+re.escape(key)+r'"\s*:\s*',text[start:])
 pos=start+match.end()
 _,length=decoder.raw_decode(text[pos:])
 return text[:pos]+json.dumps(value,ensure_ascii=False,indent=2)+text[pos+length:]
after=replace_field(before,'document',newplan['document'])
for i in ('NAV-065','NAV-066'):
 start=after.index('"id": "'+i+'"')
 for key in ('goal','constraints_and_notes','acceptance_criteria','planned_outputs'):
  after=replace_field(after,key,ntasks[i][key],start)
 # Add revision metadata at this task's id, preserving old unknown fields.
 anchor='"id": "'+i+'",'
 extra=',\n'.join('"'+k+'": '+json.dumps(ntasks[i][k],ensure_ascii=False,indent=2) for k in ('acceptance_definition_history','acceptance_version'))
 after=after.replace(anchor,anchor+'\n'+extra+',',1)
after=after.replace(' "background": {',' "budget_decision_acceptance_revision_v2": '+json.dumps(amendment,ensure_ascii=False,indent=2)+',\n "background": {',1)
assert json.loads(after)==newplan
diff=list(difflib.unified_diff(before.splitlines(),after.splitlines(),n=2))
patch='*** Begin Patch\n*** Update File: '+str(PLAN)+'\n'+'\n'.join('@@' if s.startswith('@@') else s for s in diff[2:])+'\n*** End Patch\n'
(HERE/'activation.patch').write_text(patch)
print('registration_pin',sha(regpath),'six',len(reg['six_candidates']),'implementation_files',len(implementation),flush=True)
