"""Build an apply_patch payload only; never fit or create an approval."""
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
assert ROOT == Path.cwd() and socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == '77cd7376c200bf8a18bf4672e75d040f01b7e7a4'
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_report = NAV / 'reports/nav043_saved_applicability_audit_v1.json'
new_report = NAV / 'reports/nav043_exploration_applicability_v1.json'
entries = []
for report, exploration in ((old_report, False), (new_report, True)):
    for row in load(report)['rows']:
        trial = Path(row['trial'])
        hashes = row['source_hashes']
        assert len(hashes) == 5
        result = load(trial / 'result.json')
        entries.append({'trial': str(trial), 'level': row['level'], 'method': row['method'],
                        'map_id': result['map_id'],
                        'kind': 'W_selected' if exploration else row['trial_kind'],
                        'audited_report': str(report), 'hashes': hashes})
entries.sort(key=lambda e: (e['level'], e['kind'], e['method']))
assert len(entries) == 18 and len({e['trial'] for e in entries}) == 18
registry = NAV / 'config/nav066_decision_prereg_v2.json'
reports = [old_report, new_report, NAV / 'reports/nav043_exploration_review_v1.json',
           NAV / 'reports/nav067_engineering_acceptance_v1.json']
sources = [NAV / 'scripts/nav043_motion_response.py',
           NAV / 'tests/test_nav043_motion_response.py']
protocol = {
 'schema': 'nav043-motion-response-v1', 'status': 'PROPOSED', 'task': 'NAV-043',
 'prepared_at': datetime.now().astimezone().isoformat(),
 'source_commit': '77cd7376c200bf8a18bf4672e75d040f01b7e7a4',
 'output_root': '/pub/data/caohy/work/camra_navigation/nav043_motion_response_probe_v1',
 'data_use': 'Eighteen already-exposed integration traces; grouped feasibility diagnosis, not independent unseen validation.',
 'limits': {'models':3,'axis_solves':9,'cpu_wall_s':3600,'new_physical_episodes':0,
            'new_nominal_episodes':0,'W_queries':0,'actor_critic_updates':0,'calibration_fits':0},
 'model': {'axes':['vx','vy','Euler_yaw_rate'], 'parameters':6,
           'recurrence':'q_n = a*q_(n-1) + b*u_n per axis; zero intercept, no cross-axis terms',
           'response_definition':'SE(2) logarithm of each complete actual 0.1 s planar pose increment divided by 0.1 s; not body wz',
           'physics_dt_s':.005, 'navigation_dt_s':.1,
           'fit_pairs':'n>=1 in each trace, complete consecutive intervals only; first measured q_0 only supplies lagged training regressor; partial last interval excluded',
           'fit_weight':'one equal row per complete adjacent interval, no weighting/tuning by outcome',
           'solver':'one numpy.linalg.lstsq per axis, float64, rcond=1e-12',
           'numerical_guards':{'rank':2,'condition_max':1e8,'a_min_inclusive':0.,'a_max_exclusive':1.,'b_strictly_positive':True},
           'invalid_axis':'retain result and stop whole allocation; no clipping, regularization, model switch or retry',
           'free_replay':'align only actual warmup planar pose at t=0; q_-1=0 on every heldout trace; saved commands only, no future state or observed q_0 initialization',
           'partial_last_interval':'one ordinary navigation response update at interval start; integrate saved duration only, no fitting/padding/extrapolation',
           'replay_grid_hz':200, 'prefixes_s':[5,10,20,30,45]},
 'folds': [{'heldout_level':level,'fit_traces':12,'heldout_traces':6}
           for level in ('L1_empty','L2_single_detour','L3_multi_path')],
 'diagnostic_gate':{'per_map':'time-weighted full-window heldout planar position RMSE <= ideal baseline + 1e-10 m for each of three maps',
                    'aggregate':'time-weighted all-eighteen heldout planar position RMSE < ideal baseline - 1e-10 m',
                    'report_all':'each trace terminal/max/RMSE and complete prefixes, heading and complete-interval response error; do not drop a failure',
                    'interpretation':'diagnostic only, does not validate old delta, physical success, independent generalization, NAV-043 or G5'},
 'approval_required':str(NAV/'config/nav043_motion_response_approval_v1.json'),
 'protected_registry':{'path':str(registry),'sha256':sha(registry),'identity_count':295},
 'report_hashes':{str(p):sha(p) for p in reports},
 'implementation_hashes':{str(p):sha(p) for p in sources},
 'traces':entries}
path = NAV / 'config/nav043_motion_response_proposal_v1.json'
pin = Path(str(path) + '.sha256')
assert not path.exists() and not pin.exists()
encoded = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
payload = ['*** Begin Patch', '*** Add File: ' + str(path)]
payload.extend('+' + line for line in encoded.splitlines())
payload.extend(['*** Add File: ' + str(pin), '+' + hashlib.sha256(encoded.encode()).hexdigest()])
plan_path = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
before = plan_path.read_text()
plan = json.loads(before)
assert before == json.dumps(plan, ensure_ascii=False, indent=2) + '\n'
task = next(t for t in plan['tasks'] if t['id']=='NAV-043')
assert len(task['work_records']) == 25 and task['status'] is False
task['work_records'].append({
 'timestamp':datetime.now().astimezone().isoformat(),
 'summary':'NAV-043下一范围的可审阅实现准备：保存数据响应模型入口，正式拟合仍待批准',
 'actions':['恢复后重新读取AGENTS、legacy技能执行规则、当前1.4.1计划/依赖、最终审阅及具体提案；前一目标轮提交77cd737并验证工作区干净，完成了真实进展。',
            '当前仍NAV-043/55项true，服务于固定v2的预算内安全及时到达。消费18条已经暴露的保存轨迹和原报告，不重新执行应用/策略或改RC-PPO目标、旧冻结系统、45秒、评分及数据用途。',
            '只准备可审阅的六参数SE(2)区间响应模型、三折/九求解资源提案及隔离入口/基本合成反例；未进行真实响应模型拟合。运行入口在读取拟合数据前拒绝缺少独立批准登记，提案不充当批准。'],
 'files_changed':['CaMRA/navigation/scripts/nav043_motion_response.py','CaMRA/navigation/tests/test_nav043_motion_response.py',
                  'CaMRA/navigation/config/nav043_motion_response_proposal_v1.json','CaMRA/navigation/config/nav043_motion_response_proposal_v1.json.sha256',
                  '.worklog-testing/NAV-043/motion-response-v1/prepare_proposal.py','doc/plan/20261109_camra_navigation_implementation.json'],
 'commands':['tiger11/tmux准备提案patch，仅根据原报告生成精确来源/模型/资源及追加记录；JSON使用apply_patch落盘。随后仅运行主agent合成基本检查、18条只读预检和legacy update，不执行--run。'],
 'results':['18条来源和整图分组已确定；真实响应模型求解/新应用/物理或名义回合/W/actor或critic更新/校准拟合均0。',
            '本条只记准备开工；基本检查及预检的实际终态后续追加。NAV-043/G5仍未通过。'],
 'evidence':['CaMRA/navigation/docs/nav043_exploration_final_review_20261119.md','CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md'],
 'remaining_work':['完成主agent基本检查、只读预检及具体协议说明/封存/本地提交；正式拟合须等待新范围确认。'],
 'blockers':['新响应模型真实求解范围及独立额度尚未批准，不转用已经结束的物理试验余量。']})
after = json.dumps(plan, ensure_ascii=False, indent=2) + '\n'
diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), n=3))
payload.append('*** Update File: ' + str(plan_path))
payload.extend('@@' if line.startswith('@@') else line for line in diff[2:])
payload.append('*** End Patch')
with (HERE/'proposal.patch').open('x') as stream:stream.write('\n'.join(payload)+'\n')
print(json.dumps({'status':'PREPARATION_PATCH_ONLY','trace_count':18,'actual_model_solves':0,
                  'proposal_sha256':hashlib.sha256(encoded.encode()).hexdigest()}),flush=True)
