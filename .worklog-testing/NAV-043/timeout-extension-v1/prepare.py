"""Build an append-only approval patch from the explicit current request."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
EV = ROOT/'.worklog-testing/NAV-043/timeout-extension-v1'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
def load(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(v): return json.dumps(v, ensure_ascii=False, indent=2)+'\n'
now = datetime.now().astimezone().isoformat()
old = load(NAV/'config/nav043_exploration_v1.json')
sealed = dict(old['sealed_hashes'])
for name, digest in sealed.items(): assert sha(name) == digest, name
names = ['scripts/nav043_timeout.py','scripts/nav043_timeout_simulation.py',
    'scripts/run_nav043_timeout.py','scripts/audit_nav043_timeout.py']
for name in names: sealed[str(NAV/name)] = sha(NAV/name)
sealed[old['pilot_manifest']] = sha(old['pilot_manifest'])
oldroot = Path(old['output_root'])
for level in old['order']:
    for name in ('decision.json','result.json'):
        p = oldroot/level/'full'/name
        sealed[str(p)] = sha(p)
protocol = {'schema':'nav043-execution-timeout-diagnostic-v1',
    'task':'NAV-043', 'status':'APPROVED_CURRENT_USER_REQUEST', 'approved_at':now,
    'authorization':{'user_instruction':'先做一下延长超时时间 看一下效果吧 我理解上层训练后应该具有通用性 我们的核心目标是训练一套这样的到达避障系统',
        'interpretation':'Current explicit timeout-extension instruction authorizes one minimal diagnostic on the three existing integration cases; use already designed 90s execution limit. No new training or approval question.',
        'new_allocation':True, 'old_allocation_reused':False},
    'output_root':'/pub/data/caohy/work/camra_navigation/nav043_timeout_extension_v1',
    'pilot_manifest':old['pilot_manifest'], 'old_exploration_root':str(oldroot),
    'physical_gpu_uuid':old['physical_gpu_uuid'], 'physical_gpu_index':6,
    'isaaclab_origin':old['isaaclab_origin'],
    'limits':{'applications':3, 'physical_navigation_episodes':3, 'W_queries':15,
        'nominal_episodes':0, 'gpu_wall_s':10800, 'explicit_physics_calls':60000,
        'per_application_physics_calls':20000, 'automatic_retries':0,
        'training_updates':0, 'calibration_fits':0},
    'order':old['order'], 'methods':['full'], 'execution_horizon_s':90.,
    'actor_observation_horizon_s':45., 'warmup_s':1.,
    'actor_time_semantics':'Original obs-v7: max(0,45-elapsed)/45, zero after 45s; not obs-v8-t90 and no model or critic reinterpretation.',
    'budget_semantics':'Same original five W queries from actual warmup state, grid and reserve; no raised budget after observing failure. Selected budget and all predictions sealed before physical action.',
    'execution':'Fixed full policy/v2/clean Go2/200-50-10Hz, one attempt per exposed map; retain all failures. Physical budget deficit is recorded using unchanged physical event semantics, no budget top-up.',
    'stopping':'Business terminal retains result and proceeds to fixed next map; infrastructure/evidence failure stops allocation. No retries or automatic alternate GPU.',
    'reporting':'Describe reach, safety, actual J and internal budget separately, including 45s prefix. Not independent generalization, calibration or G5 evidence; original FAIL/HELD and frozen production remain.',
    'sealed_hashes':sealed, 'NAV_043_pass':False, 'G5_pass':False}
assert not Path(protocol['output_root']).exists() and not (EV/'launch.lock').exists()
planpath = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
before = planpath.read_text()
plan = json.loads(before)
states = {t['id']:t['status'] for t in plan['tasks']}
assert next(t['id'] for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])) == 'NAV-043'
assert plan['document']['version'] == '1.5.0' and sum(states.values()) == 55
task = next(t for t in plan['tasks'] if t['id'] == 'NAV-043')
record = {'timestamp':now,
    'summary':'用户明确要求先延长超时直接看效果；登记旧45秒actor的90秒执行诊断，不训练或继续拟合',
    'actions':['完整读取根AGENTS、legacy技能/规则和当前计划；核对NAV-043依赖与55/67状态及旧45秒物理结果。',
        '直接落实用户本轮延时测试授权，不重复询问；登记新独立3应用/3完整方法integration回合/最多15初态W、60000显式步、GPU6累计3小时的一次诊断。',
        '新增隔离90秒执行监视器，原obs-v7仍按45秒归一化并在45秒后保持0；保持原actor/双critic、底层v2、预算选择、安全/成本/停滞规则，不重训练。',
        '先基本检查、实际资源预检和本地提交，再一次物理启动，完成后主agent保存复核；无测试子agent。'],
    'files_changed':['CaMRA/navigation/'+n for n in names]+['CaMRA/navigation/config/nav043_timeout_extension_v1.json'],
    'commands':['tiger11/tmux: timeout-extension-v1/basic.sh; prepare.py生成待应用追加patch'],
    'results':['当前尚无新增应用/物理步/W查询/训练/校准；所有原始失败保持。',
        '本次只诊断现有到达避障策略能否靠增加执行时间完成；不宣称旧critic或alpha/delta覆盖90秒。'],
    'evidence':['CaMRA/navigation/config/nav043_timeout_extension_v1.json',
        '.worklog-testing/NAV-043/timeout-extension-v1/prepare.log'],
    'remaining_work':['基本检查/预检通过后提交并运行一次；独立保存能量/200Hz事件/观测和45秒前缀复核，按实际结果分析。'],
    'blockers':[]}
task['work_records'].append(record)
after = dump(plan)
assert dump(json.loads(before)) == before, 'require stable original JSON formatting'
# Small exact replacement hunk: preserve every old record and all task fields.
a = before.splitlines(); b = after.splitlines()
start = 0
while start < min(len(a),len(b)) and a[start] == b[start]: start += 1
end_a,end_b = len(a),len(b)
while end_a > start and end_b > start and a[end_a-1] == b[end_b-1]: end_a-=1;end_b-=1
lo = max(0,start-3); hi_a = min(len(a),end_a+3); hi_b = min(len(b),end_b+3)
patch = '*** Begin Patch\n'
for path, text in [(NAV/'config/nav043_timeout_extension_v1.json',dump(protocol)),
        (NAV/'config/nav043_timeout_extension_v1.json.sha256',hashlib.sha256(dump(protocol).encode()).hexdigest()+'\n')]:
    patch += '*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in text.splitlines())
patch += '*** Update File: '+str(planpath)+'\n@@\n'
patch += ''.join(' '+line+'\n' for line in a[lo:start])
patch += ''.join('-'+line+'\n' for line in a[start:end_a])
patch += ''.join('+'+line+'\n' for line in b[start:end_b])
patch += ''.join(' '+line+'\n' for line in a[end_a:hi_a])
patch += '*** End Patch\n'
(EV/'approval.patch').write_text(patch)
print('APPROVAL_PATCH_PREPARED', len(sealed), 'sources', now, flush=True)
