"""Review preparation evidence only. Never run the real fit or create approval."""
from datetime import datetime
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
assert Path.cwd() == ROOT and os.environ.get('TMUX')
assert socket.gethostname().split('.')[0] == 'tiger11'
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
BASE = '77cd7376c200bf8a18bf4672e75d040f01b7e7a4'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == BASE
def load(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('basic','proposal','preflight'):
    assert (HERE/(name+'.exit')).read_text().strip() == '0'
log = (HERE/'basic.log').read_text()
assert 'Ran 18 tests' in log and log.rstrip().endswith('OK')
source = ROOT/'CaMRA/navigation/scripts/nav043_motion_response.py'
spec = importlib.util.spec_from_file_location('nav043_response_prep',source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
proposal = module.PROTOCOL
assert sha(proposal) == Path(str(proposal)+'.sha256').read_text().strip()
p = load(proposal)
assert p['status'] == 'PROPOSED' and not module.APPROVAL.exists()
assert not Path(str(module.APPROVAL)+'.sha256').exists()
assert not Path(p['output_root']).exists()
for group in ('implementation_hashes','report_hashes'):
    for path,digest in p[group].items():assert module.sha(path) == digest
protected = module.protected_identities(p)
for path,digest in protected.items():assert module.sha(path) == digest
raw = {path:digest for t in p['traces'] for path,digest in t['hashes'].items()}
assert len(raw) == 90
for path,digest in raw.items():assert module.sha(path) == digest
pre = load(HERE/'preflight.json')
assert pre['protocol_sha256'] == sha(proposal)
assert pre['status'] == 'PREFLIGHT_ONLY_NO_MODEL_FITS'
assert pre['actual_response_model_solves'] == 0 and pre['protected_hashes_checked'] == 295
assert len(pre['traces']) == 18 and sum(t['physics_ticks'] for t in pre['traces']) == 158797
assert sum(t['fit_pairs'] for t in pre['traces']) == 7921
assert sum(t['partial_ticks'] != 0 for t in pre['traces']) == 5
plan_path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
before_text = plan_path.read_text()
plan = json.loads(before_text)
old = json.loads(subprocess.check_output(['git','show',BASE+':'+str(plan_path.relative_to(ROOT))],text=True))
task = next(t for t in plan['tasks'] if t['id']=='NAV-043')
old_task = next(t for t in old['tasks'] if t['id']=='NAV-043')
assert len(old_task['work_records']) == 25 and len(task['work_records']) == 26
copy = json.loads(before_text)
next(t for t in copy['tasks'] if t['id']=='NAV-043')['work_records'] = old_task['work_records']
assert copy == old and sum(t['status'] for t in plan['tasks']) == 55
assert task['status'] is False and task['verification_result'] == ''
assert before_text == json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
paths = [source,ROOT/'CaMRA/navigation/tests/test_nav043_motion_response.py',proposal,
         Path(str(proposal)+'.sha256'),HERE/'basic.log',HERE/'basic.exit',HERE/'preflight.json',
         HERE/'preflight.exit',HERE/'prepare_proposal.py',Path(__file__),
         ROOT/'CaMRA/navigation/docs/nav043_motion_response_preparation_20261119.md']
evidence = {'status':'PASS_PREPARATION_ONLY_REAL_FIT_NOT_STARTED',
            'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),
            'pane':os.environ.get('TMUX_PANE'),'proposal_sha256':sha(proposal),
            'basic_tests':18,'synthetic_only_axis_solves':3,'real_response_model_solves':0,
            'trace_count':18,'raw_file_hashes':90,'protected_identities':len(protected),
            'saved_ticks':158797,'complete_adjacent_interval_pairs':7921,
            'approval_exists':False,'formal_output_exists':False,
            'NAV043_passed':False,'G5_passed':False,
            'evidence_hashes':{str(p):sha(p) for p in paths}}
module.write_new(HERE/'preparation-ready.json',evidence)
task['work_records'].append({
 'timestamp':datetime.now().astimezone().isoformat(),
 'summary':'NAV-043响应模型具体协议及隔离入口准备完成，真实拟合未获批准且未开始',
 'actions':['固定18条已暴露保存integration轨迹及90原始身份，整图留出三折、六参数响应模型、九求解/CPU一小时和失效即停规则；只准备新提案，不改原正式验收。',
            '主agent在tiger11/tmux %626执行18项合成/反例检查通过；三次线性求解仅用于人工合成矩阵。%628只读预检18条来源/时钟/导航命令持有及295保护身份通过，真实模型求解0。',
            '从实际warmup位姿仅对齐一次、响应历史置零自由递推；SE(2)区间Euler航向响应与机身wz区分，部分末区间不拟合、未完成前缀不外推。运行入口缺少独立批准时在读取拟合数据前拒绝。',
            '全局复核：RC-PPO/原生产actor双critic及价值预算决定调用链、固定v2/旧alpha/delta/45秒/评分和数据隔离保持。新入口未接入生产，不把基本检查、分组诊断提案或旧true当作新模型验收；NAV-043及下游G5保持未通过。'],
 'files_changed':['CaMRA/navigation/scripts/nav043_motion_response.py','CaMRA/navigation/tests/test_nav043_motion_response.py',
                  'CaMRA/navigation/config/nav043_motion_response_proposal_v1.json','CaMRA/navigation/config/nav043_motion_response_proposal_v1.json.sha256',
                  'CaMRA/navigation/docs/nav043_motion_response_preparation_20261119.md','CaMRA/navigation/docs/nav043_motion_model_route_proposal_20261119.md',
                  'AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
                  '.worklog-testing/NAV-043/motion-response-v1/','doc/plan/20261109_camra_navigation_implementation.json'],
 'commands':['tiger11/tmux %626：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -m unittest discover -s CaMRA/navigation/tests -p test_nav043_motion_response.py -v',
             'tiger11/tmux %627 prepare_proposal.py仅生成patch，再以apply_patch落盘；%628同一解释器执行nav043_motion_response.py --preflight .worklog-testing/NAV-043/motion-response-v1/preflight.json，不运行--run。',
             'tiger11/tmux verify_preparation.py核对证据/来源/保护身份和相对77cd737仅追加；后续同机tmux运行legacy update、最终追加及差分/白名单检查，本地提交，不推送。'],
 'results':['18基本检查通过；只读预检18条/158797保存tick/7921相邻完整区间对，五条部分末区间保留。90原始身份与295保护身份不变，精确PROPOSED协议pin9e557462。',
            '真实模型求解/新应用/物理或名义回合/W/actor或critic更新/校准拟合均0；独立APPROVED及正式输出目录不存在。',
            '计划1.4.1、55/67、NAV-043=false；原G4/效率/物理适用性失败保留，下游不启动。准备检查不表示模型有效或G5通过。'],
 'evidence':['.worklog-testing/NAV-043/motion-response-v1/basic.log','.worklog-testing/NAV-043/motion-response-v1/basic.exit',
             '.worklog-testing/NAV-043/motion-response-v1/preflight.json','.worklog-testing/NAV-043/motion-response-v1/preflight.exit',
             '.worklog-testing/NAV-043/motion-response-v1/preparation-ready.json','CaMRA/navigation/docs/nav043_motion_response_preparation_20261119.md'],
 'remaining_work':['等待新拟合范围明确确认后，单独登记APPROVED并仅执行封存的三折/九求解/CPU一小时诊断；不默认重试、换模型或转用旧额度。',
                   '根据真实诊断再评审新名义模型接口/观测、45秒横向转向残余校准、地图适用性、上层训练和物理检验；各自登记范围及额度，不沿用旧验收直接放行。'],
 'blockers':['主计划uncertainty_policy及legacy执行规则要求新范围明确；当前新运动响应拟合尚未批准，不将推荐方案自动视为批准。']})
after = json.dumps(plan,ensure_ascii=False,indent=2)+'\n'
lines = list(difflib.unified_diff(before_text.splitlines(),after.splitlines(),n=3))
patch = ['*** Begin Patch','*** Update File: '+str(plan_path)]
patch.extend('@@' if line.startswith('@@') else line for line in lines[2:])
patch.append('*** End Patch')
with (HERE/'preparation-close.patch').open('x') as stream:stream.write('\n'.join(patch)+'\n')
print(json.dumps(evidence,ensure_ascii=False),flush=True)
