import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT/'CaMRA/navigation'
OUT = ROOT/'.worklog-testing/NAV-067/clean-v2'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg = json.loads((NAV/'config/nav066_decision_prereg_v2.json').read_bytes())
protected = {str(NAV/k):v for k,v in reg['implementation_hashes'].items()}
protected.update(reg['protected_hashes'])
protected.update({r['checkpoint']:r['checkpoint_sha256'] for r in reg['selected_candidates']})
for name,digest in protected.items():
    assert sha(name)==digest, name
base = json.loads(subprocess.check_output(['git','show','HEAD:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
path = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
raw = path.read_text()
plan = json.loads(raw)
for a,b in zip(base['tasks'],plan['tasks'],strict=True):
    assert a['id']==b['id'] and a['status']==b['status'] and a['depends_on']==b['depends_on']
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-067': assert a==b
sources = ['src/camra_navigation/envs/physical_v2.py','src/camra_navigation/envs/clean_v2.py','tests/test_physical_v2.py']
for rel in sources: ast.parse((NAV/rel).read_text(),filename=rel)
assert '16 passed' in (OUT/'basic-tests.log').read_text()
report = {'task':'NAV-067','status':'PARTIAL_BASIC_VERIFIED_NOT_PHYSICS_VALIDATED',
    'timestamp':datetime.now().astimezone().isoformat(),
    'basic_tests':16,'protected_hashes_checked':len(protected),
    'protected_originals_unchanged':True,'plan_states_unchanged':True,
    'sources':{r:sha(NAV/r) for r in sources},
    'new_physical_episodes':0,'training_updates':0,'new_budget_queries':0,
    'limitations':['No real Isaac run yet.','New production runner/candidate binding still required.',
       'Same-configuration repeatability and real controlled cases need registered independent quota.',
       'Stub counterexamples are basic tests only. NAV-067 remains false.']}
report_path = NAV/'reports/nav067_substep_preparation_v1.json'
if report_path.exists():
    saved=json.loads(report_path.read_bytes())
    assert saved['sources']==report['sources'] and saved['protected_hashes_checked']==report['protected_hashes_checked']
    report=saved  # preserve the report from the interrupted patch preparation
else:
    with report_path.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')
record={'timestamp':report['timestamp'],'summary':'NAV-067阶段保存：200Hz事件基础实现及16项基本检查通过，真实接线未验收',
  'actions':['新增隔离physical_v2与clean_v2：显式obs-v7/45秒、实际物理子步、不可逆安全历史、真实终止快照、部分周期模型计费和活跃掩码。',
    'Isaac适配按v2 action_manager每子步施加PD，使用实际观测命令及时钟，绕开gym.step自动复位；此处仅代码准备，尚未在真实Isaac运行。',
    '主agent完成中间穿障后退出、危险历史、旋转扫掠、同时到达碰撞、异步结束、部分周期计费和缺失/陈旧命令拒绝等16项基本测试。',
    f'校验原注册绑定的{len(protected)}个源码/配置/资源/模型hash不变，原67项顺序/依赖/状态及全部旧记录前缀保留。'],
  'files_changed':[str(NAV/r) for r in sources]+[str(NAV/'reports/nav067_substep_preparation_v1.json'),str(path)],
  'commands':['tmux nav064-r14-repair-20261115:nav067-clean-v2; cwd=/home/caohy/repositories/Paper/CaMRA/navigation; PYTHONPATH=src:scripts:../calibration/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B -m pytest -q tests/test_physical_v2.py'],
  'results':['16项基本测试通过；三个新文件语法解析通过；无新训练/物理回合/W查询。','本次仅基础实现检查，尚未接通新版完整生产入口，不构成NAV-067或物理性能通过。'],
  'evidence':['CaMRA/navigation/reports/nav067_substep_preparation_v1.json','.worklog-testing/NAV-067/clean-v2/basic-tests.log'],
  'remaining_work':['新版候选绑定与可执行生产入口；实际clean参数/资产/碰撞体读回。','有效登记后执行6回合pilot、真实受控反例和同配置重复性，受资源边界约束。','依照所有局部及全局标准验收后再进入NAV-043。'],'blockers':[]}
block=json.dumps(record,ensure_ascii=False,indent=2)
block='\n'.join('   '+line for line in block.splitlines())
start=raw.index('"id": "NAV-067"')
end=raw.index('"inserted_by"',start)
lines=raw[start:end].splitlines()
last=lines[-6:-1]
assert last[-1].strip()=='],' and last[-2].strip()=='}',last
patch='*** Begin Patch\n*** Update File: '+str(path)+'\n@@\n'
patch+=''.join(' '+line+'\n' for line in last[:-2])
patch+='-'+last[-2]+'\n+'+last[-2]+',\n'
patch+=''.join('+'+line+'\n' for line in block.splitlines())
patch+=' '+last[-1]+'\n*** End Patch\n'
(OUT/'checkpoint.patch').write_text(patch)
print(json.dumps(report,ensure_ascii=False,indent=2))
