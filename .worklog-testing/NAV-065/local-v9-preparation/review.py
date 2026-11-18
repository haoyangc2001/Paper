"""Verify v9 preparation and generate append-only documentation patches."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
baseline = read(HERE / 'baseline.json')
current = read(PLAN)
def verify_plan(extra):
    assert {k:v for k,v in current.items() if k != 'tasks'} == {k:v for k,v in baseline['plan'].items() if k != 'tasks'}
    for old, new in zip(baseline['plan']['tasks'], current['tasks'], strict=True):
        if old['id'] == 'NAV-065':
            assert {k:v for k,v in old.items() if k != 'work_records'} == {k:v for k,v in new.items() if k != 'work_records'}
            assert new['work_records'][:len(old['work_records'])] == old['work_records']
            assert len(new['work_records']) == len(old['work_records']) + extra
        else:
            assert new == old
verify_plan(1)
report_path = NAV / 'reports/nav065_local_probe_preflight_v9.json'
r = read(report_path)
assert r['source_train_gate_exact_match'] and r['source_models_unchanged']
assert (r['available_train_episodes'], r['available_train_steps'], r['local_rows']) == (162, 59877, 25)
assert r['false_feasible_breakdown'] == {'total':13, 'reach_target_positive':9, 'cost_only_target_block':4}
assert r['optimizer_steps'] == r['new_rollouts'] == r['development_scoring_rounds'] == 0
assert not r['scientific_pass'] and not r['full_train_gate']['pass']
assert '16 passed' in (HERE / 'basic-attempt1.log').read_text()
proposal_path = NAV / 'config/nav065_local_probe_proposal_v9.json'
p = read(proposal_path)
assert p['approved'] is False and p['resource_limit']['optimizer_trials_total_max'] == 400
assert not (NAV / 'config/nav065_local_probe_prereg_v9.json').exists()
assert not Path('/pub/data/caohy/work/camra_navigation/nav065_local_probe_v9').exists()
v2 = read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
assert len(v2) == 33
for name,digest in v2.items():
    assert sha(Path(name)) == digest
protected = ['RC-PPO', 'HiRA', 'manuscripts', 'CaMRA/locomotion', 'CaMRA/calibration']
assert not subprocess.check_output(['git','diff','--name-only','--',*protected],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'HiRA',text=True).strip()
evidence_paths = [report_path, proposal_path, NAV/'scripts/probe_nav065_local.py',
    NAV/'tests/test_nav065_local_probe.py', NAV/'docs/nav065_local_probe_v9_20261118.md',
    HERE/'basic-attempt1.log', HERE/'preflight-attempt1.log', HERE/'context.py', HERE/'check.py']
report = {'schema':'camra-nav065-local-preparation-review-v9', 'host':socket.getfqdn(),
    'tmux':os.environ['TMUX'], 'python':sys.executable, 'base_git':baseline['git'],
    'preparation_pass':True, 'basic_tests_passed':16, 'source_gate_exact_match':True,
    'available_train_episodes':162, 'available_train_steps':59877, 'local_rows':25,
    'false_feasible':13, 'reach_target_positive':9, 'cost_only_target_block':4,
    'formal_optimizer_steps':0, 'new_rollouts':0, 'development_scoring_rounds':0,
    'synthetic_basic_tests_use_frozen_candidate':False,
    'source_models_unchanged':True, 'v2_files_verified':33,
    'proposal_approved':False, 'requested_new_optimizer_steps':400,
    'scientific_pass':False, 'original_G4':'FAIL', 'NAV-065':False,
    'NAV-066_started':False, 'completed_tasks':52,
    'evidence':{str(path.relative_to(ROOT)):sha(path) for path in evidence_paths},
    'limitations':['Read-only preparation and synthetic checks are not candidate training or scientific acceptance.',
        'Selected train mistakes are an intentionally biased local fitting probe; no generalization claim.',
        'Full feature rank and frozen-head bounds do not guarantee numerical trainability or sufficient full-task representation.']}
with (NAV/'reports/nav065_local_probe_v9_preparation_review.json').open('x') as f:
    json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
summary = ('用户指定legacy按前述分析推进；主agent完成16项基本检查和v8末次162条train/59877步只读核验，无测试子agent。'
    '原13次误放行分为9次到达分量应拦截、4次仅成本应拦截；固定13错例及相邻真实可行点去重共25起点。'
    '到达末层特征在报告容差下行秩25，不等于全任务观测充分或保证拟合成功。隔离v9入口和未批准400步单候选局部拟合提案已备妥；'
    '正式更新/新回合/开发评分0，原目标/架构/actor/v2及原v5 train/G4不变。'
    '准备通过不等于部署通过；原G4 FAIL、52/67、NAV-065=false和NAV-066未启动保持。新额度待明确确认，不重用v8余量。')
changes = []
def change(path, old, new):
    changes.append('*** Update File: '+str(path)+'\n@@\n'+
        '\n'.join('-'+line for line in old.splitlines())+'\n'+
        '\n'.join('+'+line for line in new.splitlines())+'\n')
for path, prefix, link in [(NAV/'README.md','2026-11-18最新准备：','docs/'),
        (NAV/'docs/implementation_handoff.md','## 2026-11-18 NAV-065 v9准备：局部可拟合性诊断\n\n','')]:
    heading = path.read_text().splitlines()[0]
    addition = heading+'\n\n'+prefix+summary+'\n\n详见[v9方案]('+link+'nav065_local_probe_v9_20261118.md)。以下为历史时点。'
    change(path, heading, addition)
agents = ROOT/'AGENTS.md'
anchor = '## 前置状态：v2 选择与历史校准复核（2026-11-09）'
change(agents,anchor,'### NAV-065最新补充（2026-11-18，局部诊断v9准备）\n\n'+summary+
    '见[方案](CaMRA/navigation/docs/nav065_local_probe_v9_20261118.md)。\n\n'+anchor)
record = {'timestamp':datetime.now().astimezone().isoformat(),
    'summary':'NAV-065/1.3.1 v9准备收尾：25点局部诊断入口与16基本检查通过，正式拟合待新额度',
    'actions':['直接复用v8末次来源和原共同停止，重建162条train的原完整train_gate，逐字段匹配。',
        '按固定规则将13误放行与同地图最近真实可行预算配对，去重25起点；测量原误判分型及末层特征，不拟合真实模型。',
        '实现隔离的批准受控局部拟合入口、逐步日志、原全train复查、深拷贝/actor保护与诊断模型CPU重载；完成16项基本检查并同步交接。'],
    'files_changed':['doc/plan/20261109_camra_navigation_implementation.json','AGENTS.md','CaMRA/navigation/README.md',
        'CaMRA/navigation/docs/implementation_handoff.md','CaMRA/navigation/scripts/probe_nav065_local.py',
        'CaMRA/navigation/tests/test_nav065_local_probe.py','CaMRA/navigation/config/nav065_local_probe_proposal_v9.json',
        'CaMRA/navigation/docs/nav065_local_probe_v9_20261118.md',
        'CaMRA/navigation/reports/nav065_local_probe_preflight_v9.json',
        'CaMRA/navigation/reports/nav065_local_probe_v9_preparation_review.json',str(HERE.relative_to(ROOT))+'/'],
    'commands':['host=tiger11.ios.ac.cn；tmux nav064-r14-repair-20261115:nav065-v9-preparation；cwd='+str(NAV)+'；Python='+sys.executable,
        '<Python> -B -m pytest -q tests/test_nav065_local_probe.py',
        '<Python> -B -u scripts/probe_nav065_local.py --diagnose reports/nav065_local_probe_preflight_v9.json',
        'legacy validate_worklog.py doc/plan/20261109_camra_navigation_implementation.json --mode update'],
    'results':[summary,'实际生产原目标/原门禁由新版入口直接复用；局部拟合的原数值阈值与符号检查仅作诊断，不能代替原v5 train/G4。',
        '拟申请一次full_s0深拷贝、25起点全批次分层MSE、Adam lr=1e-3最多400新更新，新回合/开发评分0；过程与新数字额度未批准。',
        '局部成功仅支持这些已见点可拟合，失败也不证明架构不可能；本轮不实现或训练其他架构，不自动扩展、重启或发布部署模型。',
        '当前任务所有历史状态/定义/verification_result/旧记录保留；本项与上下游原理/版本检查通过，NAV-065整体科学验收仍不满足。'],
    'evidence':['CaMRA/navigation/reports/nav065_local_probe_preflight_v9.json',
        'CaMRA/navigation/reports/nav065_local_probe_v9_preparation_review.json',str(HERE/'basic-attempt1.log'),
        str(HERE/'preflight-attempt1.log'),'CaMRA/navigation/reports/nav065_pilot_refit_v8_review.json'],
    'remaining_work':['新具体额度确认后独立登记pin、执行一次局部拟合并检查全train影响，再按证据评审下一动作。',
        '原六候选完整train、development/G4及部署身份全部达标后才完成NAV-065并允许NAV-066。'],
    'blockers':['新400步仅为未批准提案，不能将当前准备授权当作额外优化额度，不能挪用v8未执行边界预算。']}
text=PLAN.read_text();needle='        }\n      ],\n   "test_reports": ['
# The opening record has its own indentation, so append at the actual array end.
nav=text.index('"id": "NAV-065"');end=text.index('\n      ],\n   "test_reports": [',nav)
tail=text.rfind('\n}',nav,end)
assert tail >= 0
old=text[tail:end+len('\n      ],\n   "test_reports": [')]
new='\n},\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n      ],\n   "test_reports": ['
change(PLAN,old.lstrip('\n'),new.lstrip('\n'))
with (HERE/'closing.patch').open('x') as f:
    f.write('*** Begin Patch\n'+''.join(changes)+'*** End Patch\n')
print(json.dumps({'preparation_pass':True,'basic_tests':16,'local_rows':25,
    'formal_updates':0,'new_optimizer_steps_requested':400,'proposal_approved':False},ensure_ascii=False))
