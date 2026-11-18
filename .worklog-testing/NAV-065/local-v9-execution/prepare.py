"""Record the approved single v9 attempt and generate narrow registration patches."""
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
sys.path.insert(0, str(NAV/'scripts'))
sys.path.insert(0, str(NAV/'src'))
import probe_nav065_local as L

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
p = read(PLAN)
states = {t['id']: t['status'] for t in p['tasks']}
assert len(states) == 67 and sum(states.values()) == 52
assert next(t['id'] for t in p['tasks'] if not t['status'] and all(states[x] for x in t['depends_on'])) == 'NAV-065'
assert not git('diff','--name-only'), 'preserve existing tracked modifications'
base = git('rev-parse','HEAD')
assert read(PLAN) == json.loads(subprocess.check_output(['git','show',base+':'+str(PLAN.relative_to(ROOT))],cwd=ROOT))
with (HERE/'validation-before.log').open('x') as f:
    result = subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
assert result.returncode == 0
old = read(NAV/'reports/nav065_local_probe_v9_preparation_review.json')
for name,digest in old['evidence'].items():
    assert sha(ROOT/name) == digest, name
v2 = read(NAV/'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files']
assert len(v2) == 33
for name,digest in v2.items():
    assert sha(Path(name)) == digest, name
assert not L.R.WORK.joinpath('nav065_local_probe_v9').exists(), 'inspect existing attempt; never restart blindly'
assert not (NAV/'config/nav065_local_probe_prereg_v9.json').exists()
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.free','--format=csv,noheader'],text=True).strip()
processes = subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
assert not any('probe_nav065_local.py --registration' in line for line in processes)
protocol = read(L.PROPOSAL)
protocol.update(approved=True, status='APPROVED', authorization_reference='用户2026-11-18本轮明确回复“确认”：批准已提交4c28613的NAV-065 v9一次full_s0副本/固定25个已有train起点、Adam lr=1e-3最多400新更新、0新回合/0开发评分；沿用legacy主agent验证和此前同机tiger11/tmux例外，不自动扩展。')
protocol['implementation_hashes'] = {name:sha(NAV/name) for name in L.BOUND_INPUTS}
protocol_text = json.dumps(protocol,ensure_ascii=False,indent=2)+'\n'
pin = hashlib.sha256(protocol_text.encode()).hexdigest()
now = datetime.now().astimezone().isoformat()
context = {'timestamp':now,'host':socket.getfqdn(),'tmux':os.environ['TMUX'],'cwd':str(ROOT),
    'python':sys.executable,'python_version':sys.version,'base_git':base,'gpu':gpu,
    'status':git('status','--short'),'plan_sha256':sha(PLAN),'v2_files_verified':33,
    'approval_pin':pin,'registration':'config/nav065_local_probe_prereg_v9.json',
    'output':str(L.R.WORK/'nav065_local_probe_v9'),'optimizer_cap':400,
    'ssh_exception':'Existing explicitly authorized same-host tiger11/tmux execution; no machine change.',
    'task_records_read':len(next(t for t in p['tasks'] if t['id']=='NAV-065')['work_records'])}
with (HERE/'context.json').open('x') as f:
    json.dump(context,f,ensure_ascii=False,indent=2);f.write('\n')
record = {'timestamp':now,'summary':'NAV-065/1.3.1 v9确认后开工：批准一次25点局部拟合、最多400新更新',
    'actions':['重新完整读取AGENTS、legacy规则与计划JSON，核对首个可执行任务NAV-065、NAV-064依赖和既有记录；现场确认tiger11/tmux/绝对解释器、来源和33项v2身份。',
        '登记用户本轮“确认”，冻结与4c28613提案完全一致的新批准文件、源码/配置hash和pin；本次仅在full_s0副本上执行一次。'],
    'files_changed':['doc/plan/20261109_camra_navigation_implementation.json','CaMRA/navigation/config/nav065_local_probe_prereg_v9.json','CaMRA/navigation/config/nav065_local_probe_prereg_v9.json.sha256',str(HERE.relative_to(ROOT))+'/'],
    'commands':[sys.executable+' -B '+str(Path(__file__)),
        'host=tiger11.ios.ac.cn；tmux=nav064-r14-repair-20261115:nav065-v9-execution；cwd='+str(NAV)],
    'results':['服务于固定actor双critic预算判别可靠性，参考RC-PPO Phase II；不改actor/目标/W/beta/架构/数据划分/原v5 train/G4或固定v2。',
        '复用已选25起点做原MSE局部诊断，最多400步，新增回合/开发评分0；拟合前后直接复用原162回合train_gate，局部成功不放行部署或NAV-066。',
        '上游为v8末次checkpoint '+protocol['source_checkpoint_sha256']+'；批准pin='+pin,
        '16项基本检查证据及源码hash未变，不重复合成测试；后续验收检查真实更新次数、末次预测/门禁、CPU重载与六原模型/actor保护。'],
    'evidence':['CaMRA/navigation/config/nav065_local_probe_proposal_v9.json','CaMRA/navigation/reports/nav065_local_probe_v9_preparation_review.json',str(HERE/'context.json')],
    'remaining_work':['应用批准登记与开工记录，校验计划并按固定入口执行一次；主agent验收后追加结果和下游限制。'],
    'blockers':[]}
text=PLAN.read_text()
start=text.index('"id": "NAV-065"')
end=text.index('\n      ],\n   "test_reports": [',start)
anchor_start=text.rfind('\n  "blockers": [',start,end)
assert anchor_start > start
old_tail=text[anchor_start+1:end+len('\n      ],\n   "test_reports": [')]
new_tail=old_tail[:old_tail.rfind('\n}')]+'\n},\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n      ],\n   "test_reports": ['
patch='*** Begin Patch\n'
for path,content in [(NAV/'config/nav065_local_probe_prereg_v9.json',protocol_text),(NAV/'config/nav065_local_probe_prereg_v9.json.sha256',pin+'\n')]:
    patch+='*** Add File: '+str(path)+'\n'+'\n'.join('+'+line for line in content.splitlines())+'\n'
patch+='*** Update File: '+str(PLAN)+'\n@@\n'+'\n'.join('-'+line for line in old_tail.splitlines())+'\n'+'\n'.join('+'+line for line in new_tail.splitlines())+'\n*** End Patch\n'
with (HERE/'approved-opening.patch').open('x') as f:
    f.write(patch)
print(json.dumps({k:v for k,v in context.items() if k not in ('gpu','status')},ensure_ascii=False))
