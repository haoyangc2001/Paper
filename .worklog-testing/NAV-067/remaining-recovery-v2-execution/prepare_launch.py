"""Register the user's explicit recovery approval, verify, commit, then start once."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
from nav067_sources import load, sha, write_new
import run_nav067_remaining_validation as V1
import run_nav067_remaining_recovery_v2 as V2

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def add_files(files):
    patch='*** Begin Patch\n'
    for path,content in files.items():
        assert not path.exists(),str(path)
        patch+='*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in content.splitlines())
    patch+='*** End Patch\n'
    subprocess.run(['apply_patch'],input=patch,text=True,cwd=ROOT,check=True)

assert socket.gethostname().split('.')[0]=='tiger11'
assert os.environ.get('TMUX_PANE')=='%600'
assert os.path.realpath(sys.executable)==os.path.realpath(V1.PYTHON)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='4a64e64cdcd2f8d3e8a9b1b9555f17f7f70bd296'
assert not V2.APPROVAL.exists()
assert not (V1.EXPECTED_ROOT/'C01/attempt2').exists()
assert not (V1.EXPECTED_ROOT/'RECOVERY_V2_OWNER.json').exists()
assert not Path('/proc/'+str(load(V1.EXPECTED_ROOT/'C01/PROCESS.json')['pid'])).exists()
proposal=load(V2.PROPOSAL)
assert sha(V2.PROPOSAL)=='e8b70cb9d7c81177d576189772b5aecb6054410df71dde867aa5ad1c4c034cac'
for key,value in proposal['fixed_environment'].items():
    os.environ[key]=value
gpu=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu.split(',')[2].strip())>=6000,gpu
inventory=[]
for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().strip('\0').split('\0'):
    path=ROOT/name
    inventory.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
write_new(HERE/'initial-untracked-inventory.json',{'at':now(),'files':inventory,'count':len(inventory),'bytes':sum(x['bytes'] for x in inventory)})
plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan_text=plan_path.read_text()
plan=json.loads(plan_text)
write_new(HERE/'initial-plan.json',plan)
task=next(t for t in plan['tasks'] if t['id']=='NAV-067')
assert plan['document']['version']=='1.4.1' and not task['status']
assert sum(t['status'] for t in plan['tasks'])==54
assert next(t for t in plan['tasks'] if t['id']=='NAV-066')['status']
approval={'schema':'camra-nav067-remaining-recovery-approval-v2','task':'NAV-067','status':'APPROVED',
    'at':now(),'approval_source':'用户在收到明确一次恢复及累计启动9→10、总GPU4小时不变的说明后回复：确认，继续推进吧，另外为什么本地有那多没有提交的修改，是不需要提交吗？',
    'proposal_sha256':sha(V2.PROPOSAL),'proposal_path':str(V2.PROPOSAL.relative_to(ROOT)),
    'limits':proposal['limits'],'execution_order':proposal['execution_order'],
    'failure_policy':proposal['failure_policy'],'original_failed_attempt_retained':True,
    'no_test_subagent':True,'remote_push_authorized':False}
approval_text=json.dumps(approval,ensure_ascii=False,indent=2)+'\n'
approval_sha=hashlib.sha256(approval_text.encode()).hexdigest()
record={'timestamp':now(),'summary':'NAV-067恢复v2开工：用户已确认一次C01恢复及累计10应用；同时核对并补交历史未跟踪证据',
 'actions':['完整解析当前1.4.1计划、67任务及历史记录，重读根约束/legacy/当前交接及实际恢复代码；按依赖继续NAV-067。',
    '依据用户明确确认，单独登记APPROVED及pin，原PROPOSED/失败/源代码保持；本项用于补齐固定v2物理事件接入验收，不重训。',
    '现场核对tiger11/tmux/绝对Python/GPU6及既有输出；记录未跟踪文件逐项路径、大小、hash，后续补交必要报告与归档索引。'],
 'files_changed':[str(V2.APPROVAL.relative_to(ROOT)),str(V2.APPROVAL.relative_to(ROOT))+'.sha256',str(HERE.relative_to(ROOT))+'/', 'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md'],
 'commands':['tiger11/tmux nav064-r14-repair-20261115:nav067-remaining-plan %600；cwd=/home/caohy/repositories/Paper；'+sys.executable+' -B '+str(Path(__file__)),
    '新批准后先run_nav067_remaining_recovery_v2.py --check、legacy update校验并提交，再执行同入口一次。'],
 'results':['上游NAV-066为明确接受局限后的true，旧科学FAIL保留；NAV-067仍false、54/67。',
    '累计最多10应用/单GPU串行4小时/60000环境子步，含原失败118.444秒和408预留；精确57830，受控全过后才六导航重复。',
    '固定原两模型/45秒/obs-v7/200-50-10Hz/J/全套v2，原三生产源码和295保护输入由生产context复核；本次批准不含NAV-043/final_*、训练/W查询/共享框架变更/推送。'],
 'evidence':[str(V2.PROPOSAL.relative_to(ROOT)),str(V2.APPROVAL.relative_to(ROOT)),str((HERE/'initial-untracked-inventory.json').relative_to(ROOT)),str((HERE/'preflight.json').relative_to(ROOT))],
 'remaining_work':['核验新批准、提交后只启动一次恢复；逐项审阅实际受控/重复证据和上下游一致性后才判NAV-067。','核对历史未提交文件，补交应保留的报告，解释大快照及备份的保留方式。'],
 'blockers':[]}
last=task['work_records'][-1]
old='\n'.join('        '+s for s in json.dumps(last,ensure_ascii=False,indent=2).splitlines())
new='\n'.join('        '+s for s in json.dumps(record,ensure_ascii=False,indent=2).splitlines())
assert plan_text.count(old)==1
patch='*** Begin Patch\n*** Update File: '+str(plan_path)+'\n@@\n'+''.join('-'+s+'\n' for s in old.splitlines())+''.join('+'+s+'\n' for s in (old+',\n'+new).splitlines())+'*** End Patch\n'
add_files({V2.APPROVAL:approval_text,Path(str(V2.APPROVAL)+'.sha256'):approval_sha+'\n'})
subprocess.run(['apply_patch'],input=patch,text=True,cwd=ROOT,check=True)
for path,title,body in [
 (ROOT/'AGENTS.md','# Agent 接手说明','## 最新补充（2026-11-19，NAV-067恢复v2已确认）\n\n用户已明确确认一次C01恢复及累计应用上限9→10、总GPU4小时不变。独立APPROVED登记为`CaMRA/navigation/config/nav067_remaining_validation_recovery_v2.json`及pin；只用原目录C01/attempt2，C01–C06全过后继续原未启动六导航重复。原失败、PROPOSED和v1源码不改，固定v2/GPU6/tiger11/tmux，首个新实际失败仍保存停下。新增进度以实际输出为准，不把本次批准写成通过；NAV-067=false、54/67，NAV-043不开启。当前另核对历史未跟踪报告/快照，补齐必要证据提交；未授权推送。下方未批准描述为历史时点。'),
 (NAV/'docs/implementation_handoff.md','# 上层导航实施交接','## 2026-11-19 NAV-067恢复v2已获用户确认，开始一次执行\n\n用户已确认隔离plane/grid修复及一次C01/attempt2恢复，累计应用上限10、GPU6串行累计4小时含旧118.444秒、60000环境子步含旧408预留；精确预留57830。APPROVED登记及pin另存`config/nav067_remaining_validation_recovery_v2.json`，原提案和失败保持。先核验生产context和保护身份、提交，再执行恢复入口；C01–C06全过才继续原六导航重复，任一新实际证据/运行/轨迹失败保存即停。NAV-067仍false、54/67，NAV-043未执行，无训练/W查询/final_*或推送。主agent按legacy验证，无测试子agent。并行只读核对工作树历史未跟踪材料，补交报告和归档说明；真实执行结论随后追加。下方未批准为先前时点。')]:
    text=path.read_text()
    assert text.startswith(title+'\n\n')
    patch='*** Begin Patch\n*** Update File: '+str(path)+'\n@@\n '+title+'\n \n'+''.join('+'+s+'\n' for s in (body+'\n\n').splitlines())+'*** End Patch\n'
    subprocess.run(['apply_patch'],input=patch,text=True,cwd=ROOT,check=True)
context=V2.context()
check_command=[sys.executable,'-B',str(NAV/'scripts/run_nav067_remaining_recovery_v2.py'),'--check']
check=subprocess.run(check_command,cwd=ROOT,capture_output=True,text=True)
write_new(HERE/'production-check.json',{'at':now(),'command':check_command,'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr})
assert check.returncode==0
validator=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update']
subprocess.run(validator,cwd=ROOT,check=True)
write_new(HERE/'preflight.json',{'at':now(),'host':socket.getfqdn(),'python':sys.executable,'tmux':os.environ['TMUX'],'pane':os.environ['TMUX_PANE'],
    'GPU6':gpu,'environment':proposal['fixed_environment'],'approval_sha256':sha(V2.APPROVAL),'context':'PASS','new_applications_before_commit':0})
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True).strip()
paths=[str(V2.APPROVAL.relative_to(ROOT)),str(V2.APPROVAL.relative_to(ROOT))+'.sha256','AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md',str(plan_path.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),*[str(p.relative_to(ROOT)) for p in HERE.glob('*.json')]]
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','chore(nav): register approved one-shot NAV-067 recovery'],cwd=ROOT,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
V2.context()
with (HERE/'coordinator.log').open('x') as stream:
    process=subprocess.Popen(proposal['command'],cwd=ROOT,env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
write_new(HERE/'launch.json',{'at':now(),'pid':process.pid,'command':proposal['command'],'git_head':head,'cwd':str(ROOT),'tmux':os.environ['TMUX'],'pane':os.environ['TMUX_PANE'],'approval_sha256':sha(V2.APPROVAL),'no_automatic_retry':True})
print(json.dumps({'started_once':True,'pid':process.pid,'commit':head,'approval_sha256':sha(V2.APPROVAL)},ensure_ascii=False),flush=True)
