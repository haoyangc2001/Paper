"""Save terminal failure, append original plan history, validate and commit locally."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
from nav067_sources import load, sha, write_new, read_manifest
import run_nav067_remaining_validation as V1
import run_nav067_remaining_recovery_v2 as V2

assert os.environ.get('TMUX_PANE')=='%599'
assert load(HERE/'recovery-verification.json')['count']==26
assert not V2.APPROVAL.exists()
assert not (V1.EXPECTED_ROOT/'C01/attempt2').exists()
assert not Path('/proc/'+str(load(V1.EXPECTED_ROOT/'C01/PROCESS.json')['pid'])).exists()
manifest,prereg,nets=read_manifest()
for name,digest in load(V1.EXECUTION)['source_hashes'].items():
    assert sha(ROOT/name)==digest
proposal=load(V2.PROPOSAL)
for name,digest in proposal['source_hashes'].items():
    assert sha(ROOT/name)==digest
for name,digest in proposal['installed_source_hashes'].items():
    assert sha(name)==digest
for name,digest in proposal['evidence_hashes'].items():
    assert sha(ROOT/name)==digest
snapshot=HERE/'execution-attempt1'
snapshot.mkdir(exist_ok=False)
for source in V1.EXPECTED_ROOT.rglob('*'):
    if source.is_file():
        relative=source.relative_to(V1.EXPECTED_ROOT)
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan=load(plan_path)
task=next(t for t in plan['tasks'] if t['id']=='NAV-067')
assert not task['status'] and sum(t['status'] for t in plan['tasks'])==54
record={'timestamp':datetime.now().astimezone().isoformat(),
 'summary':'NAV-067剩余包首次真实执行按失败规则停止；已定位实际Isaac版本接口偏差，隔离恢复准备通过但尚未批准',
 'actions':[
   '准备代码及精确执行清单先提交a70214e，再在tiger11/tmux/GPU6启动一次串行包；C01真实原点检查失败，协调器不再分配后续槽位。',
   '保存ERROR及清理记录；SIGTERM释放已失败的应用，收到退出码0仍因ERROR/缺失FINISHED判失败。核对原目录、进程、实际导入包与当前安装源码。',
   '根因：准备误读/pub/data/caohy/IsaacLab，而实际解释器用IsaacLab_v0472；当前generator忽略新增开关，总按单个terrain origin分配，受控双环境未隔离。',
   '保留全部封存v1和原输出，新增仅用于受控夹具的plane/grid派生v2、拒绝前原点落盘及一次恢复入口；26项基本CPU检查通过，未启动新物理应用。'],
 'files_changed':[
   'CaMRA/navigation/reports/nav067_remaining_validation_v1_execution_review.json',
   'CaMRA/navigation/config/nav067_remaining_validation_recovery_proposal_v2.json',
   'CaMRA/navigation/scripts/nav067_controlled_simulation_v2.py',
   'CaMRA/navigation/scripts/run_nav067_remaining_recovery_v2.py',
   'CaMRA/navigation/src/camra_navigation/envs/controlled_terrain_v2.py',
   'CaMRA/navigation/docs/nav067_remaining_recovery_v2_20261119.md',
   'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md',str(HERE.relative_to(ROOT))+'/' ],
 'commands':[
   'tiger11/tmux nav064-r14-repair-20261115:nav067-remaining-plan %600；CUDA_VISIBLE_DEVICES=6；/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -B CaMRA/navigation/scripts/run_nav067_remaining_validation.py',
   'tiger11/tmux nav064-r14-repair-20261115:nav067-evidence-close %599；同绝对Python运行stop_failed_child.py、recovery_basic.py、close_review_commit.py；命令和原始日志保留在remaining-v1。'],
 'results':[
   '新增Kit启动1，受控通过0，额外导航回合0；C02–C06与L1/L2/L3重复均未执行。显式warmup/受控测量循环未到达，但初始化物理步未完整记录，不声称总物理步为0。',
   '实际118.44418772985227秒GPU进程时间，失败槽仍预留408环境步；无自动重试、无训练、无W查询、无NAV-043/final_*或推送。',
   '新恢复提案PROPOSED pin=e8b70cb9d7c81177d576189772b5aecb6054410df71dde867aa5ad1c4c034cac；请求C01/attempt2一次例外后继续未启动项，累计应用上限9→10，4小时/60000环境步上限不变，精确含失败预留57830。',
   '基本26检查仅为准备；原实际原点数值未保存，源代码CPU复算不能冒充实测。旧科学FAIL、完整方法预算保守接受、旧六回合和v2全部不改；NAV-067=false、54/67，NAV-043未执行。'],
 'evidence':[
   '/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1/HELD.json',
   'CaMRA/navigation/reports/nav067_remaining_validation_v1_execution_review.json',
   str((HERE/'failed-child-stop.json').relative_to(ROOT)),str((HERE/'recovery-verification.json').relative_to(ROOT)),
   str(snapshot.relative_to(ROOT)), 'CaMRA/navigation/config/nav067_remaining_validation_recovery_proposal_v2.json'],
 'remaining_work':[
   '取得一次恢复及累计10应用额度的明确确认，另存APPROVED登记/pin后才能由v2入口恢复；不重启v1，不覆盖旧槽或改变原目标。',
   '受控全通过后才执行六导航重复；全部原NAV-067验收和端到端版本/调用链证据仍需真实重验，不能签NAV-043或G5。'],
 'blockers':['本包已按首次实际失败规则停止；原批准不含自动重试且9应用上限不足以含失败再完成全部项，新恢复例外尚未获批准。']}
last=task['work_records'][-1]
oldblock='\n'.join('        '+line for line in json.dumps(last,ensure_ascii=False,indent=2).splitlines())
newblock='\n'.join('        '+line for line in json.dumps(record,ensure_ascii=False,indent=2).splitlines())
text=plan_path.read_text()
assert text.count(oldblock)==1
patch='*** Begin Patch\n*** Update File: '+str(plan_path)+'\n@@\n'+''.join('-'+line+'\n' for line in oldblock.splitlines())+''.join('+'+line+'\n' for line in (oldblock+',\n'+newblock).splitlines())+'*** End Patch\n'
subprocess.run(['apply_patch'],cwd=ROOT,input=patch,text=True,check=True)
updated=load(plan_path)
before=json.loads(subprocess.check_output(['git','show','a70214e:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
assert {k:v for k,v in before.items() if k!='tasks'}=={k:v for k,v in updated.items() if k!='tasks'}
for a,b in zip(before['tasks'],updated['tasks'],strict=True):
    assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])]==a['work_records']
    if a['id']!='NAV-067':
        assert a==b
validator=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update']
verification=subprocess.run(validator,cwd=ROOT,capture_output=True,text=True)
(HERE/'close-plan-check.log').write_text(verification.stdout+verification.stderr)
assert verification.returncode==0
protected=subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True)
assert not protected.strip(),protected
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True)
assert not staged.strip(),'unrelated staged files present'
paths=['AGENTS.md','doc/plan/20261109_camra_navigation_implementation.json','CaMRA/navigation/docs/implementation_handoff.md',
    'CaMRA/navigation/docs/nav067_remaining_recovery_v2_20261119.md',
    'CaMRA/navigation/reports/nav067_remaining_validation_v1_execution_review.json',
    'CaMRA/navigation/config/nav067_remaining_validation_recovery_proposal_v2.json',*proposal['source_hashes']]
paths.extend(str(p.relative_to(ROOT)) for p in HERE.rglob('*') if p.is_file() and p.name!='close-review.log')
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','fix(nav): prepare installed-API recovery and preserve NAV-067 failure'],cwd=ROOT,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
write_new(HERE/'closure-commit.json',{'at':datetime.now().astimezone().isoformat(),'head':head,
    'status':'FAILURE_PRESERVED_RECOVERY_PREPARED_NOT_APPROVED','NAV_067_status':False,'completed_tasks':54,
    'proposal_sha256':sha(V2.PROPOSAL),'push':False,'new_recovery_applications':0})
print(json.dumps({'commit':head,'status':'REVIEW_AND_LOCAL_COMMIT_COMPLETE','NAV_067_status':False,
    'pending':'Explicit one-time C01 recovery; 10 cumulative applications, old failure retained.'},ensure_ascii=False),flush=True)
