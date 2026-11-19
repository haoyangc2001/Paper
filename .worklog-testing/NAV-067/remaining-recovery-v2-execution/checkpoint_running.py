"""Commit verified controlled evidence and a truthful running checkpoint."""
import ast
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
PLAN=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
sys.path[:0]=[str(NAV/'scripts'),str(NAV/'src'),str(ROOT/'CaMRA/calibration/src')]
from nav067_sources import load, sha, write_new
import run_nav067_remaining_recovery_v2 as V2

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def patch(text):
    subprocess.run(['apply_patch'],input=text,text=True,cwd=ROOT,check=True)

def add(path,content):
    assert not path.exists(),str(path)
    patch('*** Begin Patch\n*** Add File: '+str(path)+'\n'+''.join('+'+s+'\n' for s in content.splitlines())+'*** End Patch\n')

def prepend(path,title,body):
    assert path.read_text().startswith(title+'\n\n')
    patch('*** Begin Patch\n*** Update File: '+str(path)+'\n@@\n '+title+'\n \n'+''.join('+'+s+'\n' for s in (body+'\n\n').splitlines())+'*** End Patch\n')

assert socket.gethostname().split('.')[0]=='tiger11'
assert os.environ.get('TMUX_PANE')=='%599'
assert sys.executable=='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
proposal,execution,manifest,prereg,nets=V2.context(initial=False)
launch=load(HERE/'launch.json')
assert b'run_nav067_remaining_recovery_v2.py' in Path('/proc',str(launch['pid']),'cmdline').read_bytes()
assert not (V2.V1.EXPECTED_ROOT/'RECOVERY_V2_HELD.json').exists()
assert not (V2.V1.EXPECTED_ROOT/'RECOVERY_V2_FINISHED.json').exists()
for name in ('review_saved.py','watch_saved_review.py','checkpoint_running.py'):
    ast.parse((HERE/name).read_text(),filename=name)
spec=importlib.util.spec_from_file_location('saved_review',HERE/'review_saved.py')
review=importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
controlled=review.controlled_review()
assert controlled==load(HERE/'actual-controlled-review.json')['controlled']
review.check_repetition_runtime(load(V2.V1.EXPECTED_ROOT/'L1_empty/runtime.json'))
assert 'ACTUAL_REPETITION_RUNTIME_READBACK_PASS' in (HERE/'review-runtime.log').read_text()
archive=load(NAV/'reports/historical_worktree_archive_20261119_v1.json')
archive_verification=load(ROOT/'.worklog-testing/NAV-067/worktree-audit-20261119/verification.json')
assert archive_verification['counts']=={'reviewed':196,'commit_small':188,'external_archive':8}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip().startswith('3ca89c2')

with (HERE/'saved-review-watcher.log').open('x') as stream:
    watcher=subprocess.Popen([sys.executable,'-B',str(HERE/'watch_saved_review.py')],cwd=ROOT,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
    if (HERE/'saved-review-watcher-start.json').exists():
        break
    assert watcher.poll() is None,'read-only watcher exited before recording start'
    time.sleep(.05)
assert (HERE/'saved-review-watcher-start.json').exists()

artifact_index=[]
for cid in V2.V1.SLOTS[:6]:
    for path in sorted(V2.output_for(cid).rglob('*')):
        if path.is_file():
            artifact_index.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
report_path=NAV/'reports/nav067_remaining_recovery_v2_progress_20261119.json'
report={'schema':'camra-nav067-recovery-running-progress-v2','at':now(),'task':'NAV-067',
    'status':'CONTROLLED_ACTUAL_REPLAY_PASS_NAVIGATION_REPETITIONS_RUNNING',
    'NAV_067_status':False,'completed_plan_tasks':54,'total_plan_tasks':67,
    'controlled':controlled,'controlled_artifact_index':artifact_index,
    'controlled_and_navigation_performance_are_distinct':True,
    'original_failed_C01_preserved':True,'original_failed_HELD_sha256':sha(V2.V1.EXPECTED_ROOT/'HELD.json'),
    'live_coordinator':launch,'read_only_review_watcher_pid':watcher.pid,
    'read_only_review_watcher':load(HERE/'saved-review-watcher-start.json'),
    'after_all_slots':'Run saved-evidence review once; final NAV-067 closure still requires main-agent acceptance review.',
    'limits':proposal['limits'],'exact_reserved_environment_ticks_including_failure':57830,
    'reserved_ticks_are_not_actual_consumption':True,'approval_sha256':sha(V2.APPROVAL),
    'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (HERE/'review_saved.py',HERE/'watch_saved_review.py',Path(__file__))},
    'production_context':'PASS original sealed sources/evidence/295 protected identities and installed IsaacLab hashes',
    'historical_git_archive':{'commit':'3ca89c2cdf0ee996b2cfc246ab94d7f416d9b325','verification':archive_verification},
    'training_updates':0,'W_queries':0,'NAV_043_executed':False,'final_data_access':False,
    'scientific_FAIL_preserved':True,'remote_push':False}
add(report_path,json.dumps(report,ensure_ascii=False,indent=2)+'\n')

doc_path=NAV/'docs/nav067_recovery_v2_progress_20261119.md'
body='''当前NAV-067已修复受控夹具在实际IsaacLab版本中的双环境原点问题。原失败C01仍完整保留，用户确认后只在原目录C01/attempt2恢复了一次；框架、导航地图和固定v2未修改。

六项受控病例已实际通过，并用保存的200Hz读回重新经过生产事件/预算链复算：中间碰撞后退出、危险历史、转向扫掠、同刻到达和碰撞、批内异步结束、原生机器人—障碍接触。能量离线复算最大相对差小于4e-16。C01–C05为写入夹具后的实际读回，C06为初次放置后的原生接触；不能把这六项记为自然导航成功。

原三地图×两方法的六个重复导航回合正在按原顺序串行执行；尚无本轮重复结果结论。每种配置完成后将有原一次和新一次，只报告时间、机械功、终点和轨迹差异，不承诺两次物理逐位相同或普遍稳定。NAV-067仍false、54/67，NAV-043未执行；旧六回合超预算和原科学FAIL保持。

应用在tiger11、GPU6、tmux `nav064-r14-repair-20261115`中运行，使用绝对Python3.11.15。协调器PID及读回核验见[机器进度报告](../reports/nav067_remaining_recovery_v2_progress_20261119.json)；不得重启协调器。新增只读审阅器等待已有分配的FINISHED/HELD，全部结束后只运行一次离线复核，不启动应用、不重试、不改任务状态；最终收尾仍需逐条验收。

恢复累计上限10次应用、4小时单GPU墙钟和60000环境子步，包含原失败；精确预留57830不等于实际消耗。原C01失败标记HELD.json是历史，新的终态为RECOVERY_V2_FINISHED.json或RECOVERY_V2_HELD.json。首次新实际失败即停，不修改夹具补跑，不训练、不作W查询、不访问final_*。

本地未跟踪材料也已补齐：188份小文件按原字节提交于3ca89c2，7份大型明细和1份旧计划备份逐字节核验归档、原件保留，Git保存完整索引。此前部分报告确有归档遗漏；不是所有未跟踪文件都不需要提交。见[归档说明](historical_untracked_archive_20261119.md)。新运行尚未生成的证据将在实际审阅后再提交，未推送。
'''
add(doc_path,body)

plan_text=PLAN.read_text(); plan=json.loads(plan_text)
task=next(t for t in plan['tasks'] if t['id']=='NAV-067')
assert not task['status'] and sum(t['status'] for t in plan['tasks'])==54
record={'timestamp':now(),'summary':'NAV-067六项实际受控与离线复算通过；六导航重复运行中，历史未跟踪证据已补交，任务不提前结案',
    'actions':['按已批准一次恢复执行原C01/attempt2，C01–C06实际检查全过，生产事件链重放及能量复算再次核验。',
        '保持原两模型/45秒/obs-v7/200-50-10Hz/J/固定v2，六导航重复由同一协调器串行继续；只读审阅器等待原分配终态，不增加物理额度或自动重试。',
        '逐项核对196未跟踪文件：188小文件原样补交，7大型明细和1备份核验外部归档、原件保留、Git登记精确索引；未执行旧脚本或更改旧失败。'],
    'files_changed':[str(report_path.relative_to(ROOT)),str(doc_path.relative_to(ROOT)),str(HERE.relative_to(ROOT))+'/',
        'doc/plan/20261109_camra_navigation_implementation.json','AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md'],
    'commands':['tiger11/tmux nav064-r14-repair-20261115:nav067-evidence-close %599；cwd='+str(ROOT)+'；'+sys.executable+' -B '+str(Path(__file__)),
        sys.executable+' -B '+str(HERE/'review_saved.py'),sys.executable+' -B '+str(HERE/'watch_saved_review.py'),
        'git commit 3ca89c2 chore(nav): archive missing historical evidence and index large reports'],
    'results':['六项实际病例通过；受控检查不作为自然导航成功率。机械功离线复算最大相对差<4e-16，事件/终止后active屏蔽一致。',
        '当前六导航重复尚未完成；NAV-067=false、54/67、NAV-043未执行。原科学FAIL和所有原始失败保持，不凭接线通过宣称物理性能通过。',
        '196原文件hash保持，122份JSON/41份Python语法检查；历史空JSON失败保留而非伪造通过。188份补交及8份归档索引已提交3ca89c2。'],
    'evidence':[str(report_path.relative_to(ROOT)),str((HERE/'actual-controlled-review.json').relative_to(ROOT)),
        str((HERE/'launch.json').relative_to(ROOT)),str((HERE/'saved-review-watcher-start.json').relative_to(ROOT)),
        'CaMRA/navigation/reports/historical_worktree_archive_20261119_v1.json'],
    'remaining_work':['等待同一已启动分配完成6导航重复；逐条审阅全部12原/新回合、实际clean/碰撞体/重复偏差，补齐amendment及上下游一致性后再判NAV-067。',
        '实际终态及完整离线审阅报告生成后再提交；本次不执行NAV-043或额外物理/训练/W查询/final_*。'],
    'blockers':[]}
old='\n'.join('        '+s for s in json.dumps(task['work_records'][-1],ensure_ascii=False,indent=2).splitlines())
new='\n'.join('        '+s for s in json.dumps(record,ensure_ascii=False,indent=2).splitlines())
assert plan_text.count(old)==1
patch('*** Begin Patch\n*** Update File: '+str(PLAN)+'\n@@\n'+''.join('-'+s+'\n' for s in old.splitlines())+''.join('+'+s+'\n' for s in (old+',\n'+new).splitlines())+'*** End Patch\n')
after=load(PLAN)
for a,b in zip(plan['tasks'],after['tasks']):
    if a['id']=='NAV-067':
        assert b['work_records'][:-1]==a['work_records'] and b['status']==a['status']
        assert {k:v for k,v in a.items() if k!='work_records'}=={k:v for k,v in b.items() if k!='work_records'}
    else:
        assert a==b
prepend(ROOT/'AGENTS.md','# Agent 接手说明','## 最新进度（2026-11-19，六项实际受控通过，导航重复运行中）\n\n恢复C01/attempt2后C01–C06全部实际通过，主agent保存读回重放与能量复算通过；原失败保留。这些是受控接线证据，不是六次自然导航成功。原六导航重复由既有协调器继续运行，NAV-067仍false、54/67，NAV-043未执行。新增只读审阅器等待本分配终态，运行一次保存证据复核，不重启应用、不改任务状态；先检查进程及RECOVERY_V2_FINISHED/HELD，勿误读原HELD。当前入口见`CaMRA/navigation/docs/nav067_recovery_v2_progress_20261119.md`。历史遗漏188份小证据已补交3ca89c2；7大型明细+1备份核验归档并保留原件，Git有索引。旧科学FAIL、固定v2及其它边界保持。')
prepend(NAV/'docs/implementation_handoff.md','# 上层导航实施交接','## 2026-11-19 六项实际受控及保存读回重放通过，六导航重复运行中\n\n已按一次恢复批准完成C01/attempt2、C02–C06，实际事件、active屏蔽和200Hz机械功重放复核通过；原失败不覆盖。C01–C05为人为写入后的实际读回，C06为原生接触，均非自然导航性能分母。六导航重复仍由原协调器PID2588869串行运行；只读审阅器等待RECOVERY_V2_FINISHED/HELD，全部结束后一次离线复核、不自动重试或标true。NAV-067=false、54/67、NAV-043未执行，完整验收和amendment待实际重复结果。见[进度与恢复说明](nav067_recovery_v2_progress_20261119.md)和[机器证据](../reports/nav067_remaining_recovery_v2_progress_20261119.json)。\n\n用户同时询问未提交文件：已原样补交188小文件于3ca89c2；7大型逐步明细和1计划备份逐字节复制核验、原件保留，Git保存大小/hash/路径索引。历史空JSON失败保留，未执行旧脚本。当前运行尚未生成的终态证据后续审阅再提交，未推送。下方历史状态和科学FAIL保留。')
V2.context(initial=False)
validator=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update']
validated=subprocess.run(validator,cwd=ROOT,capture_output=True,text=True)
write_new(HERE/'running-checkpoint-verification.json',{'at':now(),'status':'VERIFIED_CONTROLLED_RUNNING_CHECKPOINT_NOT_TASK_CLOSURE',
    'validator_command':validator,'validator_exit':validated.returncode,'stdout':validated.stdout,'stderr':validated.stderr,
    'controlled_replay':6,'production_context':'PASS','runtime_readback':'PASS actual L1 repeat',
    'all_previous_tasks_and_records_preserved':True,'plan_status_unchanged':True,'new_simulator_applications':0,
    'review_watcher_pid':watcher.pid})
assert validated.returncode==0
assert not subprocess.check_output(['git','diff','--name-only','--','manuscripts','*.tex','*.bib','RC-PPO','HiRA'],cwd=ROOT,text=True).strip()
paths=[str(report_path.relative_to(ROOT)),str(doc_path.relative_to(ROOT)),str(PLAN.relative_to(ROOT)),
    'AGENTS.md','CaMRA/navigation/docs/implementation_handoff.md',
    *[str((HERE/name).relative_to(ROOT)) for name in ('actual-controlled-review.json','launch.json','review_saved.py',
        'watch_saved_review.py','checkpoint_running.py','saved-review-watcher-start.json','running-checkpoint-verification.json')]]
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','test(nav): record six actual boundary checks and running repetitions'],cwd=ROOT,check=True)
print(json.dumps({'at':now(),'committed':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'NAV_067_status':False,'controlled_checks':6,'repetitions':'RUNNING','watcher_pid':watcher.pid},ensure_ascii=False),flush=True)
