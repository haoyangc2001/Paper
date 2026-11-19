"""Save verified L1 progress; never launch, retry, or mark NAV-067 complete."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAV = ROOT/'CaMRA/navigation'
PLAN = ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1')
assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX_PANE') == '%599'
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
assert b'run_nav067_remaining_recovery_v2.py' in Path('/proc/2588869/cmdline').read_bytes()
assert Path('/proc/2636818/cmdline').exists()
assert not (OUTPUT/'RECOVERY_V2_HELD.json').exists()
assert not (OUTPUT/'RECOVERY_V2_FINISHED.json').exists()
audit = json.loads((HERE/'saved-map-review-L1_empty.json').read_bytes())
assert audit['status'] == 'SAVED_MAP_TRACE_RUNTIME_EVENT_REPLAY_PASS_NOT_TASK_CLOSURE'
assert all(pair['physics_event_replay'] == 'PASS_EVERY_SAVED_SUBSTEP' for pair in audit['comparisons'])
assert len(audit['comparisons']) == 2
for name in ('resume_review.py','review_finished_map.py','close_verified_task.py',Path(__file__).name):
    ast.parse((HERE/name).read_text(),filename=name)

def patch(text):
    subprocess.run(['apply_patch'],input=text,text=True,cwd=ROOT,check=True)

def add(path, content):
    assert not path.exists(), str(path)
    patch('*** Begin Patch\n*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in content.splitlines())+'*** End Patch\n')

finished = [level for level in ('L1_empty','L2_single_detour','L3_multi_path')
            if (OUTPUT/level/'FINISHED.json').exists()]
assert finished == ['L1_empty'], finished
timestamp = datetime.now().astimezone().isoformat()
report = {'at':timestamp,'task':'NAV-067','status':'L1_REPEAT_REVIEW_PASS_OTHER_MAPS_RUNNING',
          'finished_repeat_episodes':2,'allocated_repeat_episodes':6,
          'controlled_cases_passed':6,'saved_map_review':audit,
          'existing_coordinator_pid':2588869,'existing_watcher_pid':2636818,
          'new_applications_from_this_review':0,'plan_status_changes':0,
          'NAV_067_status':False,'NAV_043_executed':False,
          'historical_scientific_FAIL_preserved':True,'remote_push':False}
report_path = NAV/'reports/nav067_l1_repeat_progress_20261119.json'
add(report_path,json.dumps(report,ensure_ascii=False,indent=2)+'\n')
raw = PLAN.read_bytes()
before = json.loads(raw)
task = next(t for t in before['tasks'] if t['id']=='NAV-067')
assert not task['status'] and len(task['work_records']) == 20
record = {'timestamp':timestamp,'summary':'NAV-067续跑观察：L1两方法重复及逐子步重放通过，L2/L3仍由原分配继续',
    'actions':['恢复后完整解析1.4.1计划、核对全部67任务/241记录及当前约束；只跟进既有协调器，不重复启动。',
               '主agent对L1已保存的两方法做实际clean/碰撞体读回、基线配对、命令/时钟/J复算，并逐200Hz状态重放生产事件/预算链；新增仿真、训练和W查询0。',
               '保留首次只读审阅遗漏CUDA_VISIBLE_DEVICES的拒绝日志；按登记GPU6环境重作离线核验，不涉及应用重试。'],
    'files_changed':[str(report_path.relative_to(ROOT)),str(HERE.relative_to(ROOT))+'/',str(PLAN.relative_to(ROOT))],
    'commands':['tiger11/tmux nav064-r14-repair-20261115:nav067-evidence-close %599；cwd='+str(ROOT)+'；'+sys.executable+' -B '+str(HERE/'resume_review.py'),
                'CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '+sys.executable+' -B '+str(HERE/'review_finished_map.py')+' L1_empty',
                sys.executable+' -B '+str(Path(__file__))],
    'results':['六项受控实际证据保持；L1重复两回合已通过保存轨迹、事件/预算逐子步重放及同配置配对核验，不据此声称其余四回合通过。',
               'NAV-067仍false、54/67；原科学FAIL及失败尝试保留，NAV-043未执行；只读审阅器继续等待同一分配终态。'],
    'evidence':[str(report_path.relative_to(ROOT)),str((HERE/'saved-map-review-L1_empty.json').relative_to(ROOT)),str((HERE/'resumed-plan-read.json').relative_to(ROOT))],
    'remaining_work':['等待L2/L3原四重复回合和既有只读审阅器的终态核验；全部逐条验收后再判定NAV-067并提交最终证据。'],
    'blockers':[]}
old = '\n'.join('        '+line for line in json.dumps(task['work_records'][-1],ensure_ascii=False,indent=2).splitlines())
new = '\n'.join('        '+line for line in json.dumps(record,ensure_ascii=False,indent=2).splitlines())
assert PLAN.read_text().count(old) == 1
patch('*** Begin Patch\n*** Update File: '+str(PLAN)+'\n@@\n'+''.join('-'+line+'\n' for line in old.splitlines())+''.join('+'+line+'\n' for line in (old+',\n'+new).splitlines())+'*** End Patch\n')
if raw.count(b'\r\n') and raw.count(b'\r\n') == raw.count(b'\n'):
    changed = PLAN.read_bytes().replace(b'\r\n',b'\n')
    PLAN.write_bytes(changed.replace(b'\n',b'\r\n'))
after = json.loads(PLAN.read_bytes())
for a,b in zip(before['tasks'],after['tasks']):
    if a['id'] == 'NAV-067':
        assert a['work_records'] == b['work_records'][:-1]
        assert {k:v for k,v in a.items() if k!='work_records'} == {k:v for k,v in b.items() if k!='work_records'}
    else:
        assert a == b
assert {k:v for k,v in before.items() if k!='tasks'} == {k:v for k,v in after.items() if k!='tasks'}
validation = subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update'],cwd=ROOT,capture_output=True,text=True)
assert validation.returncode == 0, validation.stdout+validation.stderr
add(HERE/'l1-progress-verification.json',json.dumps({'at':timestamp,'all_prior_records_and_task_fields_preserved':True,
    'plan_status_changes':0,'validator':validation.stdout,'new_simulator_starts':0,'physics_replay_cases':2,
    'read_only_guard_failure_log_retained':str(HERE/'saved-map-review-L1_empty.log')},ensure_ascii=False,indent=2)+'\n')
paths = [str(report_path.relative_to(ROOT)),str(PLAN.relative_to(ROOT)),
    *[str((HERE/name).relative_to(ROOT)) for name in ('resume_review.py','resumed-plan-read.json',
         'review_finished_map.py','saved-map-review-L1_empty.json','checkpoint_finished_l1.py',
         'close_verified_task.py','l1-progress-verification.json')]]
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','test(nav): preserve L1 repeat and substep replay evidence'],cwd=ROOT,check=True)
print(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'status':'L1_PROGRESS_COMMITTED_NAV067_PENDING','completed_repeats':2,'total_repeats':6},ensure_ascii=False),flush=True)
