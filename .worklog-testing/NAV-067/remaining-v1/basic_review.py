"""Main-agent CPU checks; synthetic inputs here are never physical evidence."""
import ast
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
from nav067_sources import E, sha, load, write_new, read_manifest
from run_nav067_remaining_validation import context, allowed_slot, EXPECTED_ROOT
from camra_navigation.envs.controlled_cases_v1 import CASE_IDS, build_fixture, fixture_spec, ScheduledMonitor, review_actual_case
from camra_navigation.envs.physical_v2 import PhysicalState, SubstepNavigation
from camra_navigation.evaluation.recorder import NavEnergyRecorder, offline_recompute

torch.set_num_threads(1)
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve()==Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python').resolve()
assert not EXPECTED_ROOT.exists()
checks=[]
def save_or_check(path,value):
    if path.exists():
        assert load(path)==value,'existing basic evidence changed'
    else:
        write_new(path,value)
def check(name,condition=True):
    assert condition,name
    checks.append(name)

plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan=load(plan_path)
old=json.loads(subprocess.check_output(['git','show','27d10fe:'+str(plan_path.relative_to(ROOT))],cwd=ROOT))
check('complete plan parsed; all top fields preserved',{k:v for k,v in plan.items() if k!='tasks'}=={k:v for k,v in old.items() if k!='tasks'})
for before,after in zip(old['tasks'],plan['tasks'],strict=True):
    check('definition/history preserved '+before['id'],
        {k:v for k,v in before.items() if k!='work_records'}=={k:v for k,v in after.items() if k!='work_records'}
        and after['work_records'][:len(before['work_records'])]==before['work_records'])
    if before['id']!='NAV-067':
        assert before==after
states={t['id']:t['status'] for t in plan['tasks']}
ready=[t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on'])]
check('NAV-067 sole eligible; 54/67 true',len(ready)==1 and ready[0]['id']=='NAV-067' and sum(states.values())==54)
save_or_check(HERE/'plan-readback.json',{'plan_sha256':sha(plan_path),'all_tasks_and_records_read':len(plan['tasks']),
    'top_keys':list(plan),'current_task_definition':{k:v for k,v in ready[0].items() if k!='work_records'},
    'latest_work_record':ready[0]['work_records'][-1],
    'global_constraints':plan['global_constraints'],'uncertainty_policy':plan['uncertainty_policy'],
    'execution_alignment_policy':plan['execution_alignment_policy'],'upstream_correction_policy':plan['upstream_correction_policy']})
for path in [NAV/'scripts/nav067_collider_readback.py',NAV/'scripts/nav067_controlled_simulation.py',
             NAV/'scripts/run_nav067_remaining_validation.py',NAV/'src/camra_navigation/envs/controlled_cases_v1.py']:
    ast.parse(path.read_text())
    check('parse '+path.name)
with tempfile.TemporaryDirectory(prefix='nav067-basic-') as directory:
    missing=Path(directory)/'missing-approval.json'
    try:
        context(missing)
    except ValueError as error:
        check('missing approval rejected before application', 'absent' in str(error) and 'isaaclab.app' not in sys.modules)
    else:
        raise AssertionError('missing approval accepted')
    command=[sys.executable,'-B',str(NAV/'scripts/run_nav067_remaining_validation.py'),'--check','--approval',str(missing)]
    negative=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    check('production CLI refuses missing approval without output allocation',negative.returncode!=0 and 'file absent' in negative.stderr and not EXPECTED_ROOT.exists())
    write_new(HERE/'missing-approval-2.json',{'command':command,'exit_code':negative.returncode,'stdout':negative.stdout,'stderr':negative.stderr})
    root=Path(directory)/'slots'
    root.mkdir()
    try:
        allowed_slot(root,'C02')
    except ValueError:
        check('later slot refuses unmet predecessor')
    else:
        raise AssertionError('skipped predecessor')
manifest,prereg,nets=read_manifest()
check('protected upstream and exact selected pair revalidated',set(nets)=={'uncalibrated','full'})
ctx=E.build_rollout_ctx('full')
geom=ctx['geom']
fixtures={case:build_fixture(case,geom) for case in CASE_IDS}
for case,fixture in fixtures.items():
    check('fixture roundtrip '+case,fixture_spec(fixture).to_dict()==fixture['spec'])
    check('fixture bound '+case,1<=fixture['num_envs']<=2 and fixture['measurement_tick_cap']<=200 and fixture['warmup_policy_calls']==50)
    if case=='C06':
        continue
    poses=deepcopy(fixture['initial'])
    def states_from_poses():
        return [PhysicalState(x,y,yaw,0.,0.,0.,.35,(float(np.cos(yaw/2)),0.,0.,float(np.sin(yaw/2)))) for x,y,yaw in poses]
    def place(target):
        poses[:]=deepcopy(target)
    monitor=SubstepNavigation(fixture_spec(fixture),geom,ctx['cm'],states_from_poses(),[1000.]*len(poses),prereg)
    scheduled=ScheduledMonitor(monitor,fixture,place)
    recorder=NavEnergyRecorder(fixture['measurement_tick_cap'],len(poses),device='cpu')
    recorder.begin_window()
    monitor.set_commands(fixture['commands'])
    scheduled.prepare_tick(1)
    zeros=torch.zeros((len(poses),12))
    for tick in range(1,fixture['measurement_tick_cap']+1):
        before=list(monitor.active)
        recorder.record(tau_applied_lab=zeros,qdot_post_lab=zeros,active_mask=before)
        events=scheduled.advance(states_from_poses(),physics_tick=tick)
        for eid in events['ended']:
            recorder.mark_event([eid],'early_termination:'+monitor.labels[eid])
        if not any(monitor.active):
            break
    energy=recorder.finish()
    commands=[{'max_error':0.,'substeps':monitor.tick}]
    report=review_actual_case(fixture,geom,scheduled,energy,commands)
    check('CPU fixture/event/mask logic '+case,report['actual_evidence_assertions_passed'])
    for eid in range(len(poses)):
        check('CPU independent work sum '+case+str(eid),offline_recompute(energy,eid)['steps_match'])
    wrong=deepcopy(energy)
    wrong['valid_mask'][0,0]=not wrong['valid_mask'][0,0]
    try:
        review_actual_case(fixture,geom,scheduled,wrong,commands)
    except ValueError as error:
        check('mask corruption rejected '+case,'mask' in str(error))
    else:
        raise AssertionError('corrupt mask accepted')
check('quota upper bound',sum(f['num_envs']*(200+f['measurement_tick_cap']) for f in fixtures.values())+6*46*200<=60000)
gpu=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
check('GPU6 resource preflight',int(gpu.rsplit(',',1)[1].strip())>=12000)
proposal=load(NAV/'config/nav067_remaining_validation_proposal_v1.json')
for path,digest in proposal['sealed_sources'].items():
    check('baseline immutable '+path,sha(ROOT/path)==digest)
validation=[sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(plan_path),'--mode','update']
subprocess.run(validation,check=True,cwd=ROOT)
source_paths=['CaMRA/navigation/scripts/nav067_collider_readback.py','CaMRA/navigation/scripts/nav067_controlled_simulation.py',
    'CaMRA/navigation/scripts/run_nav067_remaining_validation.py','CaMRA/navigation/src/camra_navigation/envs/controlled_cases_v1.py']
write_new(HERE/'fixtures.json',fixtures)
write_new(HERE/'verification.json',{'status':'MAIN_AGENT_BASIC_CPU_PASS_NOT_PHYSICAL_ACCEPTANCE',
    'at':datetime.now().astimezone().isoformat(),'host':socket.getfqdn(),'python':sys.executable,
    'tmux':os.environ['TMUX'],'tmux_pane':os.environ.get('TMUX_PANE'),'GPU6':gpu,
    'checks':checks,'count':len(checks),'source_hashes':{p:sha(ROOT/p) for p in source_paths},
    'new_applications':0,'new_physical_ticks':0,'new_navigation_episodes':0,'training_updates':0,'W_queries':0})
print(json.dumps({'status':'BASIC_PASS','count':len(checks),'fixtures':{k:v['num_envs'] for k,v in fixtures.items()}},ensure_ascii=False),flush=True)
