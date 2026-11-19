"""One approved, sealed NAV-067 package; durable serial slots, no retries."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from nav067_sources import NAV, ROOT, LEVELS, load, sha, read_manifest, write_new
from nav067_runtime import bind_local_locomotion, guarded_application, persist

APPROVAL=NAV/'config/nav067_remaining_validation_approval_v1.json'
EXECUTION=NAV/'config/nav067_remaining_validation_execution_v1.json'
EXPECTED_ROOT=Path('/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1')
SLOTS=('C01','C02','C03','C04','C05','C06',*LEVELS)
PYTHON='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'


def pinned(path):
    path=Path(path)
    if not path.is_file():
        raise ValueError('required approval/execution file absent: '+str(path))
    pin=Path(str(path)+'.sha256')
    if not pin.is_file() or sha(path)!=pin.read_text().strip():
        raise ValueError('approval/execution pin mismatch: '+str(path))
    return load(path)


def context(approval_path=APPROVAL):
    # Authorization precedes output allocation, source imports or AppLauncher.
    approval=pinned(approval_path)
    if approval['status']!='APPROVED' or approval['task']!='NAV-067':
        raise ValueError('remaining validation not approved')
    if Path(approval_path).resolve()!=APPROVAL.resolve():
        raise ValueError('only the canonical approved registration may execute')
    proposal=approval['approved_proposal']
    if sha(ROOT/proposal['path'])!=proposal['sha256']:
        raise ValueError('approved proposal changed')
    execution=pinned(EXECUTION)
    if execution['approval_sha256']!=sha(APPROVAL) or execution['output_root']!=str(EXPECTED_ROOT):
        raise ValueError('execution identity mismatch')
    if execution['slots']!=list(SLOTS) or execution['limits']!=approval['limits']:
        raise ValueError('execution order/quota mismatch')
    if socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX'):
        raise ValueError('tiger11/tmux is mandatory')
    if os.path.realpath(sys.executable)!=os.path.realpath(PYTHON):
        raise ValueError('unexpected absolute Python')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':
        raise ValueError('only preflighted physical GPU 6 may be allocated')
    for name,digest in execution['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('sealed source changed: '+name)
    for name,digest in execution['evidence_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('sealed evidence changed: '+name)
    manifest,prereg,nets=read_manifest()
    return execution,manifest,prereg,nets


def slot_bounds(execution,slot):
    if slot.startswith('C'):
        fixture=execution['fixtures'][slot]
        return fixture['num_envs']*(200+fixture['measurement_tick_cap'])
    return 2*46*200


def allowed_slot(root,slot):
    index=SLOTS.index(slot)
    for previous in SLOTS[:index]:
        if not (root/previous/'FINISHED.json').is_file() or not (root/previous/'EXIT.json').is_file():
            raise ValueError('previous slot has not passed: '+previous)
        if load(root/previous/'EXIT.json')['exit_code']!=0 or (root/previous/'ERROR.json').exists():
            raise ValueError('previous slot failed: '+previous)


def repetition(row,out,manifest,prereg,nets,device):
    import numpy as np
    from camra_navigation.maps.generator import MapSpec
    from camra_navigation.maps.scene_adapter import build_scene
    from camra_navigation.envs import clean_v3
    from isaaclab.sim import SimulationContext
    from nav067_collider_readback import read_box
    from nav067_simulation_v3 import simulate
    boxes_layout=build_scene(MapSpec.from_dict(row['spec']),0,1)
    boxes=[*boxes_layout.wall_boxes,*boxes_layout.island_boxes]
    original=clean_v3.rigid_body_transform
    proofs=[]
    def passive(body,env_index=0):
        p=original(body,env_index)
        j=len(proofs)
        if env_index!=0 or j>=len(boxes):
            raise ValueError('unexpected passive collider readback invocation')
        b=boxes[j]
        # Origin is independently compared to the saved actual scene origins below.
        origin=np.array([[p[0]-b.center_x,p[1]-b.center_y,p[2]-b.z_center]])
        proof=read_box(SimulationContext.instance().stage,body,b,origin,0,transform_reader=original)
        proofs.append(proof)
        return p
    clean_v3.rigid_body_transform=passive
    try:
        result=simulate(row,out,manifest,prereg,nets,device)
    finally:
        clean_v3.rigid_body_transform=original
        write_new(out/'collider_geometry.json',proofs)
    runtime=load(out/'runtime.json')
    if len(proofs)!=len(boxes) or any(np.max(np.abs(np.asarray(p['actual_env_origin'])-runtime['actual_origins'][0]))>1.e-4 for p in proofs):
        raise ValueError('actual collider origin differs from actual robot/scene origin')
    from nav067_remaining_audit import audit_episode, compare
    source=load(NAV/'config/nav067_remaining_validation_proposal_v1.json')
    old=Path(source['inputs']['manifest']['path']).parent
    registration=next(s for s in source['repetitions']['slots'] if s['level']==row['level'])
    baseline=old/registration['baseline_relative_directory']
    audits=[audit_episode(out/method) for method in ('uncalibrated','full')]
    paired=[compare(baseline/method,out/method) for method in ('uncalibrated','full')]
    write_new(out/'trace_audit.json',audits)
    write_new(out/'repeatability.json',paired)
    return result


def child(slot,device):
    execution,manifest,prereg,nets=context()
    out=EXPECTED_ROOT/slot
    allowed_slot(EXPECTED_ROOT,slot)
    claim=load(out/'STARTED.json')
    if claim['execution_sha256']!=sha(EXECUTION) or (out/'APP_READY.json').exists():
        raise ValueError('slot already started or incorrectly claimed')
    origin=bind_local_locomotion(NAV)
    persist(out/'ENTRY.json',{'pid':os.getpid(),'host':socket.getfqdn(),'python':sys.executable,
        'cwd':str(Path.cwd()),'argv':sys.argv,'local_locomotion':origin,'tmux':os.environ['TMUX'],
        'tmux_pane':os.environ.get('TMUX_PANE'),'execution_sha256':sha(EXECUTION)})
    def factory():
        from isaaclab.app import AppLauncher
        return AppLauncher(headless=True,device=device).app
    def work():
        if slot.startswith('C'):
            from nav067_controlled_simulation import simulate
            return simulate(execution['fixtures'][slot],out,manifest,prereg,device)
        row=next(r for r in manifest['maps'] if r['level']==slot)
        return repetition(row,out,manifest,prereg,nets,device)
    return guarded_application(out,factory,work)


def coordinate():
    execution,_,_,_=context()
    # One coordinator allocation only; no automatic resume or replacement root.
    root=EXPECTED_ROOT
    root.mkdir(exist_ok=False)
    persist(root/'OWNER.json',{'at':datetime.now().astimezone().isoformat(),'host':socket.getfqdn(),
        'pid':os.getpid(),'tmux':os.environ['TMUX'],'tmux_pane':os.environ.get('TMUX_PANE'),
        'python':sys.executable,'execution_sha256':sha(EXECUTION),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    used_wall=0.
    reserved_ticks=0
    starts=0
    try:
        for slot in SLOTS:
            allowed_slot(root,slot)
            cap=slot_bounds(execution,slot)
            if starts>=9 or reserved_ticks+cap>60000 or used_wall>=14380.:
                raise ValueError('approved resource boundary reached')
            out=root/slot
            out.mkdir(exist_ok=False)
            starts+=1
            reserved_ticks+=cap
            persist(out/'STARTED.json',{'at':datetime.now().astimezone().isoformat(),'slot':slot,
                'execution_sha256':sha(EXECUTION),'reserved_environment_ticks_including_warmup':cap,
                'kit_start_index':starts,'cumulative_reserved_environment_ticks':reserved_ticks,
                'remaining_gpu_wall_s':14400.-used_wall})
            cmd=[PYTHON,'-B',str(Path(__file__).resolve()),'--child',slot,'--device','cuda:0']
            begin=time.monotonic()
            with (out/'stdout.log').open('x') as stream:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                persist(out/'PROCESS.json',{'pid':process.pid,'command':cmd,'env':{'CUDA_VISIBLE_DEVICES':'6'},'cwd':str(ROOT)})
                timeout=False
                try:
                    code=process.wait(timeout=max(.001,14380.-used_wall))
                except subprocess.TimeoutExpired:
                    timeout=True
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        code=process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        code=process.wait(timeout=10)
            elapsed=time.monotonic()-begin
            used_wall+=elapsed
            persist(out/'EXIT.json',{'exit_code':code,'wall_s':elapsed,'cumulative_gpu_wall_s':used_wall,
                'timeout':timeout,'command':cmd})
            if code!=0 or timeout or not (out/'FINISHED.json').is_file() or (out/'ERROR.json').exists():
                raise ValueError('slot failed; no retry allowed: '+slot)
            print('PASSED_SLOT',slot,'gpu_wall_s',round(used_wall,2),flush=True)
        persist(root/'FINISHED.json',{'completed_slots':list(SLOTS),'kit_starts':starts,
            'reserved_environment_ticks':reserved_ticks,'cumulative_gpu_wall_s':used_wall,
            'acceptance':'actual evidence requires final independent offline main-agent review'})
        return 0
    except BaseException as error:
        persist(root/'HELD.json',{'at':datetime.now().astimezone().isoformat(),'error':str(error),
            'kit_starts':starts,'reserved_environment_ticks':reserved_ticks,'cumulative_gpu_wall_s':used_wall,
            'no_automatic_retry':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--approval',default=str(APPROVAL),help='Only --check permits a custom path for refusal tests')
    parser.add_argument('--child',choices=SLOTS)
    parser.add_argument('--device',default='cuda:0',choices=['cuda:0'])
    args=parser.parse_args()
    if args.check:
        context(Path(args.approval))
        print('SEALED_CONTEXT_PASS_NO_APPLICATION_OR_OUTPUT_ALLOCATION',flush=True)
        return 0
    if Path(args.approval).resolve()!=APPROVAL.resolve():
        raise ValueError('cannot execute a noncanonical approval')
    return child(args.child,args.device) if args.child else coordinate()


if __name__=='__main__':
    raise SystemExit(main())
