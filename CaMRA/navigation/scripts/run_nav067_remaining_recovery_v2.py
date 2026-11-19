"""Prepared one-shot C01 recovery. Refuses execution without new pinned approval."""
import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from nav067_sources import NAV, ROOT, load, sha
from nav067_runtime import bind_local_locomotion, guarded_application, persist
import run_nav067_remaining_validation as V1

APPROVAL=NAV/'config/nav067_remaining_validation_recovery_v2.json'
PROPOSAL=NAV/'config/nav067_remaining_validation_recovery_proposal_v2.json'


def output_for(slot):
    return V1.EXPECTED_ROOT/('C01/attempt2' if slot=='C01' else slot)


def context(approval_path=APPROVAL, *, initial=True):
    # Even --check is forbidden before an independent approved recovery exists.
    approval=V1.pinned(approval_path)
    if Path(approval_path).resolve()!=APPROVAL.resolve() or approval['status']!='APPROVED':
        raise ValueError('new explicit NAV-067 recovery approval is required')
    if approval['proposal_sha256']!=sha(PROPOSAL):
        raise ValueError('recovery proposal identity mismatch')
    proposal=load(PROPOSAL)
    if proposal['status']!='PROPOSED' or proposal['automatic_execution_authorized']:
        raise ValueError('historical proposal must be preserved separately from approval')
    execution,manifest,prereg,nets=V1.context()
    if sha(V1.EXECUTION)!=proposal['original_execution_sha256']:
        raise ValueError('original exact fixture/source manifest changed')
    for name,digest in proposal['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('recovery source changed: '+name)
    for name,digest in proposal['evidence_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('recovery evidence changed: '+name)
    for name,digest in proposal['installed_source_hashes'].items():
        if sha(name)!=digest:
            raise ValueError('actual installed Isaac API changed: '+name)
    if importlib.util.find_spec('isaaclab').origin!=proposal['actual_isaaclab_origin']:
        raise ValueError('a different IsaacLab installation is active')
    root=V1.EXPECTED_ROOT
    held=load(root/'HELD.json')
    if held['kit_starts']!=1 or held['reserved_environment_ticks']!=408:
        raise ValueError('original failed allocation differs')
    if held['cumulative_gpu_wall_s']!=proposal['previous_gpu_wall_s']:
        raise ValueError('old GPU allocation changed')
    for slot in V1.SLOTS[1:]:
        if initial and (root/slot).exists():
            raise ValueError('an originally unstarted slot is already allocated: '+slot)
    return proposal,execution,manifest,prereg,nets


def allowed(slot):
    for previous in V1.SLOTS[:V1.SLOTS.index(slot)]:
        out=output_for(previous)
        if not (out/'FINISHED.json').is_file() or not (out/'EXIT.json').is_file():
            raise ValueError('preceding recovery slot has not passed: '+previous)
        if load(out/'EXIT.json')['exit_code']!=0 or (out/'ERROR.json').exists():
            raise ValueError('preceding recovery slot failed: '+previous)


def child(slot):
    # Revalidate authorization/source identities, allowing this coordinator's
    # already-claimed directories. Full initial slot emptiness is checked once.
    proposal,execution,manifest,prereg,nets=context(initial=False)
    allowed(slot)
    out=output_for(slot)
    claim=load(out/'STARTED.json')
    if claim['recovery_approval_sha256']!=sha(APPROVAL) or (out/'ENTRY.json').exists():
        raise ValueError('recovery slot already entered or claimed incorrectly')
    origin=bind_local_locomotion(NAV)
    persist(out/'ENTRY.json',{'at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
        'python':sys.executable,'argv':sys.argv,'cwd':str(Path.cwd()),'tmux':os.environ['TMUX'],
        'tmux_pane':os.environ.get('TMUX_PANE'),'local_locomotion':origin,
        'actual_isaaclab_origin':importlib.util.find_spec('isaaclab').origin})
    def factory():
        from isaaclab.app import AppLauncher
        return AppLauncher(headless=True,device='cuda:0').app
    def work():
        if slot.startswith('C'):
            from nav067_controlled_simulation_v2 import simulate
            return simulate(execution['fixtures'][slot],out,manifest,prereg,'cuda:0')
        row=next(r for r in manifest['maps'] if r['level']==slot)
        return V1.repetition(row,out,manifest,prereg,nets,'cuda:0')
    return guarded_application(out,factory,work)


def coordinate():
    proposal,execution,_,_,_=context()
    root=V1.EXPECTED_ROOT
    persist(root/'RECOVERY_V2_OWNER.json',{'at':datetime.now().astimezone().isoformat(),
        'pid':os.getpid(),'approval_sha256':sha(APPROVAL),'proposal_sha256':sha(PROPOSAL),
        'old_HELD_sha256':sha(root/'HELD.json'),'python':sys.executable,'tmux':os.environ['TMUX']})
    used=proposal['previous_gpu_wall_s']
    reserved=408
    starts=1
    try:
        for slot in V1.SLOTS:
            allowed(slot)
            cap=V1.slot_bounds(execution,slot)
            if starts>=10 or reserved+cap>60000 or used>=14380.:
                raise ValueError('explicit cumulative recovery resource boundary reached')
            out=output_for(slot)
            out.mkdir(exist_ok=False)
            starts+=1
            reserved+=cap
            persist(out/'STARTED.json',{'at':datetime.now().astimezone().isoformat(),'slot':slot,
                'recovery_approval_sha256':sha(APPROVAL),'kit_start_index_including_failure':starts,
                'reserved_environment_ticks_including_warmup':cap,'cumulative_reserved_environment_ticks':reserved,
                'remaining_gpu_wall_s':14400.-used})
            command=[V1.PYTHON,'-B',str(Path(__file__).resolve()),'--child',slot]
            before=time.monotonic()
            with (out/'stdout.log').open('x') as stream:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                persist(out/'PROCESS.json',{'pid':process.pid,'command':command,'cwd':str(ROOT),
                    'CUDA_VISIBLE_DEVICES':'6','no_automatic_retry':True})
                failed_cleanup=False
                timeout=False
                while process.poll() is None:
                    remaining=14380.-used-(time.monotonic()-before)
                    if (out/'ERROR.json').exists() or remaining<=0:
                        failed_cleanup=(out/'ERROR.json').exists()
                        timeout=remaining<=0
                        os.killpg(process.pid,signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL)
                            process.wait(timeout=5)
                        break
                    try:
                        process.wait(timeout=min(5,max(.001,remaining)))
                    except subprocess.TimeoutExpired:
                        pass
            wall=time.monotonic()-before
            used+=wall
            persist(out/'EXIT.json',{'exit_code':process.returncode,'wall_s':wall,'cumulative_gpu_wall_s':used,
                'timeout':timeout,'failed_application_cleanup':failed_cleanup,'command':command})
            if process.returncode!=0 or timeout or failed_cleanup or (out/'ERROR.json').exists() or not (out/'FINISHED.json').is_file():
                raise ValueError('recovery slot failed; no retry: '+slot)
            print('RECOVERY_PASSED_SLOT',slot,flush=True)
        persist(root/'RECOVERY_V2_FINISHED.json',{'completed_slots':list(V1.SLOTS),'kit_starts_including_failure':starts,
            'reserved_environment_ticks_including_failure':reserved,'cumulative_gpu_wall_s':used,
            'NAV_067_acceptance':'requires main-agent actual-evidence review'})
        return 0
    except BaseException as error:
        persist(root/'RECOVERY_V2_HELD.json',{'at':datetime.now().astimezone().isoformat(),'error':str(error),
            'kit_starts_including_failure':starts,'reserved_environment_ticks_including_failure':reserved,
            'cumulative_gpu_wall_s':used,'no_automatic_retry':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--approval',default=str(APPROVAL))
    parser.add_argument('--child',choices=V1.SLOTS)
    args=parser.parse_args()
    if args.check:
        context(Path(args.approval))
        print('RECOVERY_PINNED_CONTEXT_PASS_NO_APPLICATION')
        return 0
    if Path(args.approval).resolve()!=APPROVAL.resolve():
        raise ValueError('noncanonical recovery approval refused')
    return child(args.child) if args.child else coordinate()


if __name__=='__main__':
    raise SystemExit(main())
