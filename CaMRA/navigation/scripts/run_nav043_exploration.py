"""One isolated, bounded actual-state budget exploration; never a G5 gate."""
import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from nav067_sources import NAV, ROOT, LEVELS, load, sha
from nav067_runtime import bind_local_locomotion, guarded_application, persist
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from run_nav066_decision_validation import networks

PROTOCOL = NAV/'config/nav043_exploration_v1.json'
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'


def context():
    reg = load(PROTOCOL)
    if sha(PROTOCOL) != Path(str(PROTOCOL)+'.sha256').read_text().strip() or reg['status']!='APPROVED_DELEGATED_ROUTE':
        raise ValueError('unapproved or modified exploration protocol')
    if socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX') or sys.executable!=PYTHON:
        raise ValueError('requires registered tiger11/tmux/absolute interpreter')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':
        raise ValueError('requires physical GPU6, logical cuda:0')
    if importlib.util.find_spec('isaaclab').origin != reg['isaaclab_origin']:
        raise ValueError('actual installed IsaacLab differs')
    for path,digest in reg['sealed_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('sealed source identity mismatch: '+path)
    plan = load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
    states = {t['id']:t['status'] for t in plan['tasks']}
    ready = next(t['id'] for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
    if ready!='NAV-043' or plan['document']['version']!='1.4.1':
        raise ValueError('NAV-043 is not eligible in the registered plan')
    pair = load(NAV/'config/nav066_decision_prereg_v2.json')
    protected = {str(NAV/k):v for k,v in pair['implementation_hashes'].items()}
    protected.update(pair['protected_hashes'])
    protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
    for path,digest in protected.items():
        if sha(path)!=digest:
            raise ValueError('protected identity mismatch: '+path)
    manifest = load(reg['pilot_manifest'])
    if manifest['candidates']!=pair['selected_candidates']:
        raise ValueError('selected pair mismatch')
    prereg,_ = verify_prereg(REGISTRATION)
    if prereg['environment']['t_max_s']!=45. or prereg['implementation_versions']['observation']!='obs-v7':
        raise ValueError('frozen task mismatch')
    nav = load(NAV/prereg['binding']['navigation_contract']['path'])
    nets = networks(pair,prereg,nav)
    return reg,manifest,prereg,nets,len(protected)


def gpu_ready(reg):
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    row = next(x.split(', ') for x in gpu.strip().splitlines() if x.split(', ')[0]=='6')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if row[1]!=reg['physical_gpu_uuid'] or int(row[2])>100 or reg['physical_gpu_uuid'] in apps:
        raise ValueError('registered GPU is unavailable; no automatic replacement')
    return {'gpu':row,'compute_apps':apps}


def child(level):
    reg,manifest,prereg,nets,count = context()
    out = OUTPUT/level
    started = load(out/'STARTED.json')
    if started['protocol_sha256']!=sha(PROTOCOL) or (out/'ENTRY.json').exists():
        raise ValueError('slot already entered or incorrect protocol')
    origin = bind_local_locomotion(NAV)
    persist(out/'ENTRY.json',{'at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
        'python':sys.executable,'argv':sys.argv,'cwd':str(Path.cwd()),
        'tmux':os.environ['TMUX'],'tmux_pane':os.environ.get('TMUX_PANE'),
        'local_locomotion':origin,'protected_hashes_checked':count,
        'actual_isaaclab_origin':importlib.util.find_spec('isaaclab').origin})
    def factory():
        from isaaclab.app import AppLauncher
        return AppLauncher(headless=True,device='cuda:0').app
    def work():
        # Count all calls to the installed physical step, including setup/reset.
        from isaaclab.sim import SimulationContext
        original = SimulationContext.step
        counter = {'calls':0,'max':20000,'initialization_native_steps':'only explicit SimulationContext.step calls observable'}
        def counted(sim,*args,**kwargs):
            if counter['calls']>=counter['max']:
                raise ValueError('per-application physical step cap reached')
            counter['calls']+=1
            return original(sim,*args,**kwargs)
        SimulationContext.step = counted
        try:
            from nav043_simulation import simulate
            row = next(m for m in manifest['maps'] if m['level']==level)
            return simulate(row,out,manifest,prereg,nets,'cuda:0')
        finally:
            persist(out/'physical_step_accounting.json',counter)
            SimulationContext.step = original
    return guarded_application(out,factory,work)


def coordinate():
    reg,_,_,_,count = context()
    gpu = gpu_ready(reg)
    OUTPUT.mkdir(exist_ok=False)
    persist(OUTPUT/'OWNER.json',{'at':datetime.now().astimezone().isoformat(),'pid':os.getpid(),
        'protocol_sha256':sha(PROTOCOL),'host':socket.getfqdn(),'python':sys.executable,
        'tmux':os.environ['TMUX'],'tmux_pane':os.environ.get('TMUX_PANE'),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'gpu':gpu,'protected_hashes_checked':count})
    used,starts = 0.,0
    deadline = time.monotonic()+10780.  # reserve 20 seconds for cleanup/accounting
    try:
        for level in LEVELS:
            gpu_ready(reg)
            if starts>=3 or time.monotonic()>=deadline:
                raise ValueError('new allocation exhausted')
            out = OUTPUT/level
            out.mkdir(exist_ok=False)
            starts+=1
            persist(out/'STARTED.json',{'at':datetime.now().astimezone().isoformat(),
                'application_index':starts,'protocol_sha256':sha(PROTOCOL),
                'reserved_physical_steps':20000,'max_navigation_episodes':2,
                'max_nominal_episodes':2,'max_initial_W_queries':10})
            command = [PYTHON,'-B',str(Path(__file__).resolve()),'--child',level]
            before=time.monotonic()
            with (out/'stdout.log').open('x') as stream:
                process = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                persist(out/'PROCESS.json',{'pid':process.pid,'command':command,'cwd':str(ROOT),
                    'CUDA_VISIBLE_DEVICES':'6','automatic_retry':False})
                timeout=False
                while process.poll() is None:
                    remaining=deadline-time.monotonic()
                    if remaining<=0 or (out/'ERROR.json').exists():
                        timeout=remaining<=0
                        os.killpg(process.pid,signal.SIGTERM)
                        try: process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL)
                            process.wait(timeout=5)
                        break
                    try: process.wait(timeout=min(5,remaining))
                    except subprocess.TimeoutExpired: pass
            used+=time.monotonic()-before
            persist(out/'EXIT.json',{'exit_code':process.returncode,'cumulative_gpu_wall_s':used,'timeout':timeout})
            if process.returncode!=0 or timeout or (out/'ERROR.json').exists() or not (out/'FINISHED.json').is_file():
                raise ValueError('application/evidence failure; stop without retry: '+level)
            print('EXPLORATION_SLOT_FINISHED',level,flush=True)
        persist(OUTPUT/'FINISHED.json',{'applications':starts,'gpu_wall_s':used,
            'NAV_043_pass':False,'G5_pass':False,'requires_saved_evidence_review':True})
        return 0
    except BaseException as error:
        persist(OUTPUT/'HELD.json',{'error':str(error),'applications':starts,
            'gpu_wall_s':used,'no_automatic_retry':True,'NAV_043_pass':False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--child',choices=LEVELS)
    args=parser.parse_args()
    if args.check:
        reg,_,_,_,count=context()
        print('PINNED_CONTEXT_PASS',count,gpu_ready(reg),flush=True)
        return 0
    return child(args.child) if args.child else coordinate()


if __name__=='__main__':
    raise SystemExit(main())
