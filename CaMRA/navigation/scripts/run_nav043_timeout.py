"""Once-only, three-case execution deadline diagnostic; never train or pass G5."""
import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from nav067_sources import NAV, ROOT, LEVELS, load, sha
from nav067_runtime import bind_local_locomotion, guarded_application, persist
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from run_nav066_decision_validation import networks
from run_nav043_exploration import gpu_ready
from nav043_process_cleanup_v2 import stop_group

PROTOCOL = NAV/'config/nav043_timeout_extension_v1.json'
OUTPUT = Path('/pub/data/caohy/work/camra_navigation/nav043_timeout_extension_v1')
EVIDENCE = ROOT/'.worklog-testing/NAV-043/timeout-extension-v1'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'


def context():
    if not PROTOCOL.is_file():
        raise ValueError('exact timeout diagnostic approval is absent')
    reg = load(PROTOCOL)
    if (sha(PROTOCOL) != Path(str(PROTOCOL)+'.sha256').read_text().strip()
            or reg['status'] != 'APPROVED_CURRENT_USER_REQUEST'
            or reg['output_root'] != str(OUTPUT)
            or reg['execution_horizon_s'] != 90.
            or reg['actor_observation_horizon_s'] != 45.
            or reg['order'] != list(LEVELS)
            or reg['methods'] != ['full']
            or reg['limits'] != {'applications':3, 'physical_navigation_episodes':3,
                'W_queries':15, 'nominal_episodes':0, 'gpu_wall_s':10800,
                'explicit_physics_calls':60000, 'per_application_physics_calls':20000,
                'automatic_retries':0, 'training_updates':0, 'calibration_fits':0}):
        raise ValueError('modified approval or diagnostic scope')
    if (socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX')
            or sys.executable != PYTHON or os.environ.get('CUDA_VISIBLE_DEVICES') != '6'):
        raise ValueError('requires registered tiger11/tmux/Python/GPU6')
    if importlib.util.find_spec('isaaclab').origin != reg['isaaclab_origin']:
        raise ValueError('installed IsaacLab differs')
    for name, digest in reg['sealed_hashes'].items():
        if sha(name) != digest:
            raise ValueError('diagnostic source mismatch: '+name)
    plan = load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
    states = {t['id']:t['status'] for t in plan['tasks']}
    ready = next(t['id'] for t in plan['tasks'] if not t['status']
                 and all(states[d] for d in t['depends_on']))
    if ready != 'NAV-043' or plan['document']['version'] != '1.5.0':
        raise ValueError('NAV-043 is not the current task')
    pair_path = NAV/'config/nav066_decision_prereg_v2.json'
    pair = load(pair_path)
    if sha(pair_path) != Path(str(pair_path)+'.sha256').read_text().strip():
        raise ValueError('selected pair registration mismatch')
    protected = {str(NAV/k):v for k,v in pair['implementation_hashes'].items()}
    protected.update(pair['protected_hashes'])
    protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
    for name, digest in protected.items():
        if sha(name) != digest:
            raise ValueError('protected identity mismatch: '+name)
    manifest = load(reg['pilot_manifest'])
    if manifest['candidates'] != pair['selected_candidates']:
        raise ValueError('selected pair mismatch')
    prereg, _ = verify_prereg(REGISTRATION)
    if (prereg['environment']['t_max_s'] != 45.
            or prereg['implementation_versions']['observation'] != 'obs-v7'):
        raise ValueError('frozen actor task differs')
    nav = load(NAV/prereg['binding']['navigation_contract']['path'])
    nets = networks(pair, prereg, nav)
    return reg, manifest, prereg, nets, len(protected)


def child(level):
    reg, manifest, prereg, nets, count = context()
    out = OUTPUT/level
    if (load(out/'STARTED.json')['protocol_sha256'] != sha(PROTOCOL)
            or (out/'ENTRY.json').exists()):
        raise ValueError('invalid or consumed child slot')
    origin = bind_local_locomotion(NAV)
    persist(out/'ENTRY.json', {'at':datetime.now().astimezone().isoformat(),
        'pid':os.getpid(), 'python':sys.executable, 'argv':sys.argv, 'cwd':str(Path.cwd()),
        'tmux':os.environ['TMUX'], 'tmux_pane':os.environ.get('TMUX_PANE'),
        'local_locomotion':origin, 'protected_hashes_checked':count})
    def factory():
        from isaaclab.app import AppLauncher
        return AppLauncher(headless=True, device='cuda:0').app
    def work():
        from isaaclab.sim import SimulationContext
        original = SimulationContext.step
        counter = {'calls':0, 'max':20000,
            'native_initialization_steps':'not observable by the explicit Python step guard'}
        def counted(sim, *args, **kwargs):
            if counter['calls'] >= counter['max']:
                raise ValueError('application physical step cap reached')
            counter['calls'] += 1
            return original(sim, *args, **kwargs)
        SimulationContext.step = counted
        try:
            from nav043_timeout_simulation import simulate
            row = next(r for r in manifest['maps'] if r['level'] == level)
            return simulate(row, out, manifest, prereg, nets, 'cuda:0')
        finally:
            persist(out/'physical_step_accounting.json', counter)
            SimulationContext.step = original
    return guarded_application(out, factory, work)


def coordinate():
    reg, _, _, _, count = context()
    gpu = gpu_ready(reg)
    if OUTPUT.exists() or (EVIDENCE/'launch.lock').exists():
        raise ValueError('attempt exists; do not retry or change output root')
    persist(EVIDENCE/'launch.lock', {'pid':os.getpid(), 'protocol_sha256':sha(PROTOCOL)})
    OUTPUT.mkdir(exist_ok=False)
    persist(OUTPUT/'OWNER.json', {'at':datetime.now().astimezone().isoformat(),
        'pid':os.getpid(), 'host':socket.getfqdn(), 'python':sys.executable,
        'protocol_sha256':sha(PROTOCOL), 'protected_hashes_checked':count,
        'tmux':os.environ['TMUX'], 'tmux_pane':os.environ.get('TMUX_PANE'),
        'gpu':gpu, 'git_head':subprocess.check_output(
            ['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()})
    starts = 0
    before_all = time.monotonic()
    deadline = before_all+10780.
    try:
        for level in LEVELS:
            gpu_ready(reg)
            out = OUTPUT/level
            out.mkdir(exist_ok=False)
            starts += 1
            persist(out/'STARTED.json', {'at':datetime.now().astimezone().isoformat(),
                'application_index':starts, 'protocol_sha256':sha(PROTOCOL),
                'max_physical_episodes':1, 'max_initial_W_queries':5,
                'reserved_explicit_steps':20000})
            command = [PYTHON,'-u','-B',str(Path(__file__).resolve()),'--child',level]
            with (out/'stdout.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                persist(out/'PROCESS.json', {'pid':process.pid, 'command':command,
                    'cwd':str(ROOT), 'CUDA_VISIBLE_DEVICES':'6', 'automatic_retry':False})
                cleanup = None
                while process.poll() is None:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0 or (out/'ERROR.json').exists():
                        cleanup = stop_group(process)
                        break
                    try:
                        process.wait(timeout=min(5., remaining))
                    except subprocess.TimeoutExpired:
                        pass
            elapsed = time.monotonic()-before_all
            persist(out/'EXIT.json', {'exit_code':process.returncode,
                'cumulative_gpu_wall_s':elapsed, 'cleanup':cleanup})
            if (process.returncode != 0 or cleanup is not None
                    or (out/'ERROR.json').exists() or not (out/'FINISHED.json').is_file()):
                raise ValueError('infrastructure/evidence failure: '+level)
            print('TIMEOUT_DIAGNOSTIC_SLOT_FINISHED', level, flush=True)
        persist(OUTPUT/'FINISHED.json', {'applications':starts,
            'gpu_wall_s':time.monotonic()-before_all,
            'requires_saved_evidence_review':True, 'NAV_043_pass':False, 'G5_pass':False})
        return 0
    except BaseException as error:
        persist(OUTPUT/'HELD.json', {'error':repr(error), 'applications':starts,
            'gpu_wall_s':time.monotonic()-before_all, 'automatic_retry':False,
            'NAV_043_pass':False, 'G5_pass':False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--child', choices=LEVELS)
    args = parser.parse_args()
    if args.check:
        reg, _, _, _, count = context()
        if OUTPUT.exists() or (EVIDENCE/'launch.lock').exists():
            raise ValueError('attempt already exists')
        print('TIMEOUT_PREFLIGHT_PASS', count, gpu_ready(reg), flush=True)
        return 0
    return child(args.child) if args.child else coordinate()


if __name__ == '__main__':
    raise SystemExit(main())
