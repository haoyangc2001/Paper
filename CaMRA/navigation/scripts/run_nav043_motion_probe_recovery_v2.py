"""Approval-gated once-only launcher for the original 12-case motion proposal.

No CLI changes to output, GPU, commands, attempts, time or step limits exist.
--check requires approval but creates no markers and never starts an app.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import nav043_motion_probe_recovery_guard as guard
from nav043_process_cleanup_v2 import stop_group


def prepare(*, child=False):
    p, a, r = guard.approved()
    context = guard.registered_context(p)
    identities = guard.protected_sources()
    if not child:
        guard.refuse_conflicts(p, Path(__file__).resolve())
    return p, context, identities


def child():
    p, context, identities = prepare(child=True)
    out = Path(p['output_root'])
    owner = guard.load(out/'OWNER.json')
    lock = guard.load(guard.EVIDENCE/'launch.lock')
    if (owner['pid'] != os.getppid() or owner != lock
            or owner['proposal_sha256'] != guard.sha(guard.PROPOSAL)
            or owner['approval_sha256'] != guard.sha(guard.APPROVAL)
            or owner['readiness_sha256'] != guard.sha(guard.READINESS)):
        raise ValueError('child is not owned by the unique registered launcher')
    guard.persist(out/'ENTRY.json', {'pid':os.getpid(), 'context':context,
        'identities':identities, 'at':datetime.now().astimezone().isoformat()})
    from nav067_runtime import bind_local_locomotion
    binding = bind_local_locomotion(guard.NAV)
    guard.persist(out/'LOCAL_SOURCE.json', binding)
    app, original, simulation_class, budget = None, None, None, None
    def interrupted(signum, frame):
        raise TimeoutError('allocation terminated by launcher; save partial capture')
    primary_error = None
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    with (out/'step_calls.jsonl').open('x') as journal:
        try:
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=True, device='cuda:0').app
            guard.persist(out/'APP_READY.json', {'at':datetime.now().astimezone().isoformat()})
            from isaaclab.sim import SimulationContext
            simulation_class, original = SimulationContext, SimulationContext.step
            budget = guard.ExplicitStepBudget(journal,
                total_cap=p['limits']['all_explicit_simulation_step_calls'],
                reserve_cap=p['limits']['setup_reset_reserve_step_calls'])
            def counted(sim, *args, **kwargs):
                return budget.call(original, sim, *args, **kwargs)
            SimulationContext.step = counted
            from nav043_motion_probe_recovery_simulation import simulate
            return simulate(p, out, budget)
        except BaseException as error:
            primary_error = repr(error)
            guard.persist(out/'ERROR.json', {'error':repr(error), 'traceback':traceback.format_exc(),
                'step_accounting':budget.snapshot() if budget else None})
            raise
        finally:
            if budget:
                guard.persist(out/'pre_cleanup_step_accounting.json', budget.snapshot())
            try:
                # Keep the installed step hook active through application cleanup.
                if app is not None:
                    app.close()
            except BaseException as cleanup_error:
                guard.persist(out/'child_cleanup_error.json',
                    {'primary_error':primary_error, 'cleanup_error':repr(cleanup_error)})
                if primary_error is None:
                    raise
            finally:
                if original is not None:
                    simulation_class.step = original
                if budget:
                    guard.persist(out/'child_step_accounting.json', budget.snapshot())
                signal.signal(signal.SIGTERM, previous_term)


def coordinate():
    started = time.monotonic()
    p, context, identities = prepare()
    gpu = guard.gpu_ready(p)
    output = Path(p['output_root'])
    guard.EVIDENCE.mkdir(exist_ok=True)
    owner = {'pid':os.getpid(), 'at':datetime.now().astimezone().isoformat(),
        'proposal_sha256':guard.sha(guard.PROPOSAL),
        'approval_sha256':guard.sha(guard.APPROVAL),
        'readiness_sha256':guard.sha(guard.READINESS), 'output_root':str(output),
        'context':context, 'identities':identities, 'gpu':gpu,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=guard.ROOT,text=True).strip()}
    guard.persist(guard.EVIDENCE/'launch.lock', owner)
    process, error, timed_out, code, owns_output = None, None, False, -1, False
    cleanup = None
    try:
        output.mkdir(exist_ok=False)
        owns_output = True
        guard.persist(output/'OWNER.json', owner)
        # Ten seconds remain inside the total cap for TERM/KILL/accounting.
        deadline = started+p['limits']['gpu_wall_s']-10.
        if time.monotonic() >= deadline:
            raise TimeoutError('allocation exhausted during preflight')
        command = [p['environment']['python'], '-B', str(Path(__file__).resolve()), '--child']
        with (output/'stdout.log').open('x') as stream:
            environment = os.environ.copy()
            environment['PYTHONPATH'] = os.pathsep.join((str(guard.NAV/'src'),
                str(guard.NAV.parent/'calibration/src'), environment.get('PYTHONPATH','')))
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            process = subprocess.Popen(command, cwd=guard.ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            guard.persist(output/'PROCESS.json', {'pid':process.pid, 'command':command,
                                                  'automatic_retry':False})
            while process.poll() is None:
                remaining = deadline-time.monotonic()
                if remaining <= 0 or (output/'ERROR.json').exists():
                    timed_out = remaining <= 0
                    cleanup = stop_group(process)
                    break
                try:
                    process.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    pass
        if cleanup is None:
            cleanup = stop_group(process)
        code = cleanup['returncode']
        if (code != 0 or timed_out or (output/'ERROR.json').exists()
                or not cleanup['cleanup_confirmed']):
            raise ValueError('first application failure/timeout; no retry')
        if not (output/'CAPTURE_FINISHED.json').is_file():
            raise ValueError('application exited without complete capture evidence')
        ledger = guard.load(output/'child_step_accounting.json')
        if (ledger['completed_cases'] != 12 or ledger['in_flight']
                or ledger['case_id'] is not None
                or ledger['explicit_step_calls'] != ledger['returned_calls']
                or ledger['explicit_step_calls'] > p['limits']['all_explicit_simulation_step_calls']
                or ledger['reserve_calls'] > p['limits']['setup_reset_reserve_step_calls']
                or ledger['explicit_step_calls']-ledger['reserve_calls'] != 26400):
            raise ValueError('explicit physical allocation/evidence mismatch')
        guard.persist(output/'execution_ledger.json', {**ledger,
            'cumulative_explicit_step_calls':895+ledger['explicit_step_calls']})
        if time.monotonic()-started > p['limits']['gpu_wall_s']:
            raise TimeoutError('total allocation including cleanup exceeded')
        guard.persist(output/'FINISHED.json', {'applications':1, 'controlled_cases':12,
            'gpu_wall_s':time.monotonic()-started, 'saved_evidence_review_required':True,
            'NAV043_passed':False, 'G5_passed':False})
        return 0
    except BaseException as exception:
        error = repr(exception)
        application_error = (guard.load(output/'ERROR.json')
            if owns_output and (output/'ERROR.json').is_file() else None)
        # Persist the first failure before any further cleanup can fail or stall.
        if owns_output and not (output/'HELD.json').exists():
            guard.persist(output/'HELD.json', {'error':error,
                'primary_application_error':application_error,
                'applications_started':int(process is not None), 'exit_code':code,
                'timeout':timed_out, 'wall_s':time.monotonic()-started,
                'cleanup_pending':cleanup is None or not cleanup['cleanup_confirmed'],
                'no_automatic_retry':True, 'NAV043_passed':False, 'G5_passed':False})
        raise
    finally:
        if process is not None and cleanup is None:
            cleanup = stop_group(process)
        if cleanup is not None:
            code = cleanup['returncode']
        if owns_output and cleanup is not None:
            guard.persist(output/'process_cleanup.json', cleanup)
        guard.persist(guard.EVIDENCE/'exit.json', {'exit_code':code, 'error':error,
            'timeout':timed_out, 'wall_s':time.monotonic()-started,
            'failed_slot_wall_charge_s':p['recovery']['failed_slot_wall_charge_s'],
            'cleanup':cleanup, 'output_exists':output.exists(), 'retry_performed':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.check and args.child:
        parser.error('choose one entry')
    if args.check:
        p, context, identities = prepare()
        print('APPROVED_PREFLIGHT_PASS', context, identities, guard.gpu_ready(p), flush=True)
        return 0
    return child() if args.child else coordinate()


if __name__ == '__main__':
    raise SystemExit(main())
