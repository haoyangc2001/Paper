"""Run exactly one explicitly authorized feedback attempt; audit before release.

The proposal has no SHA256 pin and training_approved=false, so this entrypoint
cannot turn a numerical repair or an unapproved draft into a training run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

NAV = Path(__file__).resolve().parents[1]
ROOT = Path('/pub/data/caohy/work/camra_navigation')
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'


def attempt_paths(prereg, root=ROOT):
    """Every output belongs to this registry; no historical attempt defaults."""
    prefix = prereg['runtime_contract']['run_prefix']
    if prefix != 'nav064_feedback_v1':
        raise ValueError('unregistered feedback attempt prefix')
    return {'coordinator': root / prefix,
            'audit100': root / (prefix + '_audit_u100'),
            'audit600': root / (prefix + '_audit_u600'),
            'development': NAV / 'reports' / (prefix + '_development.json'),
            'closure': NAV / 'reports' / (prefix + '_closure.json')}


def training_command(row, prereg_name):
    return [PYTHON, '-u', 'scripts/train_formal.py', '--run-id', row['run_id'],
            '--variant', row['variant'], '--seed', str(row['seed']), '--prereg-name', prereg_name]


def audit_command(n, prereg_name, paths):
    return [PYTHON, '-u', 'scripts/audit_r14a2.py', '--prereg', prereg_name,
            '--updates', str(n), '--out', str(paths[f'audit{n}'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prereg', required=True)
    args = ap.parse_args()
    if (not os.environ.get('TMUX') or socket.gethostname().split('.')[0] != 'tiger11'
            or sys.executable != PYTHON):
        raise SystemExit('registered tiger11/tmux/interpreter required')
    sys.path.insert(0, str(NAV / 'scripts'))
    import train_formal as F
    p, pin = F.verify_prereg(args.prereg)
    if p['implementation_versions']['estimator'] != 'rcppo-feedback-v4':
        raise ValueError('feedback estimator required')
    if (p['window_semantics']['max_updates'] != 600
            or p['window_semantics']['env_steps_per_update'] != 4086):
        raise ValueError('allocation must be six runs x600 x4086')
    paths = attempt_paths(p)
    for path in list(paths.values()) + [ROOT / r['run_id'] for r in p['run_registry']]:
        if path.exists(): raise FileExistsError(path)
    work = paths['coordinator']; work.mkdir()

    def write(name, value):
        with (work / name).open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')

    write('start.json', {'prereg_sha256': pin, 'time': datetime.datetime.now().astimezone().isoformat(),
          'host': socket.getfqdn(), 'tmux': os.environ['TMUX'], 'python': sys.executable,
          'cwd': str(NAV), 'git_head': subprocess.check_output(
              ['git', 'rev-parse', 'HEAD'], cwd=NAV, text=True).strip(), 'runs': p['run_registry']})
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=f'{NAV}/scripts:{NAV}/src:{NAV.parent}/calibration/src')
    processes = []; handles = []
    try:
        for row in p['run_registry']:
            handle = (work / (row['run_id'] + '.log')).open('x'); handles.append(handle)
            cmd = training_command(row, args.prereg)
            proc = subprocess.Popen(cmd, cwd=NAV, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((row, proc))
        write('pids.json', [dict(row, pid=proc.pid) for row, proc in processes])

        def audit(n):
            with (work / f'audit_u{n}.log').open('x') as log:
                subprocess.run(audit_command(n, args.prereg, paths),
                               cwd=NAV, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

        audited = False
        while True:
            counts = []
            for row, proc in processes:
                count = 0; log = ROOT / row['run_id'] / 'train_log.jsonl'
                if log.exists():
                    for line in log.read_text().splitlines():
                        try: record = json.loads(line)
                        except json.JSONDecodeError: continue
                        count += record.get('kind') == 'update'
                counts.append(count)
                if proc.poll() is not None and proc.returncode:
                    raise RuntimeError(f'{row["run_id"]} exited {proc.returncode}')
            print({'time': datetime.datetime.now().astimezone().isoformat(), 'updates': counts}, flush=True)
            if min(counts) >= 100 and not audited:
                for _, proc in processes:
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGSTOP)
                audit(100); audited = True
                for _, proc in processes:
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGCONT)
            if all(proc.poll() is not None for _, proc in processes): break
            time.sleep(30)
        if counts != [600]*6: raise RuntimeError(f'incomplete allocation {counts}')
        audit(600)
        with (work / 'development.log').open('x') as log:
            subprocess.run([PYTHON, '-u', 'scripts/eval_r14_repair.py', '--prereg', args.prereg,
                            '--out', str(paths['development'])], cwd=NAV, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        dev = json.loads(paths['development'].read_bytes())
        passed = dev['performance_pass']
        result = {'owner': 'NAV-064', 'prereg_sha256': pin,
                  'verdict': 'PASS' if passed else 'FAIL_DEVELOPMENT',
                  'performance': dev['summary'], 'updates': 3600,
                  'automatic_downstream_execution': False}
        with paths['closure'].open('x') as f: json.dump(result, f, indent=2)
        write('FINISHED.json', result)
        return 0 if passed else 2
    except BaseException as exc:
        # Preserve incomplete runs. No automatic restart or quota expansion.
        for _, proc in processes:
            if proc.poll() is None: os.kill(proc.pid, signal.SIGSTOP)
        write('HELD.json', {'reason': repr(exc), 'time': datetime.datetime.now().astimezone().isoformat()})
        raise
    finally:
        for handle in handles: handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
