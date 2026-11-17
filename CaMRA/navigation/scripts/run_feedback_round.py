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
    versions = prereg.get('implementation_versions', {})
    allowed = {'nav064_feedback_v1': 'obs-v5', 'nav064_time_feedback_v1': 'obs-v6',
               'nav064_clearance_feedback_v1': 'obs-v7', 'nav064_episode_feedback_v1': 'obs-v7',
               'nav064_avoidance_prior_v1': 'obs-v7', 'nav064_avoidance_prior_v2': 'obs-v7',
               'nav064_terminal_tail_v1': 'obs-v7'}
    if (prefix not in allowed
            or versions.get('observation', 'obs-v5') != allowed[prefix]):
        raise ValueError('unregistered feedback attempt prefix')
    first, total = (allocation(prereg) if prefix in
                    ('nav064_episode_feedback_v1', 'nav064_avoidance_prior_v1',
                     'nav064_avoidance_prior_v2', 'nav064_terminal_tail_v1') else (100, 600))
    paths = {'coordinator': root / prefix,
            f'audit{first}': root / (prefix + f'_audit_u{first}'),
            f'audit{total}': root / (prefix + f'_audit_u{total}'),
            'development': NAV / 'reports' / (prefix + '_development.json'),
            'closure': NAV / 'reports' / (prefix + '_closure.json')}
    if prefix == 'nav064_terminal_tail_v1':
        paths['audit150'] = root / (prefix + '_audit_u150')
    return paths


def training_command(row, prereg_name):
    return [PYTHON, '-u', 'scripts/train_formal.py', '--run-id', row['run_id'],
            '--variant', row['variant'], '--seed', str(row['seed']), '--prereg-name', prereg_name]


def audit_command(n, prereg_name, paths):
    return [PYTHON, '-u', 'scripts/audit_r14a2.py', '--prereg', prereg_name,
            '--updates', str(n), '--out', str(paths[f'audit{n}'])]


def allocation(prereg):
    w=prereg['window_semantics']
    if prereg['runtime_contract']['run_prefix'] in (
            'nav064_episode_feedback_v1', 'nav064_avoidance_prior_v1', 'nav064_avoidance_prior_v2',
            'nav064_terminal_tail_v1'):
        if (w.get('collection')!='complete_episodes_v1' or w['max_updates']!=300
                or w['calls_per_update']!=450 or w['env_steps_per_update']!=8100):
            raise ValueError('episode allocation must be six runs x300, at most8100 samples/update')
        return 50,300
    if w['max_updates']!=600 or w['env_steps_per_update']!=4086:
        raise ValueError('allocation must be six runs x600 x4086')
    return 100,600


def enforce_exploration(prereg, root, updates, write):
    if prereg.get('exploration_gate') is None:
        return
    from check_detour_exploration import check_prefix
    exploration = check_prefix(prereg, root, updates)
    write('exploration_u50.json', exploration)
    if exploration['verdict'] != 'PASS':
        raise RuntimeError('train-prefix detour exploration failed; allocation held, no retry')


def enforce_learning_stability(prereg, root, updates, write):
    from camra_navigation.algorithm.stability import check_learning_stability
    result = check_learning_stability(prereg, root, updates)
    write(f'learning_stability_u{updates}.json', result)
    if result['verdict'] != 'PASS':
        raise RuntimeError('train learning deteriorated; allocation held, no retry')


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
    from camra_navigation.algorithm.storage import FEEDBACK_ESTIMATORS
    if p['implementation_versions']['estimator'] not in FEEDBACK_ESTIMATORS:
        raise ValueError('feedback estimator required')
    prefix_updates, total_updates = allocation(p)
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
        stability_audited = 'learning_stability_gate' not in p
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
            if min(counts) >= prefix_updates and not audited:
                for _, proc in processes:
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGSTOP)
                audit(prefix_updates); audited = True
                enforce_exploration(p, ROOT, prefix_updates, write)
                for _, proc in processes:
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGCONT)
            if not stability_audited and min(counts) >= p['learning_stability_gate']['after_updates']:
                for _, proc in processes:
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGSTOP)
                n = p['learning_stability_gate']['after_updates']
                audit(n)
                enforce_learning_stability(p, ROOT, n, write)
                stability_audited = True
                for _, proc in processes:
                    if proc.poll() is None: os.kill(proc.pid, signal.SIGCONT)
            if all(proc.poll() is not None for _, proc in processes): break
            time.sleep(30)
        if counts != [total_updates]*6: raise RuntimeError(f'incomplete allocation {counts}')
        audit(total_updates)
        with (work / 'development.log').open('x') as log:
            subprocess.run([PYTHON, '-u', 'scripts/eval_r14_repair.py', '--prereg', args.prereg,
                            '--out', str(paths['development'])], cwd=NAV, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        dev = json.loads(paths['development'].read_bytes())
        passed = dev['performance_pass']
        result = {'owner': 'NAV-064', 'prereg_sha256': pin,
                  'verdict': 'PASS' if passed else 'FAIL_DEVELOPMENT',
                  'performance': dev['summary'], 'updates': total_updates*6,
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
