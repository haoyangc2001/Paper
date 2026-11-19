"""One approved affine diagnostic only; persistent lock and no retry."""
from datetime import datetime
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import nav043_affine_response as affine

base = affine.base
HERE = base.ROOT/'.worklog-testing/NAV-043/affine-response-v2'
READINESS = base.NAV/'config/nav043_affine_response_readiness_v2.json'


def prepare():
    if (socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX')
            or Path(sys.executable).resolve() != Path(base.PYTHON).resolve()):
        raise ValueError('requires tiger11/tmux/registered Python')
    proposal, approval = affine.approval()
    if base.sha(READINESS) != Path(str(READINESS)+'.sha256').read_text().strip():
        raise ValueError('changed readiness pin')
    readiness = base.load(READINESS)
    if readiness['status'] != 'PREPARED_ONLY' or readiness['proposal_sha256'] != base.sha(affine.PROPOSAL):
        raise ValueError('readiness is not approval for this exact proposal')
    for path, digest in readiness['source_hashes'].items():
        if base.sha(path) != digest:
            raise ValueError('changed supporting source')
    if Path(proposal['output_root']).exists():
        raise ValueError('output already exists; do not retry or move directories')
    if (HERE/'launch.lock').exists():
        raise ValueError('launch already attempted; do not retry')
    return proposal, approval


def main():
    # No markers or subprocess are created while approval is missing.
    proposal, approval = prepare()
    HERE.mkdir(exist_ok=True)
    with (HERE/'launch.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    command = [base.PYTHON, '-B', str(base.NAV/'scripts/nav043_affine_response.py'), '--run']
    started, process, code, error = time.monotonic(), None, -1, None
    output = Path(proposal['output_root'])
    try:
        with (HERE/'run.log').open('x') as log:
            process = subprocess.Popen(command, cwd=base.ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            base.write_new(HERE/'launch.json', {'at': datetime.now().astimezone().isoformat(),
                'host': socket.gethostname(), 'pane': os.environ.get('TMUX_PANE'),
                'wrapper_pid': os.getpid(), 'probe_pid': process.pid, 'command': command,
                'cwd': str(base.ROOT), 'output_root': str(output),
                'proposal_sha256': base.sha(affine.PROPOSAL),
                'approval_sha256': base.sha(affine.APPROVAL),
                'readiness_sha256': base.sha(READINESS),
                'source_sha256': base.sha(base.NAV/'scripts/nav043_affine_response.py'),
                'thread_limits': {k: environment[k] for k in
                                 ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                'GPU_visible_devices': ''})
            code = process.wait(timeout=proposal['limits']['cpu_wall_s'])
    except BaseException as exception:
        error = repr(exception)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        code = -1 if process is None else process.returncode
        if code == 0:
            code = -1
    finally:
        base.write_new(HERE/'exit.json', {'at': datetime.now().astimezone().isoformat(),
            'exit_code': code, 'error': error, 'wrapper_wall_s': time.monotonic()-started,
            'finished': (output/'FINISHED.json').is_file(), 'held': (output/'HELD.json').is_file(),
            'retry_performed': False})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
