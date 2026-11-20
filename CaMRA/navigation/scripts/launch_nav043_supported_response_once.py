"""CPU-only finite response-v3 diagnostic. Missing approval creates no marker."""
import os
from pathlib import Path
import subprocess
import time

import nav043_supported_response as candidate


def main():
    candidate.context()
    p = candidate.approved()
    output = Path(p['output_root'])
    if output.exists() or (candidate.HERE/'launch.lock').exists():
        raise ValueError('already attempted; no retry or alternate directory')
    # O_EXCL lock is durable, including a failed launch.
    with (candidate.HERE/'launch.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    started = time.monotonic()
    process, code, error = None, -1, None
    command = [candidate.base.PYTHON, '-B', '-c',
               'import nav043_supported_response as c; c.run(c.approved()["output_root"])']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(candidate.base.NAV/'scripts')
    try:
        with (candidate.HERE/'run.log').open('x') as log:
            process = subprocess.Popen(command, cwd=candidate.base.ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            candidate.base.write_new(candidate.HERE/'launch.json', dict(
                wrapper_pid=os.getpid(), child_pid=process.pid, command=command,
                pane=os.environ['TMUX_PANE'], output_root=str(output),
                proposal_sha256=candidate.base.sha(candidate.PROPOSAL),
                approval_sha256=candidate.base.sha(candidate.APPROVAL)))
            code = process.wait(timeout=candidate.LIMITS['cpu_wall_s'])
        if code == 0 and (not (output/'FINISHED.json').is_file() or (output/'HELD.json').exists()):
            raise ValueError('child exit zero without complete terminal evidence')
    except BaseException as exception:
        error = repr(exception)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        code = -1 if process is None else process.returncode
        if code == 0:
            code = -1
    finally:
        candidate.base.write_new(candidate.HERE/'exit.json', dict(exit_code=code, error=error,
            wrapper_wall_s=time.monotonic()-started, finished=(output/'FINISHED.json').is_file(),
            held=(output/'HELD.json').is_file(), retry_performed=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
