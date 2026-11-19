"""Run the prepared saved-data applicability review once, after PID 59048.

This never starts, restarts or queries a navigation application or model.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LOG = ROOT / '.worklog-testing/NAV-043/exploration-v1'
PYTHON = '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
REVIEW = NAV / 'reports/nav043_exploration_review_v1.json'
OUTPUT = NAV / 'reports/nav043_exploration_applicability_v1.json'
SCRIPT = NAV / 'scripts/audit_nav043_exploration_applicability.py'
REVIEWER_PID = 59048


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(name, data):
    with (LOG / name).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        if raw.rsplit(')', 1)[1].split()[0] == 'Z':
            return None
        return raw.rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'tiger11'
    assert os.environ.get('TMUX') and sys.executable == PYTHON
    assert Path.cwd() == ROOT and not OUTPUT.exists()
    assert not (LOG / 'applicability-followup-start.json').exists()
    assert sha(SCRIPT) == 'fc37986d5a212c2b6b064c2ba1edf1f34f36fac49549f6abf57942292e148fe0'
    reviewer_start = identity(REVIEWER_PID)
    assert reviewer_start is not None, 'original review waiter is absent; inspect its result first'
    cmdline = Path(f'/proc/{REVIEWER_PID}/cmdline').read_bytes().split(b'\0')
    assert any(part.endswith(b'/review_when_finished.py') for part in cmdline)
    ready = {'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
             'pane': os.environ.get('TMUX_PANE'), 'pid': os.getpid(),
             'original_reviewer_pid': REVIEWER_PID, 'original_reviewer_start_ticks': reviewer_start,
             'script_sha256': sha(SCRIPT), 'waiter_sha256': sha(__file__),
             'new_applications': 0, 'new_rollouts': 0, 'W_queries': 0, 'training_updates': 0}
    if args.check:
        print({'status': 'PASS_READ_ONLY_FOLLOWUP_GUARDS', **ready}, flush=True)
        return
    write_new('applicability-followup-start.json', ready)
    print('WAITING_FOR_ORIGINAL_REVIEWER', REVIEWER_PID, reviewer_start, flush=True)
    deadline = time.monotonic() + 14400
    try:
        while identity(REVIEWER_PID) == reviewer_start:
            if time.monotonic() >= deadline:
                raise TimeoutError('original reviewer did not finish; no retry or physical action')
            time.sleep(5)
        assert REVIEW.is_file(), 'original reviewer ended without report; preserve and inspect its failure'
        review = json.loads(REVIEW.read_bytes())
        assert review['status'] == 'SAVED_EVIDENCE_REVIEW_NOT_G5'
        assert sha(SCRIPT) == ready['script_sha256']
        print('ORIGINAL_REVIEW_COMPLETE', sha(REVIEW), flush=True)
        completed = subprocess.run([PYTHON, '-B', str(SCRIPT)], check=False)
        assert completed.returncode == 0 and OUTPUT.is_file(), 'read-only applicability review failed; no retry'
        output = json.loads(OUTPUT.read_bytes())
        assert output['status'] == 'DESCRIPTIVE_BUDGET_SELECTED_MARGIN_REVIEW_NOT_G5'
        write_new('applicability-followup-finished.json',
                  {'at': datetime.now().astimezone().isoformat(), 'review_sha256': sha(REVIEW),
                   'output_sha256': sha(OUTPUT), 'returncode': completed.returncode,
                   'NAV043_passed': False, 'G5_passed': False})
        print('SAVED_APPLICABILITY_REVIEW_FINISHED_NOT_G5', flush=True)
    except BaseException:
        write_new('applicability-followup-error.json',
                  {'at': datetime.now().astimezone().isoformat(), 'traceback': traceback.format_exc(),
                   'physical_retry': False, 'review_retry': False})
        raise


if __name__ == '__main__':
    main()
