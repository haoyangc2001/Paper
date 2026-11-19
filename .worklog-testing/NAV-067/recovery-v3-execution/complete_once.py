"""Observe the original coordinator, then record and commit once; never simulate."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED_HEAD = '411ce7bf71a5c47332aae133015adfae71dfffd7'


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def run():
    assert socket.gethostname().split('.')[0] == 'tiger11'
    pane = os.environ['TMUX_PANE']
    tmux = subprocess.check_output(['tmux','display-message','-t',pane,'-p',
        '#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip()
    assert tmux.startswith('nav064-r14-repair-20261115:'), tmux
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == EXPECTED_HEAD
    started = json.loads((HERE/'COORDINATOR_STARTED.json').read_bytes())
    pid = started['pid']
    stat = Path('/proc')/str(pid)/'stat'
    def identity():
        try:
            return stat.read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:
            return None
    original_identity = identity()
    write_new(HERE/'REVIEW_WATCH_STARTED.json',{'at':datetime.now().astimezone().isoformat(),
        'host':socket.getfqdn(),'pid':os.getpid(),'coordinator_pid':pid,
        'coordinator_start_ticks':original_identity,'tmux':tmux,'cwd':str(ROOT),
        'python':sys.executable,'command':sys.argv,'simulation_starts':0,'automatic_retries':0})
    while original_identity is not None and identity() == original_identity:
        time.sleep(20)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == EXPECTED_HEAD
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip()
    with (HERE/'terminal-review.log').open('x') as stream:
        subprocess.run([sys.executable,'-B',str(HERE/'finish_review.py')],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    relative = HERE.relative_to(ROOT)
    paths = ['CaMRA/navigation/reports/nav067_recovery_v3_execution_review.json',
             'CaMRA/navigation/docs/implementation_handoff.md',
             'doc/plan/20261109_camra_navigation_implementation.json',
             str(relative/'finish_review.py'),str(relative/'complete_once.py')]
    exact = ['COORDINATOR_STARTED.json','REVIEW_WATCH_STARTED.json','FINISHED.json','HELD.json',
        'coordinator.log','terminal-review.log','final-plan-check.log','live-stack-L1-1.log']
    for level in ('L1_empty','L2_single_detour','L3_multi_path'):
        exact.extend(['launch-'+level+'.json','exit-'+level+'.json','audit-'+level+'.json',level+'.log'])
    paths.extend(str(relative/name) for name in exact if (HERE/name).is_file())
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
    staged = set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())
    assert staged and staged.issubset(set(paths)), staged
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','chore(nav): preserve one-time NAV-067 physical recovery results'],cwd=ROOT,check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report = json.loads((ROOT/paths[0]).read_bytes())
    write_new(HERE/'REVIEW_COMMITTED.json',{'at':datetime.now().astimezone().isoformat(),
        'git_head':head,'report_status':report['status'],'saved_episode_results':report['saved_episode_results'],
        'trace_audited_episodes':report['trace_audited_episodes'],'NAV_067_status':False,
        'remote_push':False,'simulation_starts':0,'automatic_retries':0})
    print(json.dumps({'git_head':head,'status':report['status']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        detail = traceback.format_exc()
        write_new(HERE/'REVIEW_HELD.json',{'at':datetime.now().astimezone().isoformat(),
            'traceback':detail,'automatic_retries':0})
        print(detail,file=sys.stderr,flush=True)
        raise SystemExit(1)
