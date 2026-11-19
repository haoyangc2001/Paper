"""Wait for the existing allocation, then review saved evidence once; no Kit."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUTPUT=Path('/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1')

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def save(name, data):
    with (HERE/name).open('x') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2)
        stream.write('\n')

def main():
    assert os.environ.get('TMUX_PANE')=='%599'
    launch=json.loads((HERE/'launch.json').read_bytes())
    pid=launch['pid']
    command=Path('/proc')/str(pid)/'cmdline'
    assert command.is_file() and b'run_nav067_remaining_recovery_v2.py' in command.read_bytes()
    save('saved-review-watcher-start.json',{'at':now(),'pid':os.getpid(),'existing_coordinator_pid':pid,
        'tmux_pane':os.environ['TMUX_PANE'],'python':sys.executable,
        'new_simulator_applications':0,'new_episodes':0,'training_updates':0,
        'review_retries':0,'plan_status_changes':0})
    for _ in range(482):
        if (OUTPUT/'RECOVERY_V2_HELD.json').exists():
            save('saved-review-watcher-held.json',{'at':now(),'status':'EXISTING_ALLOCATION_HELD_NO_RETRY',
                'held':json.loads((OUTPUT/'RECOVERY_V2_HELD.json').read_bytes()),'NAV_067_status':False})
            return 2
        if (OUTPUT/'RECOVERY_V2_FINISHED.json').exists():
            args=[sys.executable,'-B',str(HERE/'review_saved.py'),'--full']
            with (HERE/'review-full.log').open('x') as stream:
                result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            save('saved-review-watcher-exit.json',{'at':now(),'review_command':args,'review_exit_code':result.returncode,
                'status':'SAVED_EVIDENCE_REVIEW_PASS_PENDING_MAIN_AGENT_CLOSURE' if result.returncode==0 else 'SAVED_EVIDENCE_REVIEW_FAILED_NO_RETRY',
                'NAV_067_status':False,'plan_status_changes':0,'new_simulator_applications':0})
            return result.returncode
        if not command.exists():
            save('saved-review-watcher-held.json',{'at':now(),'status':'COORDINATOR_EXITED_WITHOUT_TERMINAL_EVIDENCE',
                'NAV_067_status':False,'no_retry':True})
            return 3
        time.sleep(30)
    save('saved-review-watcher-held.json',{'at':now(),'status':'READ_ONLY_WATCH_LIMIT_REACHED',
        'NAV_067_status':False,'no_simulator_restart':True})
    return 4

if __name__=='__main__':
    raise SystemExit(main())
