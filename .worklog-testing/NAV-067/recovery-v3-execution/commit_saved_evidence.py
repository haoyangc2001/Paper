"""Complete local evidence commit; the first close attempt used a wrong log name."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert os.environ.get('TMUX_PANE') == '%599'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == '411ce7bf71a5c47332aae133015adfae71dfffd7'
receipt = json.loads((HERE/'manual-review-receipt.json').read_bytes())
assert receipt['saved_episode_results'] == receipt['trace_audited_episodes'] == 6
assert receipt['protected_hashes_checked'] == 295
plan_path = 'doc/plan/20261109_camra_navigation_implementation.json'
old = json.loads(subprocess.check_output(['git','show','HEAD:'+plan_path],cwd=ROOT))
new = json.loads((ROOT/plan_path).read_bytes())
assert {k:v for k,v in old.items() if k!='tasks'} == {k:v for k,v in new.items() if k!='tasks'}
for a,b in zip(old['tasks'],new['tasks']):
    assert {k:v for k,v in a.items() if k!='work_records'} == {k:v for k,v in b.items() if k!='work_records'}
    assert b['work_records'][:len(a['work_records'])] == a['work_records']
    if a['id'] != 'NAV-067': assert a == b
assert sum(t['status'] for t in new['tasks']) == 54
assert b'\r\n' not in (ROOT/plan_path).read_bytes()
report_path = 'CaMRA/navigation/reports/nav067_recovery_v3_execution_review.json'
report = json.loads((ROOT/report_path).read_bytes())
assert report['status'] == 'PILOT_TRACE_PASS_NAV067_INCOMPLETE'
assert not report['main_agent_saved_episode_review_failures'] and not report['main_agent_saved_runtime_review_failures']
names = ['COORDINATOR_STARTED.json','REVIEW_WATCH_STARTED.json','FINISHED.json','REVIEW_HELD.json',
    'complete_once.py','finish_review.py','coordinator.log','terminal-review.log','final-plan-check.log',
    'live-stack-L1-1.log','review_saved_episode.py','review_runtime.py','L1_observe.py',
    'L1-observational-diagnostics.json','L1-observational-diagnostics.log','manual-review-receipt.json',
    'manual-close-plan-check.log','close_after_log_check.py','manual-close.log',Path(__file__).name]
for level in ('L1_empty','L2_single_detour','L3_multi_path'):
    names.extend([prefix+level+'.json' for prefix in ('launch-','exit-','audit-')])
    names.append(level+'.log')
    names.extend(['runtime-review-'+level+suffix for suffix in ('.json','.log')])
    for method in ('uncalibrated','full'):
        names.append('main-agent-'+level+'-'+method+'.json')
        names.append('main-agent-L1-uncalibrated.log' if level=='L1_empty' and method=='uncalibrated' else 'main-agent-'+level+'-'+method+'.log')
paths = [plan_path,report_path,'CaMRA/navigation/docs/implementation_handoff.md']
paths.extend(str((HERE/name).relative_to(ROOT)) for name in names)
missing = [p for p in paths if not (ROOT/p).is_file()]
assert not missing, missing
staged = set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines())
assert staged.issubset(set(paths))
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()) == set(paths)
subprocess.run(['git','diff','--cached','--check','--','.',':(exclude,glob)**/*.log'],cwd=ROOT,check=True)
for p in paths: assert subprocess.check_output(['git','show',':'+p],cwd=ROOT) == (ROOT/p).read_bytes()
subprocess.run(['git','commit','-m','chore(nav): preserve NAV-067 physical pilot results and verification evidence'],cwd=ROOT,check=True)
print(json.dumps({'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'committed_files':len(paths),'new_simulation_starts':0,'training_updates':0,'W_queries':0,
    'remote_push':False,'NAV_067_status':False},ensure_ascii=False,indent=2),flush=True)
