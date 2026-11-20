"""Read existing context and failed terminal accounting; never launch Isaac."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav043_motion_probe_v1/recovery_v2')
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
p = json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_text())
print('PLAN_KEYS', list(p))
for name in ('version', 'global_constraints', 'uncertainty_policy', 'execution_alignment_policy', 'upstream_correction_policy'):
    print(name, json.dumps(p.get(name), ensure_ascii=False, indent=2))
byid = {t['id']:t for t in p['tasks']}
eligible = [t['id'] for t in p['tasks'] if not t['status'] and all(byid[d]['status'] for d in t['depends_on'])]
print('TASKS', len(p['tasks']), 'TRUE', sum(t['status'] for t in p['tasks']), 'ELIGIBLE', eligible)
t = byid['NAV-043']
print('NAV043_DEFINITION', json.dumps({k:v for k,v in t.items() if k!='work_records'}, ensure_ascii=False, indent=2))
print('NAV043_HISTORY', json.dumps([{'timestamp':r['timestamp'],'summary':r['summary']} for r in t['work_records']], ensure_ascii=False, indent=2))
print('LATEST_RECORDS', json.dumps(t['work_records'][-2:], ensure_ascii=False, indent=2))
print('GIT', subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True))
print('HEAD', subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
for name in ('OWNER.json','PROCESS.json','runtime.json','CAPTURE_FINISHED.json','pre_cleanup_step_accounting.json','HELD.json','process_cleanup.json'):
    v = json.loads((OUT/name).read_text())
    print(name, json.dumps(v, ensure_ascii=False, indent=2))
for name in ('reset.json','completed_q.json','result.json','energy_metadata.json'):
    v = json.loads((OUT/'M01'/name).read_text())
    print('M01/'+name, json.dumps(v, ensure_ascii=False, indent=2) if name in ('reset.json','result.json') else list(v))
print('TMUX', subprocess.check_output(['tmux','display-message','-p','#{session_name} #{window_id} #{pane_id}'], text=True).strip())
print('REGISTERED_PROCESS_ABSENCE', {pid:not (Path('/proc')/str(pid)).exists() for pid in (2825601,2825628)})
