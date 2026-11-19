"""Read the complete current plan without modifying it or starting Kit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert os.environ.get('TMUX_PANE') == '%599'
data = PLAN.read_bytes()
plan = json.loads(data)
tasks = plan['tasks']
by_id = {t['id']: t for t in tasks}
eligible = [t['id'] for t in tasks if not t['status'] and all(by_id[d]['status'] for d in t['depends_on'])]
assert eligible[0] == 'NAV-067' and sum(t['status'] for t in tasks) == 54
assert len(by_id['NAV-067']['work_records']) == 20
records = []
for task in tasks:
    for record in task['work_records']:
        assert all(k in record for k in ('timestamp','summary','actions','files_changed','commands','results','evidence','remaining_work','blockers'))
        records.append({'task':task['id'],'summary':record['summary'],'sha256':hashlib.sha256(json.dumps(record,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
v = subprocess.run([sys.executable,'-B','/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',str(PLAN),'--mode','update'],cwd=ROOT,capture_output=True,text=True)
assert v.returncode == 0, v.stdout + v.stderr
report = {'plan_sha256':hashlib.sha256(data).hexdigest(),'bytes_read':len(data),'record_count':len(records),'records':records,'completed':54,'eligible':eligible,'policies':{k:value for k,value in plan.items() if k!='tasks'},'current_task':by_id['NAV-067'],'next_task_read_only':by_id['NAV-043'],'validator':v.stdout+v.stderr,'new_applications':0,'plan_changes':0}
with (HERE/'resumed-plan-read.json').open('x') as stream:
    json.dump(report,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps({'completed':54,'eligible':eligible,'records':len(records),'validator':v.stdout,'current_record_summaries':[r['summary'] for r in by_id['NAV-067']['work_records']],'next_task_read_only':by_id['NAV-043']},ensure_ascii=False,indent=2),flush=True)
