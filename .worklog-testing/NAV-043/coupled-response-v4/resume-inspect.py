"""Read the complete active plan and report current scope; no scientific solve."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/home/caohy/repositories/Paper')
path = root/'doc/plan/20261109_camra_navigation_implementation.json'
plan = json.loads(path.read_text())
print('PLAN_BYTES', path.stat().st_size)
print('TOP_KEYS', list(plan))
for key in ('global_constraints', 'uncertainty_policy', 'execution_alignment_policy',
            'upstream_correction_policy'):
    print(key, json.dumps(plan.get(key), ensure_ascii=False, indent=2))
states = {task['id']: task['status'] for task in plan['tasks']}
eligible = [task['id'] for task in plan['tasks']
            if not task['status'] and all(states[d] for d in task['depends_on'])]
print('STATES', states, 'ELIGIBLE', eligible, 'PANE', os.environ['TMUX_PANE'])
task = next(t for t in plan['tasks'] if t['id'] == eligible[0])
print('CURRENT_DEFINITION', json.dumps({k: v for k, v in task.items()
                                      if k != 'work_records'}, ensure_ascii=False, indent=2))
print('LATEST_RECORDS', json.dumps(task['work_records'][-3:], ensure_ascii=False, indent=2))
print('GIT', subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True))
print('HEAD', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True))
print('COMPLETE_PLAN_READ_AND_PARSED', len(plan['tasks']))
