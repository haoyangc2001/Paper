import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path('/home/caohy/repositories/Paper')
OUT = ROOT / '.worklog-testing/NAV-067/clean-v2'
raw = (ROOT / 'doc/plan/20261109_camra_navigation_implementation.json').read_bytes()
plan = json.loads(raw)
states = {t['id']: t['status'] for t in plan['tasks']}
next_task = next(t for t in plan['tasks'] if not t['status'] and all(states[d] for d in t['depends_on']))
assert next_task['id'] == 'NAV-067', next_task['id']
for key, value in plan.items():
    if key != 'tasks':
        (OUT / ('plan-' + key + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2))
(OUT / 'task-definitions.json').write_text(json.dumps([{k:v for k,v in t.items() if k not in ('work_records', 'test_reports')} for t in plan['tasks']], ensure_ascii=False, indent=2))
(OUT / 'current-task.json').write_text(json.dumps(next_task, ensure_ascii=False, indent=2))
snapshot = {
    'plan_sha256': hashlib.sha256(raw).hexdigest(),
    'version': plan['document']['version'],
    'completed': sum(states.values()),
    'next': next_task['id'],
    'top_keys': list(plan),
    'all_tasks_fully_parsed': len(plan['tasks']),
    'all_work_records_fully_parsed': sum(len(t['work_records']) for t in plan['tasks']),
    'hostname': subprocess.check_output(['hostname'], text=True).strip(),
    'python': subprocess.check_output(['/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python', '--version'], text=True).strip(),
    'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,noheader'], text=True).strip(),
    'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
}
(OUT / 'baseline.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
print(json.dumps(snapshot, ensure_ascii=False, indent=2))
print(json.dumps(next_task, ensure_ascii=False, indent=2))
