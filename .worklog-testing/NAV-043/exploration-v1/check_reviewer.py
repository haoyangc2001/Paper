"""Static review readiness only; no networks, simulation, or new rollout."""
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source=NAV/'scripts/audit_nav043_exploration.py'
code=source.read_text()
compile(code,str(source),'exec')
tree=ast.parse(code)
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
               and isinstance(n.func.value,ast.Name)
               and n.func.value.id in ('net','network')
               and n.func.attr in ('act','values') for n in ast.walk(tree))
protocol_path=NAV/'config/nav043_exploration_v1.json'
protocol=json.loads(protocol_path.read_text())
assert digest(protocol_path)==Path(str(protocol_path)+'.sha256').read_text().strip()
for path,value in protocol['sealed_hashes'].items():
    assert digest(path)==value,path
plan_path=ROOT/'doc/plan/20261109_camra_navigation_implementation.json'
plan=json.loads(plan_path.read_text())
states={t['id']:t['status'] for t in plan['tasks']}
first=next(t['id'] for t in plan['tasks']
           if not t['status'] and all(states[d] for d in t['depends_on']))
assert first=='NAV-043' and sum(states.values())==55
assert plan['document']['version']=='1.4.1'
result={
    'at':datetime.now().astimezone().isoformat(),
    'status':'STATIC_REVIEW_READY_NOT_SAVED_TRACE_ACCEPTANCE',
    'host':socket.getfqdn(),'tmux':os.environ['TMUX'],
    'tmux_pane':os.environ.get('TMUX_PANE'),
    'reviewer_sha256':digest(source),
    'sealed_sources_unchanged':len(protocol['sealed_hashes']),
    'plan_version':plan['document']['version'],
    'first_eligible':first,'completed_tasks':sum(states.values()),
    'work_records':sum(len(t['work_records']) for t in plan['tasks']),
    'all_plan_fields_parsed':list(plan),
    'checks':['syntax','no actor/critic inference calls',
              'sealed protocol and sources unchanged','current task and dependencies'],
    'saved_nominal_replay':'pending original allocation completion',
    'new_applications':0,'new_rollouts':0,'new_W_queries':0,
}
out=ROOT/'.worklog-testing/NAV-043/exploration-v1/review-ready.json'
with out.open('x') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
