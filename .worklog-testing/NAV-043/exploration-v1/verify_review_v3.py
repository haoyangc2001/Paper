"""Check saved audit sources and append-only plan changes; no new inference."""
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
root=Path.cwd()
nav=root/'CaMRA/navigation'
log=root/'.worklog-testing/NAV-043/exploration-v1'
def load(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

p='doc/plan/20261109_camra_navigation_implementation.json'
old=json.loads(subprocess.check_output(['git','show','aa17f6c:'+p],text=True))
new=load(p)
original_new=load(p)
for a,b in zip(old['tasks'],new['tasks']):
    assert a['id']==b['id'] and a['status']==b['status']
    if a['id']=='NAV-043':
        assert b['work_records'][:-1]==a['work_records']
        assert len(b['work_records'])==len(a['work_records'])+1
        b['work_records']=a['work_records']
    assert a==b,a['id']
assert old==new
tasks=original_new['tasks']
states={t['id']:t['status'] for t in tasks}
eligible=next(t['id'] for t in tasks if not t['status'] and all(states[d] for d in t['depends_on']))
assert len(tasks)==67 and sum(states.values())==55 and eligible=='NAV-043'
assert sum(len(t['work_records']) for t in tasks)==255
assert original_new['document']['version']=='1.4.1'
protocol_path=nav/'config/nav043_exploration_v1.json'
protocol=load(protocol_path)
assert sha(protocol_path)==Path(str(protocol_path)+'.sha256').read_text().strip()
for path,digest in protocol['sealed_hashes'].items(): assert sha(path)==digest,path
pair=load(nav/'config/nav066_decision_prereg_v2.json')
protected={str(nav/k):v for k,v in pair['implementation_hashes'].items()}
protected.update(pair['protected_hashes'])
protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in pair['six_candidates']})
for path,digest in protected.items(): assert sha(path)==digest,path
reviewer=nav/'scripts/audit_nav043_exploration.py'
source=reviewer.read_text()
compile(source,str(reviewer),'exec')
tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='audit_nominal_trace' for n in ast.walk(main))
report=load(log/'first-nominal-review.json')
assert report['reviewer_sha256']==sha(reviewer)
assert report['script_sha256']==sha(log/'review_first_nominal.py')
for path,digest in report['raw_sources'].items(): assert sha(path)==digest,path
assert report['preserved_first_review_error']==sha(log/'first-saved-review.log')
assert sum(r['saved_nominal_actions_replayed'] for r in report['pairs'])==726
assert sum(r['physics_ticks_accounted'] for r in report['pairs'])==17381
assert all(r['saved_business_metrics_exactly_reproduced'] for r in report['pairs'])
for name in ('review_first_saved.py','review_first_nominal.py'):
    compile((log/name).read_text(),name,'exec')
result={
    'at':datetime.now().astimezone().isoformat(),'host':socket.getfqdn(),
    'tmux':os.environ['TMUX'],'pane':os.environ.get('TMUX_PANE'),
    'status':'PASS_BASIC_REVIEW_AND_APPEND_INVARIANTS_NOT_G5',
    'baseline':'aa17f6c','eligible_task':eligible,'plan_version':'1.4.1',
    'completed_tasks':55,'total_tasks':67,'total_work_records':255,
    'only_one_NAV043_record_appended':True,'all_task_definitions_and_history_retained':True,
    'sealed_hashes_checked':len(protocol['sealed_hashes']),
    'protected_identities_checked':len(protected),
    'reviewer_sha256':sha(reviewer),'report_sha256':sha(log/'first-nominal-review.json'),
    'new_applications':0,'new_rollouts':0,'new_W_queries':0,'training_updates':0,
    'NAV_043_pass':False,'G5_pass':False,
}
with (log/'review-verification-v3.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
