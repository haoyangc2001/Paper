"""Final main-agent audit; does not run a candidate optimizer or rollout."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
NAV = ROOT / 'CaMRA/navigation'
PLAN = 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review = read(NAV / 'reports/nav065_local_probe_v9_preparation_review.json')
before = json.loads(subprocess.check_output(['git', 'show', review['base_git'] + ':' + PLAN], cwd=ROOT))
current = read(ROOT / PLAN)
assert before == read(HERE / 'baseline.json')['plan']
assert {k:v for k,v in before.items() if k != 'tasks'} == {k:v for k,v in current.items() if k != 'tasks'}
for old, new in zip(before['tasks'], current['tasks'], strict=True):
    if old['id'] == 'NAV-065':
        assert {k:v for k,v in old.items() if k != 'work_records'} == {k:v for k,v in new.items() if k != 'work_records'}
        assert new['work_records'][:len(old['work_records'])] == old['work_records']
        assert len(new['work_records']) == len(old['work_records']) + 2
    else:
        assert new == old
states = {t['id']:t['status'] for t in current['tasks']}
assert sum(states.values()) == 52 and len(states) == 67
assert next(t['id'] for t in current['tasks'] if not t['status'] and all(states[x] for x in t['depends_on'])) == 'NAV-065'
for name, digest in review['evidence'].items():
    assert sha(ROOT / name) == digest, name
for path, digest in read(NAV / 'reports/nav064_terminal_tail_training_release_v1.json')['v2_verified_files'].items():
    assert sha(Path(path)) == digest, path
assert not (NAV / 'config/nav065_local_probe_prereg_v9.json').exists()
assert not Path('/pub/data/caohy/work/camra_navigation/nav065_local_probe_v9').exists()
with (HERE / 'validation.log').open('x') as log:
    result = subprocess.run([sys.executable, '-B', '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
        str(ROOT / PLAN), '--mode', 'update'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
assert result.returncode == 0, 'inspect validation.log'
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
protected = ['RC-PPO', 'HiRA', 'manuscripts', 'CaMRA/locomotion', 'CaMRA/calibration']
assert not subprocess.check_output(['git','diff','--name-only','--',*protected],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT/'HiRA',text=True).strip()
tracked = {'AGENTS.md', 'CaMRA/navigation/README.md', 'CaMRA/navigation/docs/implementation_handoff.md', PLAN}
assert set(subprocess.check_output(['git','diff','--name-only'], cwd=ROOT, text=True).splitlines()) == tracked
report = {'timestamp':datetime.now().astimezone().isoformat(), 'base_git':review['base_git'],
    'host':socket.getfqdn(), 'tmux':os.environ['TMUX'], 'python':sys.executable,
    'legacy_validation_pass':True, 'old_plan_fields_and_records_preserved':True,
    'new_NAV065_records':2, 'completed_tasks':52, 'next_task':'NAV-065',
    'v2_files_verified':33, 'all_preparation_evidence_hashes_match':True,
    'formal_updates':0, 'new_rollouts':0, 'development_scoring_rounds':0,
    'new_registration_exists':False, 'new_formal_output_exists':False,
    'NAV065_scientific_pass':False, 'NAV066_started':False,
    'evidence':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/PLAN, ROOT/'AGENTS.md',
        NAV/'README.md', NAV/'docs/implementation_handoff.md',
        NAV/'reports/nav065_local_probe_v9_preparation_review.json', HERE/'validation.log']}}
with (HERE/'closure.json').open('x') as f:
    json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k != 'evidence'},ensure_ascii=False))
