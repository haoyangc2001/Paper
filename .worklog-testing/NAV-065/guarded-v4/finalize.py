"""Read-only closeout checks; never fit or rescore development."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
LAUNCH = Path('/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_launch_20261118_v4')
PLAN = ROOT / 'doc/plan/20261109_camra_navigation_implementation.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

current = read(PLAN)
before = read(LAUNCH / 'before-plan.json')
assert len(current['tasks']) == len(before['tasks']) == 67
for old, new in zip(before['tasks'], current['tasks']):
    assert old['id'] == new['id'] and old['status'] == new['status']
    assert old['depends_on'] == new['depends_on']
    assert new['work_records'][:len(old['work_records'])] == old['work_records'], old['id']
    if old['id'] != 'NAV-065':
        assert old == new, old['id']
assert all(before[k] == current[k] for k in before if k != 'tasks')
done = {t['id']: t['status'] for t in current['tasks']}
ready = next(t['id'] for t in current['tasks'] if not t['status'] and all(done[d] for d in t['depends_on']))
assert sum(done.values()) == 52 and ready == 'NAV-065' and done['NAV-064']

# Also describe every change since the latest local commit without normalizing JSON.
committed = json.loads(subprocess.check_output(['git', 'show', 'HEAD:' + str(PLAN.relative_to(ROOT))], cwd=ROOT))
changed_fields = []
for old, new in zip(committed['tasks'], current['tasks']):
    assert old['id'] == new['id'] and old['status'] == new['status']
    assert new['work_records'][:len(old['work_records'])] == old['work_records'], old['id']
    for key in old.keys() | new.keys():
        if key != 'work_records' and old.get(key) != new.get(key):
            changed_fields.append({'task': old['id'], 'field': key})

subprocess.run([sys.executable, '/home/caohy/.codex/skills/create-plan-worklog-legacy/scripts/validate_worklog.py',
                str(PLAN), '--mode', 'update'], cwd=ROOT, check=True)
subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--check'], cwd=ROOT, check=True)
for name, digest in read(LAUNCH / 'source_before.json').items():
    assert sha(NAV / name) == digest, name
review = read(NAV / 'reports/nav065_guarded_refit_v4_review.json')
assert review == read(LAUNCH / 'review.json')
output = Path(review['evidence']['output'])
for name, digest in review['evidence']['output_hashes'].items():
    assert sha(output / name) == digest, name
registration = NAV / 'config/nav065_guarded_refit_prereg_v4.json'
assert sha(registration) == registration.with_suffix('.json.sha256').read_text().strip() == review['registration_pin']
assert review['execution_integrity_pass'] and not review['scientific_pass']
assert review['optimizer_step_trials'] == review['accepted_updates'] == 1
assert review['development_scoring_rounds'] == 1 and review['development_false_feasible'] == 4
assert read(output / 'gate_G4_alignment_v2.json')['verdict'] == 'FAIL'
assert not (output / 'deployment_candidates_v2.json').exists()

changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
prohibited = [p for p in changed if p.startswith(('manuscripts/', 'RC-PPO/', 'HiRA/', 'docs/paper', 'submission'))
              or p.endswith(('.tex', '.bib'))]
assert not prohibited, prohibited
report = {'timestamp': datetime.now().astimezone().isoformat(), 'host': socket.getfqdn(),
          'tmux': os.environ['TMUX'], 'python': sys.executable, 'task': ready,
          'plan_completed': 52, 'plan_total': 67, 'legacy_validator_pass': True,
          'all_historical_work_records_preserved': True,
          'changes_since_HEAD_other_than_work_records': changed_fields,
          'v4_source_and_output_hashes_pass': True, 'scientific_pass': False,
          'g4_verdict': 'FAIL', 'new_fit_or_development_scoring': 0,
          'prohibited_tracked_changes': prohibited}
with (LAUNCH / 'finalize.json').open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
