"""Commit only this reviewed startup repair; no remote Git operation."""
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
OUT = ROOT/'.worklog-testing/NAV-067/startup-recovery-v2'
paths = [
    'CaMRA/navigation/scripts/nav067_runtime.py',
    'CaMRA/navigation/scripts/evaluate_navigation_v3.py',
    'CaMRA/navigation/tests/test_nav067_runtime.py',
    'CaMRA/navigation/config/nav067_startup_recovery_proposal_v2.json',
    'CaMRA/navigation/reports/nav067_startup_recovery_preparation_v2.json',
    'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json']
evidence = ['preflight.py','preflight.json','preflight.log',
    'probe.py','launch_probe.py','probe-process.json','probe-exit.json',
    'probe-console.log','probe-launch.log','probe-attempt1/stages.jsonl',
    'probe-attempt1/RESULT.json','prepare.py','prepare.log','basic-tests.log',
    'final_review.py','final-review.json','final-review.log',
    'final-basic-tests.log','final-plan-check.log','commit_recovery.py']
paths += [str((OUT/name).relative_to(ROOT)) for name in evidence]
review = json.loads((OUT/'final-review.json').read_bytes())
assert review['status'] == 'PASS_BASIC_REVIEW_ONLY'
assert '29 passed' in review['tests_output']
assert review['validator_output'].startswith('VALID:')
assert review['approved_amendment_absent']
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert not staged, staged
for path in paths:
    assert (ROOT/path).is_file(), path
    assert Path(path).suffix.lower() not in ('.tex','.bib','.pdf'), path
    assert not path.startswith(('manuscripts/','submission','docs/paper','RC-PPO/','HiRA/'))
changed = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(changed) <= set(paths), changed
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
assert set(staged) == set(paths), staged
# Preserve raw simulator logs byte for byte, including their original whitespace.
checked = [p for p in paths if Path(p).suffix not in ('.log','.txt','.csv','.jsonl')]
subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                'diff','--cached','--check','--',*checked],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m',
    'fix(nav): bind local runtime and preserve startup failures before cleanup'],cwd=ROOT,check=True)
commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
names = subprocess.check_output(['git','show','--format=','--name-only',commit],cwd=ROOT,text=True).splitlines()
assert set(names) == set(paths)
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
result = {'timestamp':datetime.now().astimezone().isoformat(),'commit':commit,
          'files':paths,'pushed':False,'navigation_episodes_started':0,
          'unrelated_untracked_artifacts_preserved':True}
with (OUT/'commit-result.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
