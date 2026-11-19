"""Finish the reviewed commit, preserving the prior whitespace-check failure."""
import json
from pathlib import Path
import subprocess

from nav067_sources import NAV, ROOT, read_manifest, sha, write_new

OUT = Path(__file__).resolve().parent
read_manifest()
proposal = json.loads((NAV/'config/nav067_startup_recovery_proposal_v3.json').read_bytes())
for rel,digest in proposal['source_hashes'].items():
    assert sha(NAV/rel) == digest, rel
for path,digest in proposal['evidence_hashes'].items():
    assert sha(path) == digest, path
assert '37 passed' in json.loads((OUT/'verification.json').read_bytes())['tests_output']
assert json.loads((OUT/'final-review.json').read_bytes())['status'] == 'PASS_BASIC_REVIEW_ONLY'
assert 'VALID:' in (OUT/'plan-check.log').read_text()
assert not (OUT/'commit-result.json').exists()
paths = ['CaMRA/navigation/src/camra_navigation/envs/clean_v3.py',
    'CaMRA/navigation/scripts/nav067_simulation_v3.py',
    'CaMRA/navigation/scripts/evaluate_navigation_v4.py',
    'CaMRA/navigation/tests/test_clean_v3.py',
    'CaMRA/navigation/config/nav067_startup_recovery_proposal_v3.json',
    'CaMRA/navigation/reports/nav067_api_repair_preparation_v3.json',
    'CaMRA/navigation/docs/implementation_handoff.md',
    'doc/plan/20261109_camra_navigation_implementation.json']
evidence = ['verify.py','verify.log','verification.json','basic-tests.log','simulator-diff.txt',
    'prepare.py','prepare.log','production-unapproved-refusal.log','plan-check.log',
    'final-review.json','check_and_commit.py']
paths += [str((OUT/name).relative_to(ROOT)) for name in evidence]
staged = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
paths += [str((OUT/name).relative_to(ROOT)) for name in ('commit_final.py','check-and-commit.log')]
assert set(staged) == set(paths), staged
paths.append(str((OUT/'commit-final.log').relative_to(ROOT)))
subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
raw_suffixes = ('.log','.txt','.jsonl','.csv')
derived = 'CaMRA/navigation/scripts/nav067_simulation_v3.py'
strict = [p for p in paths if Path(p).suffix not in raw_suffixes and p != derived]
subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
    'diff','--cached','--check','--',*strict],cwd=ROOT,check=True)
# Preserve the already hashed derived simulator's one terminal blank line;
# this whitespace-only exception changes neither source identity nor behavior.
subprocess.run(['git','-c','core.whitespace=blank-at-eol,-blank-at-eof,space-before-tab,cr-at-eol',
    'diff','--cached','--check','--',derived],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','fix(nav): repair rigid-body readback and direct-loop reset initialization'],cwd=ROOT,check=True)
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
names = subprocess.check_output(['git','show','--format=','--name-only',head],cwd=ROOT,text=True).splitlines()
assert set(names) == set(paths)
result = {'commit':head,'pushed':False,'files':paths,'basic_tests':37,
    'new_physics_steps':0,'new_navigation_episodes':0,'NAV_067_status':False,
    'whitespace_exception':'one preserved terminal blank line in the already hashed derived simulator',
    'prior_failed_check_preserved':'check-and-commit.log'}
write_new(OUT/'commit-result.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
