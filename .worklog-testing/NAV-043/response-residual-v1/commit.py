"""Commit only this review's authorized paths, after completed checks."""
from pathlib import Path
import json
import os
import socket
import subprocess

ROOT = Path('/home/caohy/repositories/Paper')
assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
checks = json.loads((ROOT/'.worklog-testing/NAV-043/response-residual-v1/close.json').read_bytes())
assert checks['protected_checked']==295 and checks['historical_task_definitions_records_and_statuses_unchanged']
paths = ['AGENTS.md','CaMRA/navigation/README.md','CaMRA/navigation/docs/implementation_handoff.md',
    'CaMRA/navigation/docs/nav043_response_model_v2_plan_20261120.md',
    'doc/plan/20261109_camra_navigation_implementation.json',
    'CaMRA/navigation/scripts/analyze_nav043_response_residuals.py',
    'CaMRA/navigation/scripts/nav043_affine_response.py',
    'CaMRA/navigation/tests/test_nav043_response_residuals.py',
    'CaMRA/navigation/tests/test_nav043_affine_response.py',
    'CaMRA/navigation/config/nav043_affine_response_proposal_v2.json',
    'CaMRA/navigation/config/nav043_affine_response_proposal_v2.json.sha256',
    'CaMRA/navigation/reports/nav043_response_residuals_v1.json',
    '.worklog-testing/NAV-043/response-residual-v1']
def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True)
staged = git('diff','--cached','--name-only').splitlines()
assert not staged,'preexisting staged files; do not include them'
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
staged = git('diff','--cached','--name-only').splitlines()
assert all(any(p==a or p.startswith(a+'/') for a in paths) for p in staged)
assert all(not (p.startswith(('manuscripts/','RC-PPO/','HiRA/')) or p.endswith(('.tex','.bib'))) for p in staged)
subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
print(git('diff','--cached','--stat'),flush=True)
subprocess.run(['git','commit','-m','feat(nav): diagnose saved yaw residuals and seal affine response proposal'],cwd=ROOT,check=True)
print(git('rev-parse','HEAD'),flush=True)
