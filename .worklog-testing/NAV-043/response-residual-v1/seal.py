"""Seal proposed scope and verify actual missing-approval refusal; no real fit."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/caohy/repositories/Paper')
HERE = ROOT/'.worklog-testing/NAV-043/response-residual-v1'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_affine_response as m

assert socket.gethostname().split('.')[0]=='tiger11' and os.environ.get('TMUX')
p = m.base.load(m.PROPOSAL)
assert p['limits']==m.LIMITS and p['status']=='PROPOSED'
for path,digest in p['source_hashes'].items():
    assert m.base.sha(ROOT/path)==digest,path
assert p['trace_protocol_sha256']==m.base.sha(m.base.PROTOCOL)
assert not m.APPROVAL.exists() and not Path(p['output_root']).exists()
pin = Path(str(m.PROPOSAL)+'.sha256')
with pin.open('x') as stream:
    stream.write(m.base.sha(m.PROPOSAL)+'\n')
process = subprocess.run([sys.executable,'-B',str(ROOT/'CaMRA/navigation/scripts/nav043_affine_response.py'),'--run'],
                         cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
with (HERE/'missing-approval-refusal.log').open('x') as stream:
    stream.write(process.stdout)
assert process.returncode!=0 and 'require separate approved registration' in process.stdout
assert not m.APPROVAL.exists() and not Path(p['output_root']).exists()
report = {'at':datetime.now().astimezone().isoformat(),'host':socket.gethostname(),
    'pane':os.environ['TMUX_PANE'],'python':sys.executable,'proposal_sha256':m.base.sha(m.PROPOSAL),
    'source_hashes':p['source_hashes'],'actual_missing_approval_exit':process.returncode,
    'actual_refusal_before_output_creation':True,'approval_exists':False,'formal_output_exists':False,
    'real_data_solves':0,'synthetic_axis_solves':10,'unique_new_basic_checks':19,
    'new_physical_episodes':0,'new_nominal_episodes':0,'W_queries':0,'network_updates':0,
    'new_calibration_fits':0,'NAV043_passed':False,'G5_passed':False}
with (HERE/'sealed-preparation.json').open('x') as stream:
    json.dump(report,stream,indent=2); stream.write('\n')
