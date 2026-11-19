"""Prepared startup repair for NAV-067; requires an explicit pinned amendment.

Original pilot manifest and v2 implementation stay immutable. The sealed simulator
loop is copied with explicit rigid-body API and direct-reset compatibility fixes.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, LEVELS, read_manifest, sha, write_new
from nav067_runtime import bind_local_locomotion, guarded_application

AMENDMENT = NAV/'config/nav067_startup_recovery_v3.json'


def recovery_context(level):
    manifest, prereg, nets = read_manifest()
    if not AMENDMENT.is_file():
        raise ValueError('explicit startup recovery amendment is absent')
    amendment = json.loads(AMENDMENT.read_bytes())
    pin = Path(str(AMENDMENT)+'.sha256')
    if not pin.is_file() or sha(AMENDMENT)!=pin.read_text().strip():
        raise ValueError('startup recovery amendment pin mismatch')
    if amendment['status']!='APPROVED':
        raise ValueError('startup recovery is not authorized')
    if amendment['original_manifest_sha256']!=sha(OUTPUT/'manifest.json'):
        raise ValueError('original pilot manifest changed')
    expected_limits={'total_navigation_episodes':6,'horizon_s':45.,'warmup_s':1.,
                     'training_updates':0,'W_queries':0,'automatic_retries':0}
    if amendment['limits']!=expected_limits:
        raise ValueError('recovery would change the original pilot quota/objective')
    for rel,digest in amendment['source_hashes'].items():
        if sha(NAV/rel)!=digest:
            raise ValueError('recovery implementation changed: '+rel)
    for path,digest in amendment['evidence_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('recovery evidence changed: '+path)
    slot=next(s for s in amendment['slots'] if s['level']==level)
    out=OUTPUT/slot['relative_output']
    if out.resolve().parent not in (OUTPUT.resolve(),(OUTPUT/'L1_empty').resolve()):
        raise ValueError('unexpected recovery output path')
    return manifest,prereg,nets,amendment,out


def run(level,device):
    manifest,prereg,nets,amendment,out=recovery_context(level)
    origin=bind_local_locomotion(NAV)
    # An already existing attempt cannot be retried, even after an error.
    out.mkdir(exist_ok=False)
    row=next(r for r in manifest['maps'] if r['level']==level)
    write_new(out/'STARTED.json',{'level':level,'started_at':datetime.now().astimezone().isoformat(),
        'pid':os.getpid(),'host':socket.getfqdn(),'cwd':str(Path.cwd()),'python':sys.executable,
        'command':sys.argv,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],
            '-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
        'binding':manifest['binding'],'locomotion_import':origin,
        'amendment_sha256':sha(AMENDMENT),'original_slot_count':2})
    def factory():
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(headless=True,device=device)
        return launcher.app
    def work():
        from nav067_simulation_v3 import simulate
        return simulate(row,out,manifest,prereg,nets,device)
    return guarded_application(out,factory,work)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',required=True,choices=LEVELS)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    raise SystemExit(run(args.level,args.device))
