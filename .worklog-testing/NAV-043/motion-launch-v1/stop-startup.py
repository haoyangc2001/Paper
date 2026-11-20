"""Stop this exact owned child on its saved startup loader errors; no retry."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT=Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_motion_probe_guard as G

assert os.environ.get('TMUX')
p,a,r=G.approved()
out=Path(p['output_root'])
process=G.load(out/'PROCESS.json')
pid=process['pid']
log=(out/'stdout.log').read_text(errors='replace')
assert 'Error: libXt.so.6: cannot open shared object file' in log
owner=G.load(out/'OWNER.json')
assert owner==G.load(G.EVIDENCE/'launch.lock')
live=False
try:
    args=(Path('/proc')/str(pid)/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    assert [v.decode() for v in args]==process['command']
    live=True
except FileNotFoundError:
    pass
record={'at':datetime.now().astimezone().isoformat(),'pane':os.environ.get('TMUX_PANE'),
    'pid':pid,'live_before_signal':live,'reason':'retained native loader startup errors: libXt.so.6 missing; no retry',
    'stdout_sha256_before_signal':hashlib.sha256((out/'stdout.log').read_bytes()).hexdigest(),
    'APP_READY_before_signal':(out/'APP_READY.json').exists(),
    'runtime_before_signal':(out/'runtime.json').exists(),
    'started_case_directories_before_signal':[f.name for f in out.glob('M*') if f.is_dir()],
    'signal_requested':'SIGTERM' if live else None,
    'new_applications':0,'no_automatic_retry':True,'NAV043_passed':False,'G5_passed':False}
G.persist(G.EVIDENCE/'startup-stop-request.json',record)
if live:
    os.kill(pid,signal.SIGTERM)
print(json.dumps(record,ensure_ascii=False),flush=True)
