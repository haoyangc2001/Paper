"""Release the already-failed C01 application; never launch or retry a slot."""
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
from nav067_sources import load, write_new, sha
out=Path('/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1/C01')
error=load(out/'ERROR.json')
assert error['message']=='actual clone origins are not separated'
pid=load(out/'PROCESS.json')['pid']
record={'at':datetime.now().astimezone().isoformat(),'pid':pid,'error_sha256':sha(out/'ERROR.json'),
    'action':'Terminate already-failed child during application cleanup; do not restart or proceed.',
    'signals':[],'no_retry':True,'new_applications':0,'python':sys.executable,'tmux':os.environ.get('TMUX')}
process=Path('/proc')/str(pid)
if process.exists():
    cmd=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'run_nav067_remaining_validation.py --child C01' in cmd
    assert os.getpgid(pid)==pid
    record['observed_cmdline']=cmd
    os.killpg(pid,signal.SIGTERM)
    record['signals'].append('SIGTERM')
    time.sleep(3)
    if process.exists():
        state=(process/'stat').read_text().split(') ',1)[1].split()[0]
        if state!='Z':
            os.killpg(pid,signal.SIGKILL)
            record['signals'].append('SIGKILL')
record['still_present']=process.exists()
source=importlib.util.find_spec('isaaclab')
record['actual_isaaclab_origin']=source.origin
record['actual_isaaclab_search_locations']=list(source.submodule_search_locations)
for location in source.submodule_search_locations:
    path=Path(location)/'terrains/terrain_importer.py'
    if path.is_file():
        record['actual_terrain_importer']={'path':str(path),'sha256':sha(path)}
        (HERE/'actual-terrain-importer-source.txt').write_bytes(path.read_bytes())
write_new(HERE/'failed-child-stop.json',record)
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
