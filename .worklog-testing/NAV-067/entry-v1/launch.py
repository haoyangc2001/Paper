from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from nav067_sources import NAV, ROOT, OUTPUT, LEVELS, read_manifest, sha, write_new

level=sys.argv[1]
assert level in LEVELS
manifest,_,_=read_manifest()
gpu=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu.rsplit(',',1)[1].strip())>=12000, gpu
assert not (OUTPUT/level).exists(), 'Existing attempt must be audited, not restarted'
row=next(m for m in manifest['maps'] if m['level']==level)
reg=json.loads((NAV/'config/nav066_decision_prereg_v2.json').read_bytes())
paths={**reg['protected_hashes'],**{str(NAV/k):v for k,v in reg['implementation_hashes'].items()}}
env=os.environ.copy()
env.update({'CUDA_VISIBLE_DEVICES':'6','OMNI_KIT_ACCEPT_EULA':'YES',
    'TMPDIR':'/pub/data/caohy/tmp','XDG_DATA_HOME':'/pub/data/caohy/xdgdata',
    'XDG_CACHE_HOME':'/pub/data/caohy/xdgcache','XDG_CONFIG_HOME':'/pub/data/caohy/xdgconfig',
    'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
    'PYTHONPATH':':'.join(str(p) for p in (NAV/'src',NAV/'scripts',NAV.parent/'calibration/src')),
    'LD_LIBRARY_PATH':'/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')})
python='/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
cmd=[python,'-B',str(NAV/'scripts/evaluate_navigation_v2.py'),'--level',level,'--device','cuda:0']
log=OUTPUT/('launch-'+level+'.log')
record={'run_id':'nav067_clean_pilot_v1_'+level,'attempt_id':'attempt1','host':'192.168.20.11',
    'hostname':socket.getfqdn(),
    'tmux_session':subprocess.check_output(['tmux','display-message','-p','#{session_name}'],text=True).strip(),
    'tmux_window':subprocess.check_output(['tmux','display-message','-p','#{window_name}'],text=True).strip(),
    'pid':None,'cwd':str(NAV),'python_absolute_path':python,'gpu_identity':gpu,
    'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'dirty_diff_identity':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
    'controller_id':manifest['binding']['controller']['id'],
    'controller_manifest_hash':sha(ROOT/'CaMRA/locomotion/docs/frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json'),
    'asset_hashes':{p:h for p,h in paths.items() if p.endswith(('.usd','.urdf','.stl','.dae'))},
    'calibration_hashes':{p:h for p,h in paths.items() if '/calibration/config/' in p},
    'contract_hash':{p:h for p,h in paths.items() if '/navigation/config/' in p},
    'map_split_hash':sha(OUTPUT/'manifest.json'),'prereg_hash':manifest['binding'],
    'full_non_sensitive_command':cmd,'recorded_environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH','OMP_NUM_THREADS','OMNI_KIT_ACCEPT_EULA')},
    'started_at':datetime.now().astimezone().isoformat(),'log_paths':[str(log)],
    'resume_state':'new exclusive two-slot level attempt; never auto-resume/restart',
    'pilot_map':row['map_id'],'launcher_source_sha256':sha(__file__)}
with log.open('x') as stream:
    process=subprocess.Popen(cmd,cwd=NAV,env=env,stdout=stream,stderr=subprocess.STDOUT)
    record['pid']=process.pid
    write_new(OUTPUT/('launch-'+level+'.json'),record)
    code=process.wait()
write_new(OUTPUT/('exit-'+level+'.json'),{'pid':process.pid,'returncode':code,'ended_at':datetime.now().astimezone().isoformat()})
raise SystemExit(code)
