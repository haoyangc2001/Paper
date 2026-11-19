import json
import os
from pathlib import Path
import subprocess

out=Path('/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1')
launch=json.loads((out/'launch-L1_empty.json').read_bytes())
pid=launch['pid']
try:
    os.kill(pid,0)
    alive=True
except ProcessLookupError:
    alive=False
prefix=Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib/python3.11/site-packages/isaacsim')
cache=[]
for parent in [prefix/'extscache',prefix/'kit/exts',prefix/'kit/data/Kit/Isaac-Sim/5.1/exts/3']:
    for item in parent.glob('*usd.mdl*'):
        library=item/'bin/librtx.usdmdl.plugin.so'
        cache.append({'path':str(item),'library_exists':library.exists(),
                      'python_module_exists':(item/'omni/UsdMdl').exists()})
kit=prefix/'kit/logs/Kit/Isaac-Sim/5.1/kit_20261119_081036.log'
result={'pid':pid,'alive':alive,
    'ps':subprocess.run(['ps','-p',str(pid),'-o','pid,ppid,stat,etime,pcpu,pmem,args'],capture_output=True,text=True).stdout,
    'wait_channel':Path(f'/proc/{pid}/wchan').read_text() if alive else None,
    'actual_tmux_pane':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],'-p','#{session_name}:#{window_name}'],text=True).strip(),
    'extension_paths':cache,
    'kit_log_tail':'\n'.join(kit.read_text(errors='replace').splitlines()[-24:]) if kit.exists() else None,
    'runtime_written':(out/'L1_empty/runtime.json').exists(),
    'terminal_marker':json.loads((out/'exit-L1_empty.json').read_bytes()) if (out/'exit-L1_empty.json').exists() else None}
print(json.dumps(result,ensure_ascii=False,indent=2))
