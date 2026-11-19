"""Check final integration additions and freeze the exact approved harness."""
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAV=ROOT/'CaMRA/navigation'
sys.path.insert(0,str(NAV/'scripts'))
from nav067_sources import E, load, sha, write_new, read_manifest
from nav067_remaining_audit import audit_episode
from nav067_collider_readback import TOLERANCES
from run_nav067_remaining_validation import SLOTS, slot_bounds, EXPECTED_ROOT, APPROVAL, EXECUTION

assert os.environ['TMUX_PANE']=='%600' and os.environ['CUDA_VISIBLE_DEVICES']=='6'
assert not EXPECTED_ROOT.exists()
basic=load(HERE/'verification.json')
assert basic['status']=='MAIN_AGENT_BASIC_CPU_PASS_NOT_PHYSICAL_ACCEPTANCE'
assert basic['source_hashes']['CaMRA/navigation/src/camra_navigation/envs/controlled_cases_v1.py']==sha(NAV/'src/camra_navigation/envs/controlled_cases_v1.py')
manifest,prereg,nets=read_manifest()
proposal=load(NAV/'config/nav067_remaining_validation_proposal_v1.json')
for source,digest in proposal['sealed_sources'].items():
    assert sha(ROOT/source)==digest
sources=['CaMRA/navigation/scripts/nav067_collider_readback.py',
 'CaMRA/navigation/scripts/nav067_controlled_simulation.py','CaMRA/navigation/scripts/run_nav067_remaining_validation.py',
 'CaMRA/navigation/scripts/nav067_remaining_audit.py','CaMRA/navigation/src/camra_navigation/envs/controlled_cases_v1.py',
 'CaMRA/navigation/scripts/nav067_runtime.py','CaMRA/navigation/scripts/nav067_sources.py']
for name in sources:
    ast.parse((ROOT/name).read_text())
command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(NAV/'tests/test_physical_v2.py'),str(NAV/'tests/test_clean_v3.py')]
environment=os.environ.copy()
environment['PYTHONPATH']=':'.join(str(p) for p in (NAV/'src',NAV/'scripts',NAV.parent/'calibration/src'))
result=subprocess.run(command,env=environment,cwd=ROOT,capture_output=True,text=True)
(HERE/'pytest-final.log').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
audits=[]
for row in proposal['repetitions']['slots']:
    base=Path(proposal['inputs']['manifest']['path']).parent/row['baseline_relative_directory']
    for method in ('uncalibrated','full'):
        audits.append(audit_episode(base/method))
write_new(HERE/'baseline-final-audit.json',audits)
gpu=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
assert int(gpu.rsplit(',',1)[1])>=12000,gpu
report={'status':'FINAL_BASIC_PASS_NOT_PHYSICAL_ACCEPTANCE','at':datetime.now().astimezone().isoformat(),
    'python':sys.executable,'tmux':os.environ['TMUX'],'pane':os.environ['TMUX_PANE'],'gpu':gpu,
    'prior_CPU_checks':basic['count'],'pytest_command':command,'pytest_exit':result.returncode,
    'baseline_episodes_audited':len(audits),'source_hashes':{p:sha(ROOT/p) for p in sources},
    'new_applications':0,'new_physical_ticks':0,'training_updates':0,'W_queries':0,
    'preparation_corrections':['GPU0 only 6429MiB free; physical GPU6 fixed before sealing.',
        'Set actual zero commands on production monitor so idle model cost is charged.',
        'Use full MapSpec.from_dict for island serialization.',
        'Isolated controlled flat terrain spans separated clone origins; repetition terrain is unchanged.',
        'Reserve 20 seconds for process cleanup within the cumulative four-hour cap.']}
write_new(HERE/'final-verification.json',report)
approval=load(APPROVAL)
def patch_new(path,value):
    assert not path.exists(),path
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    patch='*** Begin Patch\n*** Add File: '+str(path)+'\n'+''.join('+'+line+'\n' for line in text.splitlines())+'*** End Patch\n'
    subprocess.run(['apply_patch'],input=patch,text=True,check=True,cwd=ROOT)
pin=Path(str(APPROVAL)+'.sha256')
assert not pin.exists()
subprocess.run(['apply_patch'],input='*** Begin Patch\n*** Add File: '+str(pin)+'\n+'+sha(APPROVAL)+'\n*** End Patch\n',text=True,check=True,cwd=ROOT)
fixtures=load(HERE/'fixtures.json')
evidence={str((HERE/'final-verification.json').relative_to(ROOT)):sha(HERE/'final-verification.json'),
    str((HERE/'verification.json').relative_to(ROOT)):sha(HERE/'verification.json'),
    str((HERE/'fixtures.json').relative_to(ROOT)):sha(HERE/'fixtures.json'),
    str((HERE/'baseline-final-audit.json').relative_to(ROOT)):sha(HERE/'baseline-final-audit.json'),
    str((HERE/'pytest-final.log').relative_to(ROOT)):sha(HERE/'pytest-final.log')}
execution={'schema':'camra-nav067-remaining-validation-execution-v1','task':'NAV-067',
    'approval_sha256':sha(APPROVAL),'proposal_sha256':sha(ROOT/approval['approved_proposal']['path']),
    'output_root':str(EXPECTED_ROOT),'slots':list(SLOTS),'limits':approval['limits'],
    'physical_gpu_index':6,'physical_gpu_uuid':gpu.split(',')[1].strip(),
    'logical_device':'cuda:0','CUDA_VISIBLE_DEVICES':'6','python':sys.executable,
    'fixtures':fixtures,'numeric_tolerances':TOLERANCES,'source_hashes':report['source_hashes'],
    'evidence_hashes':evidence,'resource_accounting':'Per-case exact cap, per-map two 46-second episodes; all starts claimed once before AppLauncher; no transfers.',
    'fixed_environment':{'CUDA_VISIBLE_DEVICES':'6','OMNI_KIT_ACCEPT_EULA':'YES','TMPDIR':'/pub/data/caohy/tmp',
      'XDG_DATA_HOME':'/pub/data/caohy/xdgdata','XDG_CACHE_HOME':'/pub/data/caohy/xdgcache',
      'XDG_CONFIG_HOME':'/pub/data/caohy/xdgconfig','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
      'PYTHONPATH':environment['PYTHONPATH'],'LD_LIBRARY_PATH':'/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib'},
    'command':[sys.executable,'-B',str(NAV/'scripts/run_nav067_remaining_validation.py')],
    'failure_policy':approval['stop_policy'],'performance_evidence':False,
    'native_contact_disclosure':'C06 base-to-nav_box_4 native filtered contact; C01-C05 deliberately pose-injected actual readbacks.',
    'repeatability_disclosure':'Same simulator seed100 and original selected model/map/budget/reset/stepping. Passive readback only; report two runs per configuration without reliability claims.'}
assert sum(slot_bounds(execution,s) for s in SLOTS)<=60000
execution['reserved_environment_ticks_max']=sum(slot_bounds(execution,s) for s in SLOTS)
patch_new(EXECUTION,execution)
epin=Path(str(EXECUTION)+'.sha256')
subprocess.run(['apply_patch'],input='*** Begin Patch\n*** Add File: '+str(epin)+'\n+'+sha(EXECUTION)+'\n*** End Patch\n',text=True,check=True,cwd=ROOT)
print(json.dumps({'status':'FINAL_BASIC_AND_SEAL_PASS','execution_sha256':sha(EXECUTION),
    'reserved_environment_ticks_max':execution['reserved_environment_ticks_max'],'new_applications':0}),flush=True)
