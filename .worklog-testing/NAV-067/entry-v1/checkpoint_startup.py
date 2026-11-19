"""Preserve the completed startup attempt; never launch or release pilot slots."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from nav067_sources import NAV, ROOT, OUTPUT, read_manifest, sha, write_new

assert socket.gethostname().split('.')[0] == 'tiger11'
assert os.environ.get('TMUX')
manifest, _, _ = read_manifest()
out = ROOT / '.worklog-testing/NAV-067/entry-v1/startup-attempt1'
out.mkdir(exist_ok=False)
launch = json.loads((OUTPUT/'launch-L1_empty.json').read_bytes())
exit_record = json.loads((OUTPUT/'exit-L1_empty.json').read_bytes())
assert launch['pid'] == exit_record['pid'] == 1232883
assert launch['git_head'] == '22a23ccbe552b5d80f171da3590bcd849ae886ed'
try:
    os.kill(launch['pid'], 0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError('Original pilot is still alive; inspect it before recording a terminal state')
copies = {}
for rel in ('launch-L1_empty.json', 'exit-L1_empty.json', 'launch-L1_empty.log',
            'L1_empty/STARTED.json'):
    source = OUTPUT/rel
    target = out/rel.replace('/', '-')
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
    assert sha(source) == sha(target)
    copies[str(source)] = {'copy':str(target.relative_to(ROOT)), 'sha256':sha(source)}
assert not (OUTPUT/'L1_empty/runtime.json').exists()
assert not (OUTPUT/'L1_empty/FINISHED.json').exists()
assert not list((OUTPUT/'L1_empty').glob('*/result.json'))
assert not (OUTPUT/'L2_single_detour').exists()
assert not (OUTPUT/'L3_multi_path').exists()
pane = subprocess.check_output(['tmux','display-message','-t','%587','-p',
    '#{session_name}:#{window_name} #{window_id} #{pane_id} #{pane_pid}'],text=True).strip()
assert 'nav067-clean-v2 @587 %587 1127901' in pane
captured = subprocess.check_output(['tmux','capture-pane','-p','-t','%587','-S','-80'])
with (out/'runner-pane.txt').open('xb') as stream:
    stream.write(captured)
kit = Path('/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261119_081036.log')
with (out/'kit-tail.txt').open('x') as stream:
    stream.write('\n'.join(kit.read_text(errors='replace').splitlines()[-100:])+'\n')
plan = json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
base = json.loads(subprocess.check_output(['git','show','22a23cc:doc/plan/20261109_camra_navigation_implementation.json'],cwd=ROOT))
assert len(plan['tasks']) == len(base['tasks']) == 67
for old, new in zip(base['tasks'], plan['tasks'], strict=True):
    if old['id'] != 'NAV-067':
        assert old == new
    else:
        original_records = old['work_records']
        assert new['work_records'][:len(original_records)] == original_records
        assert len(new['work_records']) == len(original_records)+1
        assert {k:v for k,v in old.items() if k!='work_records'} == {k:v for k,v in new.items() if k!='work_records'}
assert sum(t['status'] for t in plan['tasks']) == 54
summary = {
    'task':'NAV-067', 'status':'HELD_STARTUP_NO_VALID_TRACE',
    'timestamp':datetime.now().astimezone().isoformat(),
    'launch_attempts':1, 'completed_physical_episodes':0,
    'reserved_episode_slots':2, 'unstarted_episode_slots':4,
    'reserved_slots_automatically_released':False,
    'original_exit_record':exit_record,
    'exit_zero_is_acceptance':False,
    'missing_outputs':['runtime.json','episode result.json','substeps.json','FINISHED.json'],
    'launch_metadata_correction':{
        'original_record_retained':True,
        'original_window_field':launch['tmux_window'],
        'authoritative_runner_pane':pane,
        'cause':'Launcher queried tmux display-message without an explicit pane target.',
        'evidence':str((out/'runner-pane.txt').relative_to(ROOT))},
    'observed_errors':['Isaac material extension native library absent.',
                       "ModuleNotFoundError: No module named 'omni.UsdMdl'."],
    'root_cause_of_missing_runtime_trace':'Not established; Kit reached app ready and then closed. No policy-performance conclusion is supported.',
    'kit_log':{'path':str(kit),'sha256':sha(kit),'tail_copy':str((out/'kit-tail.txt').relative_to(ROOT))},
    'preserved_artifacts':copies,
    'binding':manifest['binding'], 'manifest_sha256':sha(OUTPUT/'manifest.json'),
    'source_or_model_changes':False, 'new_training_updates':0, 'new_W_queries':0,
    'plan_prefix_and_status_preserved':True,
    'next':'Diagnose the startup/entry failure before any recovery attempt; retain the two claimed slots. No automatic pilot rerun, dependency replacement or NAV-043 execution.'}
write_new(out/'verification.json', summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
