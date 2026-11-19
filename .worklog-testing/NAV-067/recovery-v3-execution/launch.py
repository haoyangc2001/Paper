"""Run each authorized slot once; stop the whole sequence on infrastructure error."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np

from nav067_sources import NAV, ROOT, OUTPUT, LEVELS, read_manifest, sha, write_new
from evaluate_navigation_v4 import AMENDMENT, recovery_context

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def audit(level, out):
    assert not (out/'ERROR.json').exists(), (out/'ERROR.json').read_text()
    finished = json.loads((out/'FINISHED.json').read_bytes())
    runtime = json.loads((out/'runtime.json').read_bytes())
    assert runtime['physics_dt_s'] == .005 and runtime['decimation'] == 4
    assert runtime['ground_material_usd_readback'] and runtime['colliders']
    results = []
    for method in ('uncalibrated','full'):
        trial = out/method
        result = json.loads((trial/'result.json').read_bytes())
        steps = json.loads((trial/'substeps.json').read_bytes())
        commands = json.loads((trial/'commands.json').read_bytes())
        navigation = json.loads((trial/'navigation.json').read_bytes())
        energy = np.load(trial/'energy.npz',allow_pickle=False)
        n = result['physics_steps']
        assert n > 0 and n == len(steps)
        assert [r['physics_tick'] for r in steps] == list(range(1,n+1))
        np.testing.assert_allclose([r['t_s'] for r in steps],np.arange(1,n+1)*.005,atol=1e-8,rtol=0.)
        assert steps[-1] == result['terminal']
        assert steps[-1]['label'] == result['label'] and all(r['label'] is None for r in steps[:-1])
        assert commands[0]['tick_before'] == 200
        assert result['warmup']['physics_steps'] == 200 and result['warmup']['policy_calls'] == 50
        assert result['reset']['previous_action_max'] == 0.
        assert len(commands) == (n+3)//4 and len(navigation) == (len(commands)+4)//5
        assert sum(c['substeps'] for c in commands) == n
        for j,c in enumerate(commands):
            assert 1 <= c['substeps'] <= 4
            assert c['tick_after']-c['tick_before'] == c['substeps']
            if j: assert commands[j-1]['tick_after'] == c['tick_before']
            np.testing.assert_allclose(c['command_written'],c['command_seen'],atol=1e-6,rtol=0.)
            assert c['max_error'] <= 1e-6
            assert abs(c['sim_time_after']-c['sim_time_before']-c['substeps']*.005) <= 1e-7
        tau = energy['tau_applied'].astype(np.float64)
        vel = energy['qdot_post'].astype(np.float64)
        valid = energy['valid_mask'].astype(bool)
        assert tau.shape[0] == n and valid.shape == (n,1) and valid.all()
        actual = float((np.maximum(tau*vel,0.).sum(axis=-1)*.005*valid).sum())
        assert abs(actual-result['measured_joint_work_J']) <= 1e-9*max(1.,actual)
        assert abs(result['t_s']-n*.005) < 1e-8
        assert abs(result['z0_J']-steps[-1]['z_J']-result['internal_model_cost_J']) < 1e-8
        assert max(r['safety_y'] for r in steps) == result['safety_y']
        assert not result['performance_evidence']
        results.append({'method':method,'label':result['label'],'t_s':result['t_s'],
            'physics_steps':n,'policy_steps':len(commands),'navigation_steps':len(navigation),
            'safety_y':result['safety_y'],'z0_J':result['z0_J'],'z_J_end':result['z_J_end'],
            'measured_joint_work_J':actual,'internal_safe_and_budget_feasible':result['internal_safe_and_budget_feasible'],
            'physical_BVR':result['physical_BVR'],'last_policy_substeps':commands[-1]['substeps']})
    assert len(finished['episodes']) == 2
    return {'status':'PASS_INTEGRATION_TRACE_AUDIT_ONLY','level':level,'at':now(),
        'out':str(out),'episodes':results,'runtime_sha256':sha(out/'runtime.json'),
        'not_completed':['controlled actual counterexamples','same-configuration repeatability','NAV-043 physical applicability']}


def main():
    assert socket.gethostname().split('.')[0] == 'tiger11'
    assert os.environ.get('TMUX_PANE') == '%587'
    assert not (HERE/'COORDINATOR_STARTED.json').exists()
    preflight = json.loads((HERE/'preflight.json').read_bytes())
    assert preflight['amendment_sha256'] == sha(AMENDMENT)
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=ROOT,text=True).strip()
    write_new(HERE/'COORDINATOR_STARTED.json',{'at':now(),'pid':os.getpid(),'host':socket.getfqdn(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'amendment_sha256':sha(AMENDMENT),'sequence':list(LEVELS)})
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'6','OMNI_KIT_ACCEPT_EULA':'YES',
        'TMPDIR':'/pub/data/caohy/tmp','XDG_DATA_HOME':'/pub/data/caohy/xdgdata',
        'XDG_CACHE_HOME':'/pub/data/caohy/xdgcache','XDG_CONFIG_HOME':'/pub/data/caohy/xdgconfig',
        'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
        'PYTHONPATH':':'.join(str(p) for p in (NAV/'src',NAV/'scripts',NAV.parent/'calibration/src')),
        'LD_LIBRARY_PATH':'/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')})
    all_results = []
    for level in LEVELS:
        try:
            manifest, _, _, amendment, out = recovery_context(level)
            assert not out.exists()
            gpu = subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader,nounits'],text=True).strip()
            assert int(gpu.rsplit(',',1)[1].strip()) >= 12000, gpu
            command = [sys.executable,'-B',str(NAV/'scripts/evaluate_navigation_v4.py'),'--level',level,'--device','cuda:0']
            log = HERE/(level+'.log')
            with log.open('x') as stream:
                process = subprocess.Popen(command,cwd=NAV,env=env,stdout=stream,stderr=subprocess.STDOUT)
                write_new(HERE/('launch-'+level+'.json'),{'at':now(),'pid':process.pid,'level':level,
                    'host':socket.getfqdn(),'tmux':subprocess.check_output(['tmux','display-message','-t',os.environ['TMUX_PANE'],'-p','#{session_name}:#{window_name} #{window_id} #{pane_id}'],text=True).strip(),
                    'cwd':str(NAV),'python':sys.executable,'gpu':gpu,'command':command,
                    'amendment_sha256':sha(AMENDMENT),'manifest_sha256':sha(OUTPUT/'manifest.json'),
                    'binding':manifest['binding'],'log':str(log),'output':str(out),'launcher_sha256':sha(__file__)})
                print('LAUNCHED',level,process.pid,flush=True)
                code = process.wait()
            write_new(HERE/('exit-'+level+'.json'),{'at':now(),'pid':process.pid,'exit_code':code})
            assert code == 0, 'entry returned '+str(code)
            report = audit(level,out)
            write_new(HERE/('audit-'+level+'.json'),report)
            all_results.extend(report['episodes'])
            print('AUDITED',level,json.dumps(report['episodes'],ensure_ascii=False),flush=True)
        except BaseException:
            detail = traceback.format_exc()
            write_new(HERE/'HELD.json',{'at':now(),'level':level,'traceback':detail,
                'completed_episodes':len(all_results),'automatic_retries':0,'next_levels_not_started':True})
            print(detail,file=sys.stderr,flush=True)
            return 1
    read_manifest()  # unchanged protected input identities after all runs
    write_new(HERE/'FINISHED.json',{'at':now(),'episodes':all_results,'new_episodes':len(all_results),
        'training_updates':0,'W_queries':0,'automatic_retries':0,'NAV_067_complete':False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
