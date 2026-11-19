"""Verify saved response-probe evidence without fitting or new rollouts."""
from datetime import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys

import numpy as np

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parents[1]


def equivalent(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: equivalent(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): equivalent(x,y)
    elif isinstance(a,float):
        assert math.isclose(a,b,rel_tol=1.e-10,abs_tol=1.e-10),(a,b)
    else:
        assert a==b,(a,b)


def main():
    if socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX'):
        raise ValueError('requires tiger11/tmux')
    spec = importlib.util.spec_from_file_location('response_probe',NAV/'scripts/nav043_motion_response.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    protocol = m.verify_protocol()
    approval = m.require_approval(protocol)
    output = Path(protocol['output_root'])
    assert (output/'FINISHED.json').is_file() and not (output/'HELD.json').exists()
    result = m.load(output/'FINISHED.json')
    launch = m.load(ROOT/'.worklog-testing/NAV-043/horizon90-v1/response-launch.json')
    exit_record = m.load(ROOT/'.worklog-testing/NAV-043/horizon90-v1/response-exit.json')
    assert exit_record['exit_code']==0 and not exit_record['retry_performed']
    assert result['protocol_sha256']==m.sha(m.PROTOCOL)
    assert result['approval']==approval
    assert result['cpu_wall_s']<=protocol['limits']['cpu_wall_s']
    assert len(result['solves'])==9 and [s['ordinal'] for s in result['solves']]==list(range(1,10))
    assert len(result['rows'])==18
    traces = [m.read_trace(t) for t in protocol['traces']]
    rows,normal_checks = [],[]
    for level in m.LEVELS:
        training = [t for t in traces if t.identity['level']!=level]
        heldout = [t for t in traces if t.identity['level']==level]
        model = m.load(output/level/'model.json')
        assert model['training_trials']==[t.identity['trial'] for t in training]
        assert model['heldout_trials']==[t.identity['trial'] for t in heldout]
        assert set(model['training_trials']).isdisjoint(model['heldout_trials'])
        assert model['deployment_model'] is False
        parameters = np.asarray(model['parameters_a_b'])
        for axis,name in enumerate(m.AXES):
            record = m.load(output/level/(name+'-solve-result.json'))
            equivalent(record,next(s for s in result['solves'] if s['level']==level and s['axis']==name))
            assert record['valid'] and record['rank']==2 and record['condition']<=1.e8
            assert 0.<=record['a']<1. and record['b']>0.
            np.testing.assert_array_equal(parameters[axis],[record['a'],record['b']])
            # Verify the existing least-squares solution by its normal residual;
            # do not call lstsq, solve, inverse, SVD, or any fitting routine.
            x = np.concatenate([np.column_stack([t.observed_twists[:-1,axis],
                  t.interval_commands[1:len(t.observed_twists),axis]]) for t in training])
            y = np.concatenate([t.observed_twists[1:,axis] for t in training])
            residual = x@parameters[axis]-y
            gradient = float(np.linalg.norm(x.T@residual,ord=np.inf))
            assert gradient<=1.e-8
            assert math.isclose(float(residual@residual),record['residual_sum'][0],rel_tol=1.e-9,abs_tol=1.e-10)
            normal_checks.append({'level':level,'axis':name,'normal_residual_inf':gradient})
        for trace in heldout:
            recomputed = m.compare_trace(trace,parameters)
            saved = m.load(output/level/f"{trace.identity['kind']}-{trace.identity['method']}.json")
            equivalent(recomputed,saved)
            equivalent(saved,next(r for r in result['rows'] if r['trial']==trace.identity['trial']))
            rows.append(recomputed)
    gate = m.grouped_gate(rows)
    equivalent(gate,result['gate'])
    m.verify_protocol()
    ideal,response = gate['all_heldout_ideal_RMSE_m'],gate['all_heldout_response_RMSE_m']
    report = {'status':'EXECUTION_AND_SAVED_REPLAY_AUDIT_PASS','at':datetime.now().astimezone().isoformat(),
              'host':socket.gethostname(),'pane':os.environ.get('TMUX_PANE'),'python':sys.executable,
              'launch':launch,'exit':exit_record,'limits':protocol['limits'],
              'actual_axis_solves':9,'extra_verification_fits':0,'actual_heldout_rows':18,
              'saved_measurement_ticks':sum(t.ticks for t in traces),
              'new_physical_episodes':0,'new_nominal_episodes':0,'W_queries':0,
              'actor_critic_updates':0,'calibration_fits':0,'cpu_wall_s':result['cpu_wall_s'],
              'protected_identities_checked':295,'raw_input_hashes_checked':90,
              'normal_equation_checks':normal_checks,'gate':gate,'relative_RMSE_reduction':1.-response/ideal,
              'largest_response_max_error_m':max(r['terminal']['response']['position_max_m'] for r in rows),
              'largest_response_terminal_error_m':max(r['terminal']['response']['position_terminal_m'] for r in rows),
              'solves':result['solves'],'rows':rows,
              'output_index':[{'path':str(p),'sha256':m.sha(p),'bytes':p.stat().st_size}
                              for p in sorted(output.rglob('*.json'))],
              'deployment_model_selected':False,'ninety_second_evidence':False,
              'NAV043_passed':False,'G5_passed':False}
    m.write_new(NAV/'reports/nav043_motion_response_review_v1.json',report)


if __name__=='__main__':
    main()
