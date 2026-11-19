"""Proposed affine response diagnostic. Real fits require separate approval."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np

import nav043_motion_response as base

PROPOSAL = base.NAV/'config/nav043_affine_response_proposal_v2.json'
APPROVAL = base.NAV/'config/nav043_affine_response_approval_v2.json'
LIMITS = dict(models=3,axis_solves=9,cpu_wall_s=3600,new_physical_episodes=0,
              new_nominal_episodes=0,W_queries=0,actor_critic_updates=0,calibration_fits=0)


def approval():
    # Stop before reading trajectory data or performing any formal solve.
    if not PROPOSAL.is_file() or not APPROVAL.is_file():
        raise ValueError('new affine response fits require separate approved registration')
    for path in (PROPOSAL,APPROVAL):
        if base.sha(path)!=Path(str(path)+'.sha256').read_text().strip():
            raise ValueError('changed proposal or approval pin')
    p,a = base.load(PROPOSAL),base.load(APPROVAL)
    if (p['status']!='PROPOSED' or p['schema']!='nav043-affine-response-v2'
            or p['limits']!=LIMITS or a.get('status')!='APPROVED'
            or a.get('proposal_sha256')!=base.sha(PROPOSAL)
            or a.get('limits')!=LIMITS or a.get('output_root')!=p['output_root']):
        raise ValueError('approval does not authorize this exact finite diagnostic')
    for path,digest in p['source_hashes'].items():
        if base.sha(path)!=digest:
            raise ValueError('changed affine diagnostic source')
    return p,a


def fit_axis(x,y):
    theta,residuals,rank,singular = np.linalg.lstsq(x,y,rcond=1.e-12)
    condition = float(singular[0]/singular[-1]) if len(singular)==3 and singular[-1]>0. else None
    valid = (rank==3 and condition is not None and condition<=1.e8 and np.isfinite(theta).all()
             and 0.<=theta[0]<1. and theta[1]>0.)
    return {'a':float(theta[0]),'b':float(theta[1]),'d':float(theta[2]),'rank':int(rank),
            'singular_values':singular.tolist(),'condition':condition,'valid':bool(valid),
            'residual_sum':residuals.tolist()}


def free_replay(trace,parameters):
    """Same zero response initialization as v1, with no truth corrections."""
    pose,response = trace.initial.copy(),np.zeros(3)
    poses,responses = [],[]
    for i,command in enumerate(trace.interval_commands):
        response = parameters[:,0]*response+parameters[:,1]*command+parameters[:,2]
        responses.append(response.copy())
        for _ in range(min(base.NAV_TICKS,trace.ticks-i*base.NAV_TICKS)):
            pose = base.step(pose,response,base.PHYSICS_DT)
            poses.append(pose.copy())
    return np.asarray(poses),np.asarray(responses)


def gate(rows,old):
    def rmse(items):
        return math.sqrt(sum(r['metrics']['position_squared_error_sum'] for r in items)
                         /sum(r['metrics']['ticks'] for r in items))
    old_total = old['gate']['all_heldout_response_RMSE_m']
    new_total = rmse(rows)
    levels = {level:{'new_RMSE_m':rmse([r for r in rows if r['level']==level]),
        'old_RMSE_m':math.sqrt(sum(r['terminal']['response']['position_squared_error_sum']
            for r in old['rows'] if r['level']==level)/sum(r['terminal']['response']['ticks']
            for r in old['rows'] if r['level']==level))} for level in base.LEVELS}
    old_yaw = float(np.mean([abs(r['terminal']['response']['heading_terminal_rad']) for r in old['rows']]))
    new_yaw = float(np.mean([abs(r['metrics']['heading_terminal_rad']) for r in rows]))
    checks = {'total_position_RMSE_at_least_15pct_lower':new_total<=.85*old_total,
              'no_heldout_level_worse':all(r['new_RMSE_m']<=r['old_RMSE_m'] for r in levels.values()),
              'terminal_heading_MAE_at_least_20pct_lower':new_yaw<=.8*old_yaw}
    return {'diagnostic_pass':all(checks.values()),'checks':checks,'levels':levels,
            'new_RMSE_m':new_total,'old_RMSE_m':old_total,
            'new_terminal_heading_MAE_rad':new_yaw,'old_terminal_heading_MAE_rad':old_yaw,
            'calibration_pass':False,'NAV043_passed':False,'G5_passed':False}


def run():
    if (socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX')
            or Path(sys.executable).resolve()!=Path(base.PYTHON).resolve()):
        raise ValueError('requires tiger11/tmux/registered Python')
    p,a = approval()
    original = base.verify_protocol()
    if p['trace_protocol_sha256']!=base.sha(base.PROTOCOL):
        raise ValueError('different trajectory scope')
    output = Path(p['output_root'])
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    def timeout(*_):
        raise TimeoutError('affine diagnostic CPU wall allocation reached')
    previous = signal.signal(signal.SIGALRM,timeout)
    signal.alarm(LIMITS['cpu_wall_s'])
    rows,solves = [],[]
    try:
        base.write_new(output/'STARTED.json',{'at':datetime.now().astimezone().isoformat(),
            'pane':os.environ['TMUX_PANE'],'host':socket.gethostname(),
            'proposal_sha256':base.sha(PROPOSAL),'approval_sha256':base.sha(APPROVAL)})
        old = base.load(base.NAV/'reports/nav043_motion_response_review_v1.json')
        traces = [base.read_trace(e) for e in original['traces']]
        for level in base.LEVELS:
            training = [t for t in traces if t.identity['level']!=level]
            heldout = [t for t in traces if t.identity['level']==level]
            folder = output/level
            folder.mkdir()
            parameters = []
            for axis,name in enumerate(base.AXES):
                x = np.concatenate([np.column_stack((t.observed_twists[:-1,axis],
                    t.interval_commands[1:len(t.observed_twists),axis],
                    np.ones(len(t.observed_twists)-1))) for t in training])
                y = np.concatenate([t.observed_twists[1:,axis] for t in training])
                item = dict(level=level,axis=name,ordinal=len(solves)+1,rows=len(y))
                if item['ordinal']>LIMITS['axis_solves']:
                    raise ValueError('solve limit reached')
                base.write_new(folder/(name+'-solve-started.json'),item)
                solves.append(item)
                item.update(fit_axis(x,y))
                base.write_new(folder/(name+'-solve-result.json'),item)
                if not item['valid']:
                    raise ValueError('invalid affine response; no clipping or retries')
                parameters.append([item['a'],item['b'],item['d']])
            parameters = np.asarray(parameters)
            base.write_new(folder/'model.json',{'parameters_a_b_d':parameters.tolist(),
                'training_trials':[t.identity['trial'] for t in training],
                'heldout_trials':[t.identity['trial'] for t in heldout],
                'deployment_model':False,'initial_response':'zero, v1 matched diagnostic only'})
            for trace in heldout:
                poses,responses = free_replay(trace,parameters)
                row = {k:trace.identity[k] for k in ('trial','level','method','kind')}
                row['metrics'] = base.metrics(trace,poses,responses,trace.ticks)
                row['prefixes'] = {str(s):base.metrics(trace,poses,responses,round(s/base.PHYSICS_DT))
                    if round(s/base.PHYSICS_DT)<=trace.ticks else {'complete':False}
                    for s in base.PREFIXES}
                rows.append(row)
                base.write_new(folder/f"{trace.identity['kind']}-{trace.identity['method']}.json",row)
        base.verify_protocol(); approval()
        base.write_new(output/'FINISHED.json',{'rows':rows,'solves':solves,
            'gate':gate(rows,old),'cpu_wall_s':time.monotonic()-started,
            'status':'EXPOSED_SAVED_AFFINE_RESPONSE_DIAGNOSTIC_ONLY',
            'proposal_sha256':base.sha(PROPOSAL),'approval':a,
            'new_physical_episodes':0,'new_nominal_episodes':0,'W_queries':0,
            'actor_critic_updates':0,'calibration_fits':0})
    except BaseException as error:
        base.write_new(output/'HELD.json',{'error':repr(error),'solves':solves,
            'rows_completed':len(rows),'cpu_wall_s':time.monotonic()-started,'NAV043_passed':False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    run()
