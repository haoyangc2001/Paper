"""Descriptive accounting of saved responses; no solves or simulator/network calls."""
from datetime import datetime
import os
from pathlib import Path
import socket
import sys

import numpy as np

import nav043_motion_response as m


def rotate(yaw, vectors):
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = np.asarray(vectors).T
    return np.column_stack((c*x-s*y, s*x+c*y))


def decompose(initial, actual, predicted):
    """Exact world-displacement identity with an explicitly ordered split.

    H = (R(predicted previous yaw)-R(actual previous yaw))*predicted body delta.
    B = R(actual previous yaw)*(predicted body delta-actual body delta).
    H+B telescopes to predicted position-actual position. Norms are not causal
    contributions: their cross term can cancel. Truth is used for accounting
    only, never to correct the saved free prediction.
    """
    actual, predicted, initial = map(np.asarray, (actual, predicted, initial))
    if (actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 3
            or len(actual) == 0 or initial.shape != (3,)
            or not all(np.isfinite(a).all() for a in (actual,predicted,initial))):
        raise ValueError('finite aligned pose arrays required')
    a0 = np.vstack((initial, actual[:-1]))
    p0 = np.vstack((initial, predicted[:-1]))
    da, dp = actual[:,:2]-a0[:,:2], predicted[:,:2]-p0[:,:2]
    ba, bp = rotate(-a0[:,2],da), rotate(-p0[:,2],dp)
    h = np.cumsum(rotate(p0[:,2],bp)-rotate(a0[:,2],bp),axis=0)
    b = np.cumsum(rotate(a0[:,2],bp-ba),axis=0)
    error = predicted[:,:2]-actual[:,:2]
    closure = float(np.max(np.abs(h+b-error)))
    if closure > 1.e-9:
        raise ValueError('displacement accounting did not close')
    return h, b, error, closure


def stats(values):
    values = np.asarray(values)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('nonempty finite rows required')
    return {'rows':len(values),'mean':values.mean(axis=0).tolist(),
            'RMSE':np.sqrt(np.mean(values**2,axis=0)).tolist(),
            'max_abs':np.max(np.abs(values),axis=0).tolist()}


def accounting_metrics(h,b,error,closure):
    cross = 2.*np.sum(h*b,axis=1)
    total = np.sum(error**2,axis=1)
    split = np.sum(h*h+b*b,axis=1)+cross
    if not np.allclose(total,split,atol=1.e-8,rtol=1.e-10):
        raise ValueError('squared error cross term lost')
    return {'closure_max_abs_m':closure,'terminal_error_xy_m':error[-1].tolist(),
            'terminal_H_xy_m':h[-1].tolist(),'terminal_B_xy_m':b[-1].tolist(),
            'terminal_H_norm_m':float(np.linalg.norm(h[-1])),
            'terminal_B_norm_m':float(np.linalg.norm(b[-1])),
            'terminal_cross_term_m2':float(cross[-1]),
            'mean_H_squared_m2':float(np.mean(np.sum(h*h,axis=1))),
            'mean_B_squared_m2':float(np.mean(np.sum(b*b,axis=1))),
            'mean_cross_term_m2':float(np.mean(cross)),
            'position_RMSE_m':float(np.sqrt(total.mean()))}


def analyze_trace(trace, parameters):
    predicted, responses = m.free_replay(trace,parameters)
    h,b,error,closure = decompose(trace.initial,trace.actual,predicted)
    full = len(trace.observed_twists)
    q = trace.observed_twists
    # This diagnostic uses the already completed prior interval, not a future
    # response available to a deployed actor. No parameters are optimized.
    teacher_prediction = q[:-1]*parameters[:,0]+trace.interval_commands[1:full]*parameters[:,1]
    innovations = q[1:]-teacher_prediction
    rows = m.load(Path(trace.identity['trial'])/'substeps.json')
    states = np.asarray([r['state'][:6] for r in rows])
    endpoint_body = states[m.NAV_TICKS-1:full*m.NAV_TICKS:m.NAV_TICKS,3:6]
    result = m.load(Path(trace.identity['trial'])/'result.json')
    warmup = result['warmup']['state_at_window_start'][0]
    yaw_actual = float(np.sum(m.wrap(np.diff(np.r_[trace.initial[2],trace.actual[:,2]]))))
    row = {k:trace.identity[k] for k in ('trial','level','kind','method')}
    row.update(ticks=trace.ticks,complete_intervals=full,partial_tail_ticks=trace.ticks%20,
        accounting=accounting_metrics(h,b,error,closure),
        actual_minus_one_step_response=stats(innovations),
        predicted_minus_actual_free_interval_twist=stats(responses[:full]-q),
        endpoint_instantaneous_body_minus_completed_SE2_twist=stats(endpoint_body-q),
        warmup_instantaneous_body_vx_vy_wz=[warmup[k] for k in ('vx','vy','wz')],
        commanded_net_yaw_rad=float(trace.commands[:,2].sum()*m.PHYSICS_DT),
        predicted_net_yaw_rad=float(predicted[-1,2]-trace.initial[2]),
        actual_unwrapped_net_yaw_rad=yaw_actual)
    row['prefixes'] = {}
    for seconds in m.PREFIXES:
        ticks = round(seconds/m.PHYSICS_DT)
        row['prefixes'][str(seconds)] = ({'complete':True,
            **accounting_metrics(h[:ticks],b[:ticks],error[:ticks],closure)} if ticks<=trace.ticks
            else {'complete':False,'available_duration_s':trace.ticks*m.PHYSICS_DT})
    return row,innovations,endpoint_body-q


def main():
    if (socket.gethostname().split('.')[0]!='tiger11' or not os.environ.get('TMUX')
            or Path(sys.executable).resolve()!=Path(m.PYTHON).resolve()):
        raise ValueError('requires tiger11/tmux/registered Python')
    output = m.NAV/'reports/nav043_response_residuals_v1.json'
    if output.exists():
        raise ValueError('report already exists; retain it instead of rerunning')
    protocol = m.verify_protocol()
    prior_path = m.NAV/'reports/nav043_motion_response_review_v1.json'
    prior = m.load(prior_path)
    for item in prior['output_index']:
        if m.sha(item['path']) != item['sha256']:
            raise ValueError('changed saved diagnostic output')
    rows,innovations,observations = [],[],[]
    for entry in protocol['traces']:
        trace = m.read_trace(entry)
        model = m.load(Path(protocol['output_root'])/entry['level']/'model.json')
        if entry['trial'] not in model['heldout_trials'] or model['deployment_model']:
            raise ValueError('wrong saved fold/model')
        row,i,o = analyze_trace(trace,np.asarray(model['parameters_a_b']))
        previous = next(r for r in prior['rows'] if r['trial']==entry['trial'])
        if not np.isclose(row['accounting']['position_RMSE_m'],
                          previous['terminal']['response']['position_RMSE_m'],atol=1.e-10,rtol=1.e-10):
            raise ValueError('free replay no longer agrees with the audited result')
        rows.append(row); innovations.append(i); observations.append(o)
    m.verify_protocol()
    report = {'status':'SAVED_RESIDUAL_ACCOUNTING_ONLY','at':datetime.now().astimezone().isoformat(),
        'host':socket.gethostname(),'pane':os.environ.get('TMUX_PANE'),'python':sys.executable,
        'source_sha256':m.sha(__file__),'prior_review_sha256':m.sha(prior_path),
        'protocol_sha256':m.sha(m.PROTOCOL),'trace_count':len(rows),
        'saved_ticks':sum(r['ticks'] for r in rows),'rows':rows,
        'one_step_innovation_vx_vy_planar_yaw_rate':stats(np.concatenate(innovations)),
        'endpoint_body_minus_completed_interval_SE2':stats(np.concatenate(observations)),
        'maximum_accounting_closure_m':max(r['accounting']['closure_max_abs_m'] for r in rows),
        'protected_identities_checked':295,'raw_input_hashes_checked':90,
        'new_solves':0,'new_physical_episodes':0,'new_nominal_episodes':0,
        'W_queries':0,'network_updates':0,'calibration_fits':0,
        'production_prediction_corrected_with_truth':False,
        'limitations':['18 exposed integration traces, <=45 seconds, not independent 90-second evidence',
          'H/B split depends on ordering; norms and cross terms are bookkeeping, not causal shares',
          'completed interval SE2 twist differs from instantaneous body velocity and body wz',
          'one-step innovations use actual prior completed intervals for diagnosis only'],
        'NAV043_passed':False,'G5_passed':False}
    m.write_new(output,report)
    print(output)


if __name__=='__main__':
    main()
