"""Read saved commands and affine coefficients; no fit, replay or policy."""
from datetime import datetime
import os
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_affine_response as affine

base = affine.base
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert Path(sys.executable).resolve() == Path(base.PYTHON).resolve()
proposal, _ = affine.approval()
protocol = base.verify_protocol()
report_path = base.NAV/'reports/nav043_affine_response_review_v2.json'
previous = base.load(report_path)
assert previous['status'] == 'SAVED_AFFINE_EXECUTION_AUDIT_PASS'
commands, trace_ranges = [], []
for entry in protocol['traces']:
    rows = base.load(Path(entry['trial'])/'navigation.json')
    command = np.asarray([row['command'][0] for row in rows],dtype=float)
    assert command.shape == (len(rows),3) and np.isfinite(command).all()
    commands.append(command)
    trace_ranges.append({'trial':entry['trial'], 'intervals_including_partial_tail':len(command),
        'minimum_applied':command.min(axis=0).tolist(),
        'maximum_applied':command.max(axis=0).tolist()})
command = np.concatenate(commands)
derived = []
for level in base.LEVELS:
    saved = base.load(Path(proposal['output_root'])/level/'model.json')
    p = np.asarray(saved['parameters_a_b_d'])
    assert p.shape == (3,3) and np.isfinite(p).all() and np.all((0<=p[:,0])&(p[:,0]<1))
    derived.append({'heldout_level':level, 'parameters_a_b_d':p.tolist(),
        'predicted_first_interval_from_rest_under_zero_command':p[:,2].tolist(),
        'predicted_zero_command_equilibrium_q':(p[:,2]/(1-p[:,0])).tolist(),
        'constant_input_for_zero_equilibrium_q':(-p[:,2]/p[:,1]).tolist(),
        'meaning':'Algebra of saved diagnostic parameters, not a physical observation or compensation command.'})
base.verify_protocol()
result = {'status':'SAVED_COMMAND_SUPPORT_AND_PARAMETER_ALGEBRA_ONLY',
    'at':datetime.now().astimezone().isoformat(), 'host':socket.gethostname(),
    'pane':os.environ['TMUX_PANE'], 'python':sys.executable, 'source_sha256':base.sha(__file__),
    'upstream_audit_sha256':base.sha(report_path), 'trace_count':18,
    'interval_commands_including_partial_tails':len(command),
    'axes':list(base.AXES), 'command_quantile_levels':[0.,.05,.5,.95,1.],
    'command_quantiles':np.quantile(command,[0.,.05,.5,.95,1.],axis=0).tolist(),
    'exact_zero_command_intervals':int(np.all(command==0.,axis=1).sum()),
    'near_zero_command_intervals_max_abs_001':int((np.abs(command).max(axis=1)<=.01).sum()),
    'forward_command_abs_below_01_intervals':int((np.abs(command[:,0])<.1).sum()),
    'trace_ranges':trace_ranges, 'saved_parameter_algebra':derived,
    'protected_identities_checked':295, 'raw_input_hashes_checked':90,
    'new_axis_solves':0, 'new_replays':0, 'new_business_episodes':0,
    'W_queries':0, 'network_updates':0, 'calibration_fits':0,
    'deployment_model_selected':False, 'NAV043_passed':False, 'G5_passed':False,
    'limitation':'Bounding ranges are descriptive, not joint support, extrapolation validity, physical root cause or a ninety-second guarantee.'}
base.write_new(base.NAV/'reports/nav043_affine_support_review_v1.json',result)
print({k:v for k,v in result.items() if k not in ('trace_ranges','saved_parameter_algebra')})
for row in derived:
    print(row)
