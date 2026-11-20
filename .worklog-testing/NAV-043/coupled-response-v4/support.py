"""Saved structural review and SVD-only candidate support, never fit targets."""
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path('/home/caohy/repositories/Paper')
EV=ROOT/'.worklog-testing/NAV-043/coupled-response-v4'
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_coupled_response as c
import nav043_motion_probe_recovery_guard as g

c.old.context()
assert not (c.base.NAV/'config/nav043_coupled_response_approval_v4.json').exists()
assert not Path('/pub/data/caohy/work/camra_navigation/nav043_coupled_response_probe_v4').exists()
identities=g.protected_sources()
assert identities==dict(protected_identities=295,existing_raw_input_hashes=90,bundle_files=29)
oldreport=c.base.load(c.base.NAV/'reports/nav043_supported_response_review_v3.json')
for item in oldreport['output_index']:
    assert c.base.sha(item['path'])==item['sha256']
cases=c.old.load_cases()
nav=[case for case in cases if case.cohort=='navigation']
commands=np.concatenate([case.trace.commands for case in nav])
phi=c.old.features(commands)
assert np.min(commands[:,0])>.2 and np.max(commands[:,0])<1.
identity_error=np.max(np.abs(phi[:,3]-phi[:,0]+.1*phi[:,4])).item()
assert identity_error<1.e-12
old_rank=int(np.linalg.matrix_rank(phi,tol=1.e-10))
assert old_rank==4
folds=[]
with patch.object(np.linalg,'lstsq',side_effect=AssertionError('no coefficient solve in support')):
    for level,train,held in c.old.folds(cases):
        x,y,weights=c.design(train)
        support=c.rank_support(x)
        folds.append(dict(level=level,train=[case.key for case in train],
                          heldout=[case.key for case in held],rows=len(x),
                          weighted_y_shape=list(y.shape),**support))
probe=[]
for case in cases:
    if case.cohort=='probe':
        probe.append(dict(kind=case.trace.identity['kind'],command=case.trace.interval_commands[0].tolist(),
                          last_20_completed_q_mean=case.trace.observed_twists[-20:].mean(axis=0).tolist()))
prefixes=[]
for row in oldreport['rows']:
    if row['cohort']=='navigation' and row['level']=='L1_empty':
        prefixes.append(dict(kind=row['kind'],method=row['method'],
            candidate={s:dict(position_RMSE_m=v['position_RMSE_m'],heading_terminal_rad=v['heading_terminal_rad'])
                       for s,v in row['candidate_prefixes'].items()},
            affine={s:dict(position_RMSE_m=v['position_RMSE_m'],heading_terminal_rad=v['heading_terminal_rad'])
                    for s,v in row['affine_prefixes'].items()}))
source=[Path(c.__file__),EV/'support.py',EV/'support.sh',
        c.base.NAV/'reports/nav043_supported_response_review_v3.json',
        ROOT/'.worklog-testing/NAV-043/supported-execution-v3/close.json']
report=dict(status='SVD_ONLY_PREPARATION',host=socket.gethostname(),pane=os.environ['TMUX_PANE'],
    python=sys.executable,identities=identities,actual_parameter_solves=0,
    source_cases=30,measurement_ticks=sum(case.trace.ticks for case in cases),
    nav_command_min=commands.min(axis=0).tolist(),nav_command_max=commands.max(axis=0).tolist(),
    nav_old_command_feature_columns=5,nav_old_command_feature_rank=old_rank,
    nav_hinge_linear_identity_max_error=identity_error,
    nav_identity='hinge_vx = vx - 0.1*gate; gate=1 on every saved navigation command',
    current_model_restriction='v3 constant-command q limit is affine in positive vx for vx>=0.2; this is algebra, not measured stationarity',
    candidate_features='old five + vx^2 + vx*yaw_command + vx*abs(yaw_command)',
    candidate_state='all three completed previous q components in each axis equation',
    candidate_support=folds,candidate_support_pass=all(f['valid'] for f in folds),
    probe_descriptive_means=probe,L1_saved_prefixes=prefixes,
    new_applications=0,new_episodes=0,W_queries=0,network_updates=0,calibration_fits=0,
    production_enabled=False,NAV043=False,G5_passed=False,
    limitations=['Full rank is not Markov sufficiency or predictive success.',
                 'All cases are previously exposed; no independent/final validation.',
                 'Probe means come from one saved case each, not guaranteed steady response.',
                 'No causal attribution of physical drift from these diagnostics.'],
    source_hashes={str(path.relative_to(ROOT)):c.base.sha(path) for path in source})
c.base.write_new(EV/'support.json',report)
print('STRUCTURAL_REVIEW_SVD_ONLY',[(f['level'],f['rank'],f['condition']) for f in folds],flush=True)
print('ACTUAL_COEFFICIENT_SOLVES=0',flush=True)
