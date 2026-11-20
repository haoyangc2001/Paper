"""Main-agent basic checks of deadline separation and accounting semantics."""
import copy
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from nav067_sources import E, NAV, load
from prepare_nav065_sources import REGISTRATION
from train_formal import verify_prereg
from camra_navigation.envs.physical_v2 import PhysicalState, SubstepNavigation
from camra_navigation import observation as O
from nav043_timeout import TimeoutNavigation, diagnostic_outcome

assert os.environ.get('TMUX') and socket.gethostname().split('.')[0] == 'tiger11'
prereg, _ = verify_prereg(REGISTRATION)
ctx = E.build_rollout_ctx('full')
spec = E.M.build('L1_empty', 3001, ctx['design'])
initial = PhysicalState(*spec.start, 0., 0., 0., 0., .3, (1., 0., 0., 0.))
def make(cls=TimeoutNavigation):
    return cls(spec, ctx['geom'], ctx['cm'], [initial], [600.], prereg)
checks = []
old, new = make(SubstepNavigation), make()
for t in (0., 1., 30., 44.9, 45.):
    old.times[0] = new.times[0] = t
    np.testing.assert_array_equal(old.observation(0), new.observation(0))
    checks.append('old/new obs-v7 identity at '+str(t))
new.times[0] = 45.
zero = new.observation(0)
for t in (45.1, 60., 89.9):
    new.times[0] = t
    np.testing.assert_array_equal(zero, new.observation(0))
    checks.append('actor clock remains zero at '+str(t))
for deadline, expected in ((45., None), (90., 'timeout')):
    mon = make()
    mon.times[0] = deadline-.005
    mon.tick = int(round(deadline/.005))-1
    mon.history[0] = [(deadline-1., initial.x, initial.y)]
    mon.advance([initial], physics_tick=mon.tick+1)
    assert mon.labels[0] == expected
    checks.append('physical event at '+str(deadline)+' is '+str(expected))
for t, expected in ((44., True), (60., True), (91., False)):
    assert diagnostic_outcome('goal_reached', 0., t, 500., 550., 600.)['timely_goal'] == expected
    checks.append('arrival deadline '+str(t))
assert diagnostic_outcome('goal_reached', 0., 60., 601., 500., 600.)['safe_timely_measured_budget_success'] is False
assert diagnostic_outcome('goal_reached', .01, 60., 500., 500., 600.)['safe_timely_measured_budget_success'] is False
assert diagnostic_outcome('collision', 0., 60., 500., 500., 600.)['safe'] is False
checks.append('unsafe and over-budget arrivals remain failures')
for bad in ('timeout', 'stagnation', 'fall', 'out_of_bounds'):
    assert not diagnostic_outcome(bad, 0., 60., 500., 500., 600.)['timely_goal']
checks.append('other terminal events retain failure')
print(json.dumps({'status':'MAIN_AGENT_TIMEOUT_BASIC_PASS', 'checks':checks,
    'count':len(checks), 'applications':0, 'W_queries':0, 'training_updates':0,
    'physical_steps':0, 'tmux':os.environ['TMUX'], 'pane':os.environ.get('TMUX_PANE'),
    'python':sys.executable, 'cwd':str(Path.cwd())}, ensure_ascii=False), flush=True)
