"""Actual-state budget decisions and matching nominal rollouts, no fitting."""
import math
import numpy as np
import torch

from nav067_sources import E, sha
from nav067_runtime import persist
from run_nav066_decision_validation import make_env
from camra_navigation.budget.accounting import consistent_e_ref
from camra_navigation.budget.guarded import select_guarded_budget
from camra_navigation.envs.physical_v2 import SubstepNavigation


def decide(net, spec, ctx, initial, reference_J, prereg, trial):
    initial.validate()
    observations = []
    def query(z):
        monitor = SubstepNavigation(spec,ctx['geom'],ctx['cm'],[initial],[z],prereg)
        obs = np.asarray(monitor.observation(0),dtype=np.float32)
        observations.append(obs.tolist())
        with torch.no_grad():
            vc,vh = net.values(torch.as_tensor(obs[None,:]))
        return vc.item(),vh.item()
    decision = select_guarded_budget(query,consistent_e_ref(reference_J,ctx['cm'].variant))
    saved = {'decision':decision,'query_observations':observations,
             'initial_state':vars(initial),'query_count':len(observations),
             'nominal_reference_J':reference_J,'physical_actions_started':False,
             'basis':'actual warmup state; original five-point W rule with one-step reserve'}
    persist(trial/'decision.json',saved)
    persist(trial/'decision.pin.json',{'sha256':sha(trial/'decision.json')})
    return decision


def actual_start_env(spec, ctx, z, prereg, initial):
    """Isolated adapter for the frozen vector environment's initial state."""
    env = make_env(spec,ctx,z,prereg)
    env._pose[0] = [initial.x,initial.y,initial.yaw]
    env._applied_last[0] = [initial.vx,initial.vy,initial.wz]
    env._history[0] = [(0.,initial.x,initial.y)]
    env._refresh_obs()
    monitor = SubstepNavigation(spec,ctx['geom'],ctx['cm'],[initial],[z],prereg)
    if not np.array_equal(np.asarray(monitor.observation(0)),env.observation[0]):
        raise ValueError('physical/nominal initial observation mismatch')
    return env


def nominal(net, spec, ctx, initial, z, prereg):
    env = actual_start_env(spec,ctx,z,prereg,initial)
    trace = []
    while not env.done[0]:
        if len(trace) >= 450:
            raise ValueError('nominal 45 second horizon exceeded')
        obs = env.observation.copy()
        with torch.no_grad():
            action,_,_ = net.act(torch.as_tensor(obs,dtype=torch.float32),deterministic=True)
        rec = env.step(action.numpy())
        trace.append({'step':len(trace),'observation':obs[0].tolist(),
            'action':action[0].tolist(),'pose':env.pose[0].tolist(),
            'z_J':float(rec.z_J[0]),'safety_y':float(rec.safety_y[0]),
            'danger_violation':bool(rec.danger_violation[0]),
            'goal_distance_m':float(rec.goal_distance_m[0])})
    result = env.result(0)
    return {'initial_state':vars(initial),'z0_J':z,'label':env._label[0],
            't_s':float(result.t_s),'cost_model_J':float(result.cost_model_J),
            'feasible':bool(result.arrived_within_budget and result.safety_y==0.),'trace':trace}


def outcome(label, safety_y, t_s, measured_J, internal_J, budget_J):
    if not all(math.isfinite(x) for x in (safety_y,t_s,measured_J,internal_J,budget_J)) or budget_J<=0:
        raise ValueError('invalid physical outcome units')
    reached = label=='goal_reached' and t_s<=45.+1e-9
    safe = safety_y==0. and label not in ('collision','fall','out_of_bounds')
    return {'timely_goal':reached,'safe':safe,
            'safe_timely_measured_budget_success':reached and safe and measured_J<=budget_J,
            'internal_safe_budget_success':reached and safe and internal_J<=budget_J,
            'measured_budget_utilization':measured_J/budget_J,
            'internal_budget_utilization':internal_J/budget_J,
            'measured_overspend_J':max(0.,measured_J-budget_J),
            'measured_overspend_fraction':max(0.,measured_J/budget_J-1.),
            'grid_minimum_efficiency':'not estimated; no grid trajectory sweep authorized'}
