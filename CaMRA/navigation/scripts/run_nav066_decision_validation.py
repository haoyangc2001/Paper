"""Frozen model budget decisions followed by independent nominal execution.

Phases are separated by the plan dependency: prepare belongs to NAV-065;
decide/evaluate belong to NAV-066. No fitting or policy selection occurs here.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch

from eval_r14_repair import E, build_registered_network, rollout
from prepare_nav065_sources import tensor_digest, REGISTRATION
from train_formal import verify_prereg
from build_split_manifest import layout_hash
from camra_calibration.power import PowerModel
from camra_navigation.maps.feasibility import check_map
from camra_navigation.algorithm.deployment_oracle import reconstruct_truth
from camra_navigation.budget.accounting import consistent_e_ref
from camra_navigation.budget.guarded import GRID, select_guarded_budget, summarize_decisions
from camra_navigation.observation import field_slices

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parents[1]


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read_registration(path, phase):
    if socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX'):
        raise ValueError('requires approved tiger11/tmux execution')
    reg = load(path)
    pin = path.with_suffix('.json.sha256').read_text().strip()
    if sha(path) != pin or reg.get('status') != 'APPROVED' or not reg.get('authorization_reference'):
        raise ValueError('requires a separately approved, hash-pinned registration')
    if reg['limits'] != {'episodes':90, 'optimizer_updates':0, 'physical_episodes':0, 'queries':90}:
        raise ValueError('unapproved episode or query allocation')
    for name, digest in reg['implementation_hashes'].items():
        if sha(NAV/name) != digest:
            raise ValueError('implementation identity mismatch: '+name)
    for name, digest in reg['protected_hashes'].items():
        if sha(name) != digest:
            raise ValueError('protected source identity mismatch: '+name)
    for c in reg['six_candidates']:
        if sha(c['checkpoint']) != c['checkpoint_sha256']:
            raise ValueError('candidate identity mismatch')
    states = {t['id']:t['status'] for t in load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')['tasks']}
    if not states['NAV-064'] or states['NAV-066'] or (phase=='prepare' and states['NAV-065']) or (phase!='prepare' and not states['NAV-065']):
        raise ValueError('plan dependency or completed-run state mismatch')
    prereg, train_pin = verify_prereg(REGISTRATION)
    if prereg['environment']['t_max_s'] != reg['horizon_s'] or reg['horizon_s'] != 45.0:
        raise ValueError('must preserve the selected models frozen 45-second objective')
    nav = load(NAV/prereg['binding']['navigation_contract']['path'])
    value = load(NAV/prereg['binding']['value_contract']['path'])
    if value['quantities']['Z_REF_J']['value'] != 550 or value['quantities']['beta']['value'] != 0:
        raise ValueError('fixed value objective mismatch')
    return reg, pin, prereg, nav, value


def networks(reg, prereg, nav):
    nets = {}
    for c in reg['selected_candidates']:
        state = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
        net = build_registered_network(nav,prereg)
        net.load_state_dict(state['model'])
        net.eval()
        for role, digest in c['state_dict_sha256'].items():
            if tensor_digest(getattr(net,role)) != digest:
                raise ValueError('CPU reload tensor mismatch')
        nets[c['variant']] = net
    return nets


def prepare(reg, pin, prereg, nav, value, out):
    out.mkdir(exist_ok=False)
    write_new(out/'owner.json', {'host':socket.getfqdn(),'tmux':os.environ['TMUX'],'registration_pin':pin,'created_at':datetime.now().astimezone().isoformat()})
    # Only metadata is searched; old raw traces remain bound to their original
    # manifests. No old final trajectory or performance result is opened.
    pattern = r'(-s91700[123]\b|"seed"\s*:\s*91700[123]\b)'
    history = subprocess.run(['rg','-l','--glob','*.json',pattern,str(Path(E.WORK_ROOT))],capture_output=True,text=True)
    if history.returncode not in (0,1):
        raise ValueError('historical metadata exposure audit failed: '+history.stderr)
    if history.stdout.strip():
        raise ValueError('proposed fresh task seed was already exposed: '+history.stdout)
    old = load(E.SPLIT)
    old_maps = [m for s in old['splits'].values() for m in s['maps']]
    old_geometry = {m['geometry_hash'] for m in old_maps}
    old_seeds = {m['seed'] for m in old_maps}
    assert not old_seeds.intersection(reg['map_seeds'])
    ctx = E.build_rollout_ctx('full')
    power = PowerModel.from_dict(load(E.POWER))
    rows = []
    for level in reg['levels']:
        for seed in reg['map_seeds']:
            spec = E.M.build(level,seed,ctx['design'])
            admission = check_map(spec,ctx['geom'],power,resolution=.1,t_deadline_s=reg['horizon_s'])
            if not admission['admitted'] or spec.geometry_hash() in old_geometry:
                write_new(out/'HELD_admission.json',admission)
                raise ValueError('fresh map admission or task-instance exclusivity failed; no seed replacement')
            ref = admission['route']['predicted_energy_deadline_J']
            if not math.isfinite(ref) or ref <= 0:
                raise ValueError('invalid policy-independent budget reference')
            rows.append({'map_id':spec.map_id,'level':level,'seed':seed,
                         'geometry_hash':spec.geometry_hash(),'layout_hash':layout_hash(spec),
                         'spec':spec.to_dict(),'admission':admission,'e_ref_nominal_J':ref,
                         'e_ref_consistent_J':{m:consistent_e_ref(ref,m) for m in ('uncalibrated','full')}})
            print('admitted',spec.map_id,ref,flush=True)
    manifest = {'schema':'camra-budget-decision-selection-manifest-v2','registration_pin':pin,
                'split':'budget_selection_decision_v2','horizon_s':reg['horizon_s'],'maps':rows,
                'historical_metadata_exposure_matches':[],
                'historical_audit_scope':'all recorded JSON metadata under registered WORK_ROOT plus all original split metadata; raw traces use their pre-existing bound manifests',
                'unseen_layout_claim':False,'final_results_opened':False}
    write_new(out/'manifest.json',manifest)
    # Check all six bound identities; only the selected pair is later evaluated.
    reloads = networks({**reg, 'selected_candidates':reg['six_candidates']},prereg,nav)
    write_new(out/'readiness.json',{'status':'PROVISIONAL_RESEARCH_READY','registration_pin':pin,
                                  'manifest_sha256':sha(out/'manifest.json'),'cpu_reloaded_methods':list(reloads),
                                  'actor_or_critic_updates':0,'new_rollouts':0,
                                  'old_G4':'FAIL','scientific_validation':False,
                                  'call_chain':'NAV-066 runner -> select_guarded_budget -> same CPU-reloaded critics; actual trajectories only in evaluate phase'})


def make_env(spec, ctx, z0, prereg):
    stag, sev = prereg['stagnation'], prereg['safety_continuity']
    return E.VectorNominalNavEnv(spec,ctx['geom'],ctx['cm'],n_envs=1,z0_J=z0,
        t_max_s=prereg['environment']['t_max_s'],r_goal_m=prereg['environment']['r_goal_m'],
        observation_version=prereg['implementation_versions']['observation'],
        stagnation_window_s=stag['window_s'],stagnation_min_displacement_m=stag['min_displacement_m'],
        safety_severity_enabled=sev['enabled'],safety_severity_floor=sev['severity_floor'],
        safety_severity_ref_depth_m=sev['severity_ref_depth_m'])


def decide(reg, pin, prereg, nav, value, out):
    ready = load(out/'readiness.json')
    if ready['registration_pin'] != pin or ready['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('readiness manifest mismatch')
    manifest = load(out/'manifest.json')
    nets = networks(reg,prereg,nav)
    decisions = []
    for c in reg['selected_candidates']:
        method = c['variant']
        ctx, net = E.build_rollout_ctx(method), nets[method]
        for m in manifest['maps']:
            spec = E.M.build(m['level'],m['seed'],ctx['design'])
            observations = []
            def query(z):
                env = make_env(spec,ctx,z,prereg)
                obs = env.observation.copy()
                if not np.allclose(obs[:,field_slices('obs-v7')['z_ratio']],z/550,atol=1e-7,rtol=0):
                    raise ValueError('initial raw J budget encoded incorrectly or clipped')
                observations.append(obs[0].tolist())
                with torch.no_grad():
                    vc,vh = net.values(torch.as_tensor(obs,dtype=torch.float32))
                return vc.item(),vh.item()
            decision = select_guarded_budget(query,m['e_ref_consistent_J'][method])
            decisions.append({'method':method,'candidate_id':c['run_id'],'checkpoint_sha256':c['checkpoint_sha256'],
                              'map_id':m['map_id'],'decision':decision,'query_observations':observations})
    if len(decisions) != 18 or sum(len(d['decision']['trace']) for d in decisions) != 90:
        raise ValueError('query allocation mismatch')
    write_new(out/'decisions.json',{'registration_pin':pin,'manifest_sha256':sha(out/'manifest.json'),
                                   'query_count':90,'decisions':decisions,'rollouts_started':0})
    (out/'decisions.json.sha256').write_text(sha(out/'decisions.json')+'\n')
    print('all 18 decisions and 90 queries sealed before any rollout',flush=True)


def evaluate(reg, pin, prereg, nav, value, out):
    decision_path = out/'decisions.json'
    if sha(decision_path) != (out/'decisions.json.sha256').read_text().strip():
        raise ValueError('sealed decisions changed')
    saved = load(decision_path)
    if saved['registration_pin'] != pin or saved['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('sealed decision version mismatch')
    manifest = load(out/'manifest.json')
    maps = {m['map_id']:m for m in manifest['maps']}
    nets = networks(reg,prereg,nav)
    episodes = []
    with (out/'episodes.jsonl').open('x') as stream:
        for d in saved['decisions']:
            method = d['method']
            m = maps[d['map_id']]
            ctx, net = E.build_rollout_ctx(method), nets[method]
            spec = E.M.build(m['level'],m['seed'],ctx['design'])
            chosen = d['decision']['selected_index']
            indices = ([] if chosen is None else [chosen])+[i for i in range(5) if i != chosen]
            for i in indices:
                if len(episodes) >= reg['limits']['episodes']:
                    raise ValueError('episode allocation exhausted')
                k = GRID[i]
                stream.write(json.dumps({'kind':'episode_start','number':len(episodes)+1,'method':method,'map_id':m['map_id'],'k':k})+'\n')
                stream.flush(); os.fsync(stream.fileno())
                ep = rollout(net,spec,ctx,k*m['e_ref_consistent_J'][method],prereg)
                if ep['n_steps'] > 450:
                    raise ValueError('frozen horizon step cap exceeded')
                truth = reconstruct_truth(ep,initial_goal_distance_m=math.dist(spec.start,spec.goal),
                    value_contract=value,horizon_s=reg['horizon_s'],goal_radius_m=prereg['environment']['r_goal_m'],step_s=.1)
                row = {'method':method,'map_id':m['map_id'],'k':k,'selected':i==chosen,'episode':ep,
                       'actual_feasible':truth.feasible,'initial_truth_W':float(truth.path_residual[0])}
                episodes.append(row)
                stream.write(json.dumps({'kind':'episode_result',**row},allow_nan=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
                print('episode',len(episodes),method,m['map_id'],k,ep['label'],ep['feasible'],flush=True)
    result = summarize_decisions(saved['decisions'],episodes,manifest['maps'])
    result.update(registration_pin=pin,decisions_sha256=sha(decision_path),manifest_sha256=sha(out/'manifest.json'),
                  episodes_sha256=sha(out/'episodes.jsonl'),steps=sum(e['episode']['n_steps'] for e in episodes),
                  horizon_s=reg['horizon_s'],host=socket.getfqdn(),tmux=os.environ['TMUX'],
                  updates=0,physical_episodes=0,final_results_opened=False)
    write_new(out/'decision_validation.json',result)
    write_new(out/'FINISHED.json',{'verdict':'PASS' if result['pass'] else 'FAIL','report_sha256':sha(out/'decision_validation.json')})
    print(json.dumps({m:{k:v for k,v in s.items() if k!='details'} for m,s in result['methods'].items()},ensure_ascii=False),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','decide','evaluate'),required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    reg,pin,prereg,nav,value = read_registration(args.registration,args.phase)
    out = Path(reg['output_directory'])
    {'prepare':prepare,'decide':decide,'evaluate':evaluate}[args.phase](reg,pin,prereg,nav,value,out)


if __name__ == '__main__':
    main()
