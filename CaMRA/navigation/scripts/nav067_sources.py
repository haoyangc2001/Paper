"""Frozen selected-pair binding for NAV-067; never train or query a critic.

The old decision registration allocated nominal episodes only. This module
consumes its identities, not its exhausted quota or completed-run entry point.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket

from eval_r14_repair import E
from prepare_nav065_sources import REGISTRATION
from run_nav066_decision_validation import networks
from train_formal import verify_prereg

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parents[1]
OUTPUT = Path(E.WORK_ROOT) / 'nav067_clean_pilot_v1'
LEVELS = ('L1_empty', 'L2_single_detour', 'L3_multi_path')
MAP_SEED = 3001  # original NAV-042 wiring instances; no independence claim


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def original_pilot_quota(path):
    """Read one pinned scalar from the legacy JSON-like protocol.

    Its unrelated development total is literally ``4 * 6 * 6``. Do not
    evaluate expressions or rewrite that frozen historical document.
    """
    path = Path(path)
    if sha(path) != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('original physical-pilot protocol hash mismatch')
    matches = re.findall(r'"physical_pilot"\s*:\s*\{\s*"clean_6_episodes"\s*:\s*(\d+)\s*,',
                         path.read_text())
    if matches != ['6']:
        raise ValueError('original six-episode allocation is not unique')
    return 6


def read_sources():
    if socket.gethostname().split('.')[0] != 'tiger11' or not os.environ.get('TMUX'):
        raise ValueError('NAV-067 requires the approved tiger11/tmux session')
    plan = load(ROOT/'doc/plan/20261109_camra_navigation_implementation.json')
    state = {t['id']:t['status'] for t in plan['tasks']}
    ready = next(t['id'] for t in plan['tasks'] if not t['status']
                 and all(state[d] for d in t['depends_on']))
    if ready != 'NAV-067' or not state['NAV-066']:
        raise ValueError('NAV-067 is not the first executable task')
    path = NAV/'config/nav066_decision_prereg_v2.json'
    reg = load(path)
    pin = Path(str(path)+'.sha256').read_text().strip()
    if sha(path) != pin or reg['status'] != 'APPROVED':
        raise ValueError('selected-pair registration identity mismatch')
    accepted = load(NAV/'reports/nav066_efficiency_limitation_acceptance_v1.json')
    if accepted['disposition'] != 'ACCEPTED_WITH_LIMITATION':
        raise ValueError('independent efficiency limitation not accepted')
    for key in ('immutable_preregistration', 'immutable_result'):
        source = accepted[key]
        if sha(ROOT/source['path']) != source['sha256']:
            raise ValueError('accepted original changed: '+source['path'])
    if accepted['model_scope'] != reg['selected_candidates']:
        raise ValueError('accepted scope and selected candidates differ')
    protected = {str(NAV/k):v for k,v in reg['implementation_hashes'].items()}
    protected.update(reg['protected_hashes'])
    protected.update({c['checkpoint']:c['checkpoint_sha256'] for c in reg['six_candidates']})
    for name,digest in protected.items():
        if sha(name) != digest:
            raise ValueError('protected identity mismatch: '+name)
    prereg, train_pin = verify_prereg(REGISTRATION)
    if prereg['environment']['t_max_s'] != 45. or prereg['implementation_versions']['observation'] != 'obs-v7':
        raise ValueError('selected 45s/obs-v7 objective mismatch')
    nav = load(NAV/prereg['binding']['navigation_contract']['path'])
    nets = networks(reg, prereg, nav)  # checks all three tensor roles on CPU
    if set(nets) != {'full','uncalibrated'}:
        raise ValueError('exactly the accepted selected pair is required')
    original_pilot_quota(NAV/'config/training_loop_protocol_v1.json')
    return reg, prereg, nets, {
        'decision_prereg_sha256':pin, 'training_prereg_sha256':train_pin,
        'protected_hashes_checked':len(protected), 'plan_version':plan['document']['version'],
        'controller':nav['controller'], 'accepted_scientific_verdict':accepted['scientific_verdict']}


def prepare():
    from camra_calibration.power import PowerModel
    from camra_navigation.maps.feasibility import check_map
    from camra_navigation.maps.scene_adapter import build_scene

    reg, prereg, _, binding = read_sources()
    # x mode makes a second allocation under the same registration impossible.
    OUTPUT.mkdir(exist_ok=False)
    write_new(OUTPUT/'owner.json', {'host':socket.getfqdn(), 'tmux':os.environ['TMUX'],
        'created_at':datetime.now().astimezone().isoformat(), 'binding':binding})
    ctx = E.build_rollout_ctx('full')
    power = PowerModel.from_dict(load(E.POWER))
    rows = []
    for level in LEVELS:
        spec = E.M.build(level, MAP_SEED, ctx['design'])
        admission = check_map(spec, ctx['geom'], power, resolution=.1, t_deadline_s=45.)
        if not admission['admitted']:
            write_new(OUTPUT/('HELD_'+level+'.json'), admission)
            raise ValueError('pilot map not admitted; no replacement allowed')
        rows.append({'level':level, 'map_id':spec.map_id, 'seed':MAP_SEED,
            'geometry_hash':spec.geometry_hash(), 'spec':spec.to_dict(),
            'scene':build_scene(spec,0,1).as_dict(), 'admission':admission,
            'budget_mode':'common_budget', 'k':1.0,
            'z0_J':float(admission['route']['predicted_energy_deadline_J']),
            'budget_source':'policy-independent nominal route at frozen 45s; same J for both methods; not W-selected'})
    manifest = {'schema':'camra-nav067-clean-pilot-v1', 'binding':binding,
        'candidates':reg['selected_candidates'], 'maps':rows,
        'limits':{'episodes':6, 'episodes_per_level':2, 'horizon_s':45.,
                  'warmup_s':1., 'optimizer_updates':0, 'W_queries':0},
        'split':'integration_pilot', 'performance_evidence':False,
        'output_root':str(OUTPUT),
        'quota_source':'training_loop_protocol_v1.json resource_ledger.physical_pilot.clean_6_episodes',
        'failure_policy':'preserve every attempt; a level claims its two slots once before simulator creation; do not restart or add repetitions automatically',
        'unallocated':['same-configuration repetition','controlled physical counterexamples'],
        'new_sources':{p:sha(NAV/p) for p in ('scripts/nav067_sources.py',
            'scripts/evaluate_navigation_v2.py','src/camra_navigation/envs/clean_v2.py',
            'src/camra_navigation/envs/physical_v2.py')}}
    write_new(OUTPUT/'manifest.json', manifest)
    (OUTPUT/'manifest.json.sha256').write_text(sha(OUTPUT/'manifest.json')+'\n')
    print(json.dumps({'manifest':str(OUTPUT/'manifest.json'), 'binding':binding,
        'maps':[{'map_id':r['map_id'],'z0_J':r['z0_J']} for r in rows],
        'new_physical_episodes':0, 'training_updates':0, 'W_queries':0},ensure_ascii=False))


def read_manifest():
    reg, prereg, nets, binding = read_sources()
    path = OUTPUT/'manifest.json'
    if sha(path) != Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('pilot manifest identity mismatch')
    manifest = load(path)
    if manifest['binding'] != binding or manifest['candidates'] != reg['selected_candidates']:
        raise ValueError('pilot frozen model binding mismatch')
    for name,digest in manifest['new_sources'].items():
        if sha(NAV/name) != digest:
            raise ValueError('pilot source changed after sealing: '+name)
    if manifest['limits'] != {'episodes':6,'episodes_per_level':2,'horizon_s':45.,
                             'warmup_s':1.,'optimizer_updates':0,'W_queries':0}:
        raise ValueError('pilot quota/objective mismatch')
    return manifest, prereg, nets


if __name__ == '__main__':
    prepare()
