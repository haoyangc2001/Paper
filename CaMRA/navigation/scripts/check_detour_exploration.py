"""Read-only train-prefix exploration check, after the independent raw audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camra_navigation.algorithm.storage import EVENT_NAMES
from camra_navigation.algorithm.validation import file_sha


EXPLORATION_GATE = {
    'after_updates': 50,
    'variant': 'full',
    'minimum_safe_l2_arrivals_per_seed': 1,
    'failure_action': 'HOLD_NO_RETRY',
}


def check_prefix(prereg, root, updates):
    gate = prereg['exploration_gate']
    if updates != gate['after_updates']:
        raise ValueError('wrong exploration prefix')
    nav = Path(__file__).resolve().parents[1]
    maps = json.loads((nav/prereg['binding']['split']['path']).read_bytes())['splits']['train']['maps']
    selected = [(i, m['map_id']) for i,m in enumerate(maps) if m['map_id'].startswith('l2_')]
    if len(selected) != 6:
        raise ValueError('expected all six registered train L2 maps')
    seeds = []
    for row in prereg['run_registry']:
        if row['variant'] != gate['variant']:
            continue
        results = []
        for i, mid in selected:
            store = Path(root)/row['run_id']/f'rollout_env{i}'/'rollout'
            entries = [json.loads(s) for s in (store/'manifest.jsonl').read_text().splitlines()]
            entries = [e for e in entries if e['update_idx'] < updates]
            if len(entries) != updates or [e['update_idx'] for e in entries] != list(range(updates)):
                raise ValueError('incomplete or duplicate exploration prefix')
            safe = feasible = arrivals = 0
            for e in entries:
                path = store/e['file']
                if e['map_id'] != mid or file_sha(path) != e['file_sha256']:
                    raise ValueError('exploration raw identity/hash mismatch')
                with np.load(path) as a:
                    arrived = EVENT_NAMES[int(a['event_code'][-1])] == 'goal_reached'
                    clean = bool(np.all(a['safety_y'] == 0.) and np.all(a['safety_y_next'] == 0.))
                    arrivals += arrived
                    safe += arrived and clean
                    feasible += arrived and clean and bool(a['z_norm_next'][-1] >= 0.)
            results.append(dict(map_id=mid, episodes=len(entries), arrivals=arrivals,
                                safe_arrivals=safe, feasible_arrivals=feasible))
        count = sum(m['safe_arrivals'] for m in results)
        seeds.append(dict(seed=row['seed'], run_id=row['run_id'], maps=results,
                          safe_arrivals=count, passed=count >= gate['minimum_safe_l2_arrivals_per_seed']))
    if sorted(s['seed'] for s in seeds) != [0,1,2]:
        raise ValueError('exploration requires all three full seeds')
    return dict(verdict='PASS' if all(s['passed'] for s in seeds) else 'FAIL_EXPLORATION',
                updates=updates, seeds=seeds, split='train',
                limits='A prefix exploration prerequisite only; original final development acceptance remains necessary.')
