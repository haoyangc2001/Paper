"""Train-only progress gate; never substitutes for development acceptance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .validation import file_sha

TERMINAL_TAIL_STABILITY_GATE = {
    'after_updates': 150, 'variant': 'full',
    'baseline_interval': [0, 50], 'recent_interval': [100, 150],
    'minimum_budget_k': 1.15, 'minimum_episodes_per_seed_interval': 20,
    'minimum_safe_feasible_rate': .8, 'maximum_rate_drop': .1,
    'failure_action': 'HOLD_NO_RETRY',
}


def check_learning_stability(prereg: dict, root: Path, updates: int) -> dict:
    """Compare predeclared train windows, retaining every failure/denominator.

    Budget conditioning is declared before training. All L2 episodes are also
    counted; this conditional mechanism check does not redefine business SR.
    The independent raw audit must run before the coordinator calls this gate.
    """
    gate = prereg['learning_stability_gate']
    if updates != gate['after_updates']:
        raise ValueError('wrong stability checkpoint')
    nav = Path(__file__).resolve().parents[3]
    maps = json.loads((nav/prereg['binding']['split']['path']).read_bytes())['splits']['train']['maps']
    selected = [(i,m['map_id']) for i,m in enumerate(maps) if m['map_id'].startswith('l2_')]
    if len(selected) != 6:
        raise ValueError('stability gate requires all six train L2 maps')
    seeds = []
    for row in prereg['run_registry']:
        if row['variant'] != gate['variant']:
            continue
        windows = []
        for interval in (gate['baseline_interval'], gate['recent_interval']):
            start, end = interval
            all_count = all_feasible = count = feasible = 0
            speeds = []
            for env, map_id in selected:
                store = Path(root)/row['run_id']/f'rollout_env{env}'/'rollout'
                entries = [json.loads(line) for line in (store/'manifest.jsonl').read_text().splitlines()]
                entries = [e for e in entries if start <= e['update_idx'] < end]
                if [e['update_idx'] for e in entries] != list(range(start,end)):
                    raise ValueError('incomplete/duplicate stability window')
                for entry in entries:
                    path = store/entry['file']
                    if entry['map_id'] != map_id or file_sha(path) != entry['file_sha256']:
                        raise ValueError('stability raw identity/hash mismatch')
                    with np.load(path) as a:
                        success = (int(a['event_code'][-1]) == 3
                                   and bool(np.all(a['safety_y']==0.) and np.all(a['safety_y_next']==0.))
                                   and bool(a['z_norm_next'][-1] >= 0.))
                        all_count += 1
                        all_feasible += success
                        if entry['episode_k'] >= gate['minimum_budget_k']:
                            count += 1; feasible += success
                            speeds.append(float(a['action_applied'][:,0].mean()))
            windows.append(dict(interval=interval,all_episodes=all_count,all_feasible=all_feasible,
                                conditional_episodes=count,conditional_feasible=feasible,
                                conditional_rate=feasible/count if count else None,
                                mean_forward_command=float(np.mean(speeds)) if speeds else None))
        base, recent = windows
        enough = all(w['conditional_episodes'] >= gate['minimum_episodes_per_seed_interval'] for w in windows)
        passed = (enough and base['conditional_rate'] >= gate['minimum_safe_feasible_rate']
                  and recent['conditional_rate'] >= gate['minimum_safe_feasible_rate']
                  and recent['conditional_rate'] >= base['conditional_rate']-gate['maximum_rate_drop'])
        seeds.append(dict(seed=row['seed'],run_id=row['run_id'],windows=windows,passed=bool(passed)))
    if sorted(s['seed'] for s in seeds) != [0,1,2]:
        raise ValueError('stability gate requires all three full seeds')
    return dict(owner='NAV-064',split='train',updates=updates,
                verdict='PASS' if all(s['passed'] for s in seeds) else 'FAIL_LEARNING_STABILITY',
                seeds=seeds,limits='Additional train-only stop condition; original development/business denominators and acceptance unchanged.')
