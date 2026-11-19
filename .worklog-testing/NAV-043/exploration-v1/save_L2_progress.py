"""Snapshot completed original L1/L2 outputs; no replay, inference or simulation."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
RAW = Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')
OUT = ROOT / '.worklog-testing/NAV-043/exploration-v1/L2-completed-progress.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not OUT.exists()
sources = {}


def load(path):
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


rows = []
for level in ('L1_empty', 'L2_single_detour'):
    app = RAW / level
    load(app / 'FINISHED.json')
    exit_ = load(app / 'EXIT.json')
    assert exit_['exit_code'] == 0 and not exit_['timeout']
    load(app / 'physical_step_accounting.json')
    for method in ('uncalibrated', 'full'):
        trial = app / method
        decision = load(trial / 'decision.json')
        pin = load(trial / 'decision.pin.json')
        assert sources[str(trial / 'decision.json')] == pin['sha256']
        result = load(trial / 'result.json')
        nominal = load(trial / 'nominal.json')
        assert not result['rejected']
        assert result['z0_J'] == nominal['z0_J'] == decision['decision']['selected_budget_J']
        rows.append({'level': level, 'method': method,
                     'selected_k': decision['decision']['selected_k'],
                     **{k: result[k] for k in ('label', 't_s', 'safety_y', 'z0_J',
                         'measured_joint_work_J', 'internal_model_cost_J', 'safe',
                         'safe_timely_measured_budget_success', 'measured_budget_utilization',
                         'measured_overspend_J')},
                     'goal_distance_m': result['terminal']['goal_distance_m'],
                     'nominal': {k: nominal[k] for k in ('label', 't_s', 'cost_model_J', 'feasible')},
                     'full_saved_physics_event_review_pending': True})
for path, digest in sources.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
report = {'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
          'pane': os.environ.get('TMUX_PANE'), 'status': 'RAW_COMPLETED_L1_L2_NOT_FINAL_REVIEW',
          'rows': rows, 'source_hashes': sources,
          'new_applications': 0, 'new_episodes': 0, 'W_queries': 0, 'training_updates': 0,
          'NAV043_passed': False, 'G5_passed': False,
          'limitation': 'Saved online outcomes only; original full physical-event and applicability reviews remain pending.'}
with OUT.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print({'status': report['status'], 'rows': len(rows), 'sources': len(sources)}, flush=True)
