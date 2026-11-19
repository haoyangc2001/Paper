"""Snapshot all online outcomes; not the pending full-event review."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/home/caohy/repositories/Paper')
RAW = Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution-finished-progress.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not OUT.exists()
sources = {}


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


terminal = load(RAW / 'FINISHED.json')
assert terminal['applications'] == 3 and terminal['gpu_wall_s'] <= 10800
assert terminal['requires_saved_evidence_review']
protocol_path = ROOT / 'CaMRA/navigation/config/nav043_exploration_v1.json'
protocol = load(protocol_path)
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == Path(str(protocol_path) + '.sha256').read_text().strip()
for path, digest in protocol['sealed_hashes'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
rows = []
explicit_steps = queries = nominal_steps = measured_steps = 0
for level in protocol['order']:
    app = RAW / level
    load(app / 'FINISHED.json')
    assert load(app / 'EXIT.json')['exit_code'] == 0
    calls = load(app / 'physical_step_accounting.json')['calls']
    assert calls <= 20000
    explicit_steps += calls
    for method in protocol['methods']:
        trial = app / method
        result, nominal = load(trial / 'result.json'), load(trial / 'nominal.json')
        decision = load(trial / 'decision.json')
        assert sources[str(trial / 'decision.json')] == load(trial / 'decision.pin.json')['sha256']
        assert not result['rejected'] and result['z0_J'] == nominal['z0_J'] == decision['decision']['selected_budget_J']
        assert nominal['initial_state'] == decision['initial_state']
        assert decision['query_count'] == 5
        queries += decision['query_count']
        nominal_steps += len(nominal['trace'])
        measured_steps += result['physics_steps']
        rows.append({'level': level, 'method': method, 'selected_k': decision['decision']['selected_k'],
                     **{k: result[k] for k in ('label', 'safety_y', 't_s', 'z0_J',
                         'measured_joint_work_J', 'internal_model_cost_J',
                         'safe_timely_measured_budget_success', 'measured_budget_utilization')},
                     'terminal_goal_distance_m': result['terminal']['goal_distance_m'],
                     'nominal_label': nominal['label'], 'nominal_t_s': nominal['t_s'],
                     'nominal_cost_J': nominal['cost_model_J'], 'nominal_feasible': nominal['feasible']})
assert explicit_steps <= 60000 and queries == 30 and nominal_steps <= 2700
for path, digest in sources.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
report = {'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
          'pane': os.environ.get('TMUX_PANE'), 'status': 'EXECUTION_FINISHED_FULL_EVENT_REVIEW_PENDING',
          'allocation_terminal': terminal, 'rows': rows, 'initial_W_queries': queries,
          'explicit_physical_steps': explicit_steps, 'measurement_steps': measured_steps,
          'saved_nominal_steps': nominal_steps, 'source_hashes': sources,
          'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'sealed_sources_checked': len(protocol['sealed_hashes']),
          'new_applications': 0, 'new_episodes': 0, 'new_W_queries': 0, 'model_fits': 0,
          'NAV043_passed': False, 'G5_passed': False}
with OUT.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(report, ensure_ascii=False), flush=True)
