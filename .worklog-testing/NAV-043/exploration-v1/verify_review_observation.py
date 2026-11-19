"""Exercise the real read-only reviewer expression on saved initial vectors."""
import ast
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
HERE = Path(__file__).resolve().parent
RAW = Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')
SCRIPT = ROOT / 'CaMRA/navigation/scripts/audit_nav043_exploration.py'
PRIOR = HERE / 'audit_nav043_exploration_pre_flat_observation.py'
OUT = HERE / 'review-observation-basic.json'
assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
assert sys.executable == '/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python'
assert not OUT.exists()
sources = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    raw = Path(path).read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


assert sha(PRIOR) == 'd56a8c86d293eda3fe7bfc7965d7bb37925f22d14deaca43cec8ef9eca4df42b'
old, new = PRIOR.read_text(), SCRIPT.read_text()
needle = "load(trial/'navigation.json')[0]['observation'][0]"
replacement = "load(trial/'navigation.json')[0]['observation']"
assert old.count(needle) == 1 and new == old.replace(needle, replacement)


def expression(source, name):
    compile(source, name, 'exec')
    nodes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call)
             and 'navigation.json' in ast.unparse(node.value)
             and 'np.testing.assert_array_equal' in ast.unparse(node.value)]
    assert len(nodes) == 1
    return compile(ast.Expression(nodes[0].value), name, 'eval')


old_expression = expression(old, str(PRIOR))
new_expression = expression(new, str(SCRIPT))
rows = []
for level in ('L1_empty', 'L2_single_detour'):
    for method in ('uncalibrated', 'full'):
        trial = RAW / level / method
        saved = load(trial / 'decision.json')
        assert sha(trial / 'decision.json') == load(trial / 'decision.pin.json')['sha256']
        chosen = saved['decision']['selected_index']
        observation = load(trial / 'navigation.json')[0]['observation']
        assert np.asarray(observation).shape == (61,)
        assert np.asarray(saved['query_observations'][chosen]).shape == (61,)
        namespace = {'np': np, 'load': load, 'trial': trial, 'saved': saved, 'chosen': chosen}
        try:
            eval(old_expression, namespace)
        except AssertionError:
            pass
        else:
            raise AssertionError('old scalar/vector comparison unexpectedly accepted')
        eval(new_expression, namespace)
        changed = copy.deepcopy(saved)
        changed['query_observations'][chosen][59] += .001
        try:
            eval(new_expression, {**namespace, 'saved': changed})
        except AssertionError:
            pass
        else:
            raise AssertionError('changed budget observation was not rejected')
        rows.append({'level': level, 'method': method, 'shape': [61],
                     'old_expression_rejected_equal_full_vectors': True,
                     'new_expression_passed_saved_full_vector': True,
                     'mismatched_budget_observation_rejected': True})
protocol = load(ROOT / 'CaMRA/navigation/config/nav043_exploration_v1.json')
for path, digest in protocol['sealed_hashes'].items():
    assert sha(path) == digest, path
for path, digest in sources.items():
    assert sha(path) == digest, path
report = {'at': datetime.now().astimezone().isoformat(), 'host': socket.gethostname(),
          'pane': os.environ.get('TMUX_PANE'), 'status': 'PASS_SAVED_INITIAL_OBSERVATION_COMPARE',
          'prior_sha256': sha(PRIOR), 'reviewer_sha256': sha(SCRIPT),
          'verification_sha256': sha(__file__), 'rows': rows, 'source_hashes': sources,
          'sealed_sources_checked': len(protocol['sealed_hashes']),
          'full_event_review_performed': False, 'new_applications': 0, 'new_episodes': 0,
          'W_queries': 0, 'training_updates': 0, 'NAV043_passed': False, 'G5_passed': False}
with OUT.open('x') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(report, ensure_ascii=False), flush=True)
