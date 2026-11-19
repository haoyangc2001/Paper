"""Main-agent checks; this is not an independent subagent test."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/home/caohy/repositories/Paper')
NAV = ROOT / 'CaMRA/navigation'
HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

diagnosis = json.loads((NAV / 'reports/nav065_boundary_diagnosis_v1.json').read_bytes())
for path, expected in diagnosis['source_hashes'].items():
    assert digest(Path(path)) == expected, path
assert diagnosis['script_sha256'] == digest(NAV / 'scripts/analyze_nav065_boundary.py')
assert diagnosis['new_rollouts'] == diagnosis['optimizer_steps'] == diagnosis['threshold_changes'] == 0
assert diagnosis['saved_prediction_max_abs_difference'] == 0
for split, count, old, new in [('train', 540, 3, 13), ('development', 180, 3, 9)]:
    for version, expected_false in [('before', old), ('after', new)]:
        rows = [r for r in diagnosis['episode_rows'] if (r['split'],r['version']) == (split,version)]
        assert len(rows) == count == len({(r['run_id'],r['map_id'],r['k']) for r in rows})
        assert sum(r['false_feasible'] for r in rows) == expected_false
        for row in rows:
            assert abs(max(row['true_reach'], row['true_cost_margin_J']/550)-row['true_W']) < 1e-12
            assert abs(max(row['predicted_reach'], row['predicted_cost_margin_J']/550)-row['predicted_W']) < 1e-12
            assert row['false_feasible'] == (not row['actual_feasible'] and row['predicted_W'] <= 0)
print('DIAGNOSIS PASS: 540 train/180 development, unique paired keys, joint-value identity, 13/9 false-feasible starts')
assert not Path('/pub/data/caohy/work/camra_navigation/nav065_saved_refit_v2').exists()
assert json.loads((NAV / 'config/nav065_saved_refit_proposal_v2.json').read_bytes())['approved'] is False
assert not Path('/pub/data/caohy/work/camra_navigation/nav065_refit_v1/deployment_candidates_v2.json').exists()
print('STATE PASS: proposal unapproved, no new fit directory, no deployment bundle')

source = NAV / 'scripts/refit_nav065_saved.py'
isolated = HERE / 'isolated_saved_refit.py'
original = source.read_bytes()

def module():
    spec = importlib.util.spec_from_file_location('isolated_saved_refit',isolated)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def assertions(obj):
    row = {'kind':'rollout_result','run_id':'r','map_id':'train-a','k':1,'episode':{}}
    path = HERE / 'fixture/train_episodes.jsonl'
    path.parent.mkdir(exist_ok=True)
    inputs = SimpleNamespace(candidates=[{'run_id':'r'}])
    for label, rows in [('valid',[row]),('duplicate',[row,row]),('unknown',[{**row,'map_id':'development-a'}])]:
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        try:
            result = obj.saved_train(inputs,[{'map_id':'train-a'}],[1],path.parent,
                                     {'train_journal_sha256':digest(path)})
        except ValueError as exc:
            assert label != 'valid' and str(exc) == 'duplicate or unknown saved train key', (label,str(exc))
        else:
            assert label == 'valid' and len(result['r']) == 1, label + ' guard bypassed'

isolated.write_bytes(original)
assertions(module())
print('MUTATION ORIGINAL PASS: valid input accepted; duplicate/unknown train keys rejected')
mutant = original.decode().replace('if rid not in raw or key not in expected or key in raw[rid]:', 'if False:', 1)
assert mutant != original.decode()
(HERE / 'mutation.diff').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True), mutant.splitlines(True),fromfile='original',tofile='mutant')))
isolated.write_text(mutant)
try:
    assertions(module())
except AssertionError as exc:
    print('MUTANT KILLED by targeted assertion:', exc)
else:
    raise AssertionError('mutant survived')
isolated.write_bytes(original)
assert digest(source) == digest(isolated)
assertions(module())
print('MUTATION RESTORED PASS: isolated source restored byte-for-byte; production untouched')

before=json.loads((ROOT/'.worklog-testing/NAV-065/before-diagnosis-20261118.json').read_bytes())
after=json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_bytes())
assert [t['id'] for t in before['tasks']] == [t['id'] for t in after['tasks']]
for old, new in zip(before['tasks'], after['tasks']):
    assert old['status'] == new['status']
    assert old['work_records'] == new['work_records'][:len(old['work_records'])]
    if old['id'] != 'NAV-065':
        assert old == new
    else:
        assert old['acceptance_criteria'] == new['acceptance_criteria']
assert sum(t['status'] for t in after['tasks']) == 52
print('HISTORY PASS: 67 IDs/order/statuses and old records preserved; 52 complete, NAV-065 false')
