"""Audit one finished map from saved states only; no simulator or inference."""
from datetime import datetime
import os
from pathlib import Path
import sys

import review_saved as R

assert os.environ.get('TMUX_PANE') == '%599'
level = sys.argv[1]
assert level in ('L1_empty', 'L2_single_detour', 'L3_multi_path')
_, execution, manifest, prereg, _ = R.V2.context(initial=False)
out = R.V1.EXPECTED_ROOT / level
assert R.load(out/'EXIT.json')['exit_code'] == 0
assert (out/'FINISHED.json').is_file() and not (out/'ERROR.json').exists()
R.check_repetition_runtime(R.load(out/'runtime.json'))
assert all(p['enabled'] and max(p['errors'].values()) <= 1.e-4
           for p in R.load(out/'collider_geometry.json'))
source = R.load(R.NAV/'config/nav067_remaining_validation_proposal_v1.json')
old = Path(source['inputs']['manifest']['path']).parent
baseline = old / next(s for s in source['repetitions']['slots']
                      if s['level'] == level)['baseline_relative_directory']
map_row = next(row for row in manifest['maps'] if row['level'] == level)
comparisons = []
for method in ('uncalibrated', 'full'):
    pair = R.compare(baseline/method, out/method)
    ctx = R.E.build_rollout_ctx(method)
    spec = R.E.M.build(map_row['level'], map_row['seed'], ctx['design'])
    assert spec.to_dict() == map_row['spec']
    result = R.load(out/method/'result.json')
    steps = R.load(out/method/'substeps.json')
    initial = [R.PhysicalState(**s) for s in result['warmup']['state_at_window_start']]
    monitor = R.SubstepNavigation(spec, ctx['geom'], ctx['cm'], initial,
                                  [result['z0_J']], prereg)
    for row in steps:
        monitor.set_commands([row['applied']])
        state = row['state']
        monitor.advance([R.PhysicalState(*state[:7], tuple(state[7:]))],
                        physics_tick=row['physics_tick'])
        assert monitor.rows[-1] == row, (method, row['physics_tick'])
    assert monitor.terminal[0] == result['terminal']
    assert not monitor.active[0]
    comparisons.append({'method':method, 'physics_event_replay':'PASS_EVERY_SAVED_SUBSTEP',
                        'replayed_ticks':len(steps), **pair})
report = {'at':datetime.now().astimezone().isoformat(), 'level':level,
          'status':'SAVED_MAP_TRACE_RUNTIME_EVENT_REPLAY_PASS_NOT_TASK_CLOSURE',
          'comparisons':comparisons, 'new_applications':0, 'training_updates':0,
          'W_queries':0, 'NAV_067_status':False, 'NAV_043_executed':False}
R.write_new(R.HERE/('saved-map-review-'+level+'.json'), report)
print(R.json.dumps(report, ensure_ascii=False, indent=2), flush=True)
