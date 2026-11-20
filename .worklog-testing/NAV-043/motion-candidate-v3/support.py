"""Read-only input support/rank audit. No response coefficient solve."""
import os
import sys
from pathlib import Path

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT/'CaMRA/navigation/scripts'))
import nav043_supported_response as c

c.context()
cases = c.load_cases()
rows = []
for level, train, held in c.folds(cases):
    axes = {}
    for axis, name in enumerate(c.base.AXES):
        x, y, weights = c.design(train, axis)
        axes[name] = dict(c.rank_support(x), rows=len(y), weight_sum=float(weights.sum()))
    rows.append(dict(level=level, training=[i.key for i in train],
                     heldout=[i.key for i in held], axes=axes))
c.base.write_new(c.HERE/'support.json', dict(status='SUPPORT_AUDIT_ONLY_NO_FITS',
    pane=os.environ['TMUX_PANE'], rows=rows, cases=len(cases),
    old_navigation_ticks=sum(i.trace.ticks for i in cases if i.cohort=='navigation'),
    probe_measurement_ticks=sum(i.trace.ticks for i in cases if i.cohort=='probe'),
    support_pass=all(a['valid'] for r in rows for a in r['axes'].values()),
    actual_model_solves=0, applications=0, new_episodes=0, W_queries=0,
    old_strict_HELD_preserved=True, production_enabled=False))
print('SUPPORT', [(r['level'], {k:(v['rank'],v['condition']) for k,v in r['axes'].items()}) for r in rows])
