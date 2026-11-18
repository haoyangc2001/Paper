"""Read journals only; never run training or score development."""
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import socket

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
OUT = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_v7')
result = {'time': datetime.now().astimezone().isoformat(), 'candidates': []}
for path in sorted(OUT.glob('*.jsonl')):
    initial = last = None
    end = False
    stages = Counter()
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['kind'] == 'train_gate':
                initial = row
            elif row['kind'] == 'trial':
                last = row
                stages[row['stage']] += 1
            elif row['kind'] == 'fit_finished':
                end = True
    if last:
        gate = last['report']
        starts = Counter(s['classification'] for s in last['starts'])
        result['candidates'].append({'run_id': path.stem, 'finished': end,
            'stage_steps': dict(stages), 'false_feasible': starts['false_feasible'],
            'false_infeasible': starts['false_infeasible'], 'pass': gate['pass'],
            'original_pass': gate['original_grid']['pass'],
            'additional_reach_relative_mae': gate['additional_metrics']['vh']['relative_mae'],
            'selected_sign_mismatches': gate['selected_sign_mismatches'],
            'failures': gate['failures']})
result['complete'] = (OUT / 'train_gate.json').exists()
print(json.dumps(result, ensure_ascii=False, indent=2))
