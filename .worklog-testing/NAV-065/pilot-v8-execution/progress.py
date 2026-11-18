"""Read a complete logged pilot step without touching models or optimizers."""
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
here = Path(__file__).resolve().parent
launch = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8')
out = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_v8')
rows = []
with (out / 'nav064_terminal_tail_v1_full_s0.jsonl').open() as stream:
    for line in stream:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row['kind'] in ('train_gate', 'trial'):
            rows.append(row)
first, latest = rows[0], rows[-1]
report = {'trial': latest['trial'], 'stage': latest.get('stage', 'initial'),
    'phase_counts': dict(Counter(r['stage'] for r in rows if r['kind'] == 'trial')),
    'initial_confusion': dict(Counter(s['classification'] for s in first['starts'])),
    'latest_confusion': dict(Counter(s['classification'] for s in latest['starts'])),
    'initial_additional_metrics': first['report']['additional_metrics'],
    'latest_additional_metrics': latest['report']['additional_metrics'],
    'original_numeric': latest['report']['original_grid']['metrics'],
    'full_train_pass': latest['report']['pass'],
    'launcher_finished': (launch / 'EXIT.json').exists(), 'launcher_error': (launch / 'ERROR.json').exists()}
destination = here / sys.argv[1]
assert destination.parent == here
with destination.open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(report, indent=2))
