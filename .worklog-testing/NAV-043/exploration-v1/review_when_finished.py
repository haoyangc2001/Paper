"""Wait for the existing allocation, then run one read-only review. No retry."""
from pathlib import Path
import os
import subprocess
import sys
import time

assert os.environ.get('TMUX')
root=Path('/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1')
deadline=time.monotonic()+10800
while not any((root/name).exists() for name in ('FINISHED.json','HELD.json')):
    if time.monotonic()>=deadline:raise TimeoutError('existing allocation terminal absent')
    time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,'-B',
    '/home/caohy/repositories/Paper/CaMRA/navigation/scripts/audit_nav043_exploration.py']))
