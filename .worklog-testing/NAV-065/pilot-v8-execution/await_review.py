"""Await this single pilot and audit its artifacts once without restarting it."""
import os
from pathlib import Path
import socket
import time

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
launch = Path('/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8')
while not (launch / 'EXIT.json').exists():
    if (launch / 'ERROR.json').exists():
        raise RuntimeError('Inspect ERROR.json; automatic restart is prohibited')
    time.sleep(15)
from review import main
main()
