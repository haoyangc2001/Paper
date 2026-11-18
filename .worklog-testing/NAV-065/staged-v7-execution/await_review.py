"""Wait for the one registered run, then invoke a read-only audit once."""
import os
from pathlib import Path
import socket
import time

assert socket.gethostname().split('.')[0] == 'tiger11' and os.environ.get('TMUX')
launch = Path('/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7')
while not (launch / 'EXIT.json').exists():
    if (launch / 'ERROR.json').exists():
        raise RuntimeError('Training launcher failed; reconcile ERROR.json without restarting')
    time.sleep(30)
from review import main
main()
