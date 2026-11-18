"""Run real train-only diagnosis with fitting/development explicitly disabled."""
from pathlib import Path
import sys

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0, str(ROOT / 'CaMRA/navigation/scripts'))
import probe_nav065_ridge as G

def forbidden(*args, **kwargs):
    raise AssertionError('Fitting, rollout and development are forbidden in preparation')

G.solve_head = forbidden
G.np.linalg.lstsq = forbidden
G.torch.optim.Adam.step = forbidden
G.torch.optim.SGD.step = forbidden
G.R.evaluate = forbidden
if hasattr(G.R, 'rollout'):
    G.R.rollout = forbidden
sys.argv = ['probe_nav065_ridge.py', '--diagnose', str(G.R.NAV / 'reports/nav065_ridge_preflight_v11.json')]
G.main()
print('READONLY_PREFLIGHT_FINISHED: real candidate solves/optimizer/rollouts/development all forbidden')
