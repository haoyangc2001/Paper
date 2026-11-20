"""Isolated response-v4 candidate core; no production consumer or CLI fit mode.

q is a completed 0.1s SE(2) increment. Extra terms are engineering hypotheses
from exposed data, not measured deadzone or state-sufficiency guarantees.
"""
import numpy as np
import nav043_supported_response as old

base = old.base
FEATURE_VERSION = 'full-q-forward-curvature-turn-interaction-v1'
COLUMNS = 11


def features(commands):
    u = np.asarray(commands, dtype=float)
    original = old.features(u)
    return np.column_stack((original, u[:, 0]**2,
                            u[:, 0]*u[:, 2], u[:, 0]*np.abs(u[:, 2])))


def design(cases):
    xs, ys, ws = [], [], []
    for cohort in ('navigation', 'probe'):
        subset = [c for c in cases if c.cohort == cohort]
        if not subset:
            raise ValueError('both cohorts required')
        for case in subset:
            previous, commands, target = case.pairs()
            if len(target) == 0:
                raise ValueError('empty fitting case')
            xs.append(np.column_stack((previous, features(commands))))
            ys.append(target)
            ws.append(np.full(len(target), .5/len(subset)/len(target)))
    weights = np.concatenate(ws)
    root = np.sqrt(weights)
    return np.concatenate(xs)*root[:, None], np.concatenate(ys)*root[:, None], weights


def rank_support(x):
    singular = np.linalg.svd(x, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0]*1.e-12)) if singular[0] else 0
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    return dict(rank=rank, columns=COLUMNS, singular_values=singular.tolist(),
                condition=condition, valid=rank == COLUMNS and condition is not None and condition <= 1.e8)


def check_model(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3, COLUMNS) or not np.isfinite(theta).all():
        raise ValueError('finite 3x11 model required')
    eigenvalues = np.linalg.eigvals(theta[:, :3])
    radius = float(np.max(np.abs(eigenvalues)))
    if not np.isfinite(radius) or radius >= 1.:
        raise ValueError('unstable full-q memory; no clipping or retry')
    return dict(spectral_radius=radius,
                eigenvalues=[[float(v.real), float(v.imag)] for v in eigenvalues])


def solve_axis(x, y):
    """One least-squares axis solve. Formal callers require separate approval."""
    theta, residuals, rank, singular = np.linalg.lstsq(x, y, rcond=1.e-12)
    valid = (rank == COLUMNS and singular[-1] > 0 and singular[0]/singular[-1] <= 1.e8
             and theta.shape == (COLUMNS,) and np.isfinite(theta).all())
    return dict(theta=theta.tolist(), residual_sum=residuals.tolist(), rank=int(rank),
                singular_values=singular.tolist(), valid=bool(valid))


def free_replay(case, theta):
    theta = np.asarray(theta, dtype=float)
    check_model(theta)
    t = case.trace
    pose, response = t.initial.copy(), case.initial_q.copy()
    poses, responses = [], []
    for i, command in enumerate(t.interval_commands):
        response = theta[:, :3] @ response + theta[:, 3:] @ features(command[None])[0]
        if not np.isfinite(response).all():
            raise ValueError('nonfinite free response')
        responses.append(response.copy())
        for _ in range(min(20, t.ticks-i*20)):
            pose = base.step(pose, response, .005)
            poses.append(pose.copy())
    return np.asarray(poses), np.asarray(responses)
