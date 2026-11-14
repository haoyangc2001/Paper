"""Cross-checks for the r11 X-ray: make sure the correlations are not artefacts.

1. Is the full-variant reach critic REALLY inverted, or is it a low-variance
   correlation artefact?  Print mean V_h per distance bucket per phase.
2. Is the full variant's h saturated at 1 (safety term), as r6 once showed?
3. How sparse are arrival samples, and how many of them are feasible?
4. Does the main arm lose arrival samples over training?
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import diag_xray_signal as X  # noqa: E402  (repo layout)
except ImportError:  # pragma: no cover - temporary /tmp layout
    sys.path.insert(0, "/tmp")
    import nav064_xray as X  # noqa: E402

WORK = X.WORK
FIELDS = ["goal_distance_m", "goal_distance_m_next", "value_reach", "obj_h",
          "obj_g", "terminated", "obj_feasible", "obj_stop_now", "action_raw"]


def load_phase(run_id):
    out = {0: [], 1: [], 2: [], 3: []}
    for env_i, idx, last, p in X.chunk_paths(run_id):
        ph = min(3, int(4 * idx / max(1, last + 1)))
        with np.load(p) as z:
            out[ph].append({f: np.asarray(z[f]) for f in FIELDS})
    return {ph: {f: np.concatenate([d[f] for d in chunks])
                 for f in FIELDS} for ph, chunks in out.items() if chunks}


EDGES = [0, 3, 6, 9, 12, 100]

for tag, rid in X.RUNS:
    print(f"\n######## {tag} ({rid}) ########")
    for ph, d in sorted(load_phase(rid).items()):
        dist = d["goal_distance_m"]
        vh = d["value_reach"]
        h = d["obj_h"]
        arr = d["terminated"] & (d["goal_distance_m_next"] <= 0.5)
        print(f"-- P{ph}: n={len(dist)} dist_std={dist.std():.3f} "
              f"h_mean={h.mean():.3f} h_frac_ge_0.99={np.mean(h >= 0.99):.3f} "
              f"h_std={h.std():.3f} | Vh_std={vh.std():.3f} "
              f"arrivals={int(arr.sum())} feasible_at_arrival="
              f"{(float(d['obj_feasible'][arr].mean()) if arr.any() else float('nan')):.3f} "
              f"stop0={d['obj_stop_now'].mean():.3f} vx={d['action_raw'][:,0].mean():+.3f}")
        means = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (dist >= lo) & (dist < hi)
            means.append(f"{lo:>2}-{hi:<3}:{vh[m].mean():+.3f}(n={m.sum()})"
                         if m.sum() > 20 else f"{lo:>2}-{hi:<3}:n/a")
        print("   V_h by distance:", "  ".join(means))
