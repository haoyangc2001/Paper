#!/usr/bin/env python3
"""快速探针：r12/r13 的 chunk 里有多少回合真正进入目标集（g<0）。"""
import os
import sys

NAV = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(NAV, "..", "calibration", "src"))

import numpy as np
from camra_navigation.algorithm import storage as S

WORK = "/pub/data/caohy/work/camra_navigation"
RUNS = sys.argv[1:] or [
    "nav064_r12_uncalibrated_s0",
    "nav064_r12_full_s0",
    "nav064_r13_uncalibrated_s0",
    "nav064_r13_full_s0",
]

for rid in RUNS:
    run_dir = os.path.join(WORK, rid)
    if not os.path.isdir(run_dir):
        print(rid, "MISSING")
        continue
    envs = sorted(e for e in os.listdir(run_dir) if e.startswith("rollout_env"))
    n_ep = n_arr = 0
    labels = {}
    for env_name in envs[:6]:            # 前 6 个环境足够判断
        store = S.RolloutStore(run_dir, env_name, obs_dim=41)
        n = store.n_chunks
        idxs = np.linspace(0, n - 1, min(n, 40)).round().astype(int) if n else []
        for idx in idxs:
            arrays = store.load_chunk(int(idx))
            meta = store.entries[int(idx)]
            kind = str(meta.get("terminal_kind"))
            labels[kind] = labels.get(kind, 0) + 1
            n_ep += 1
            d_last = float(np.asarray(arrays["goal_distance_m_next"])[-1])
            d_min = float(np.min(np.asarray(arrays["goal_distance_m"])))
            if d_last <= 0.5 or d_min <= 0.5:
                n_arr += 1
    print(f"{rid}: episodes={n_ep} with_goal_steps={n_arr} "
          f"({100.0 * n_arr / max(n_ep, 1):.1f}%) kinds={labels}")
