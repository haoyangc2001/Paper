"""Acceptance for the overspend-only cost mask (CM1-CM4)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAV = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(NAV, "..", "calibration", "src"))

from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.algorithm import targets as T  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))
    print(("  OK   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))


def main() -> int:
    # synthetic: T=4 steps, z starts positive then goes negative from step 2
    g_ext = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    c = np.array([0.01, 0.02, 0.03, 0.04])
    z = np.array([0.5, 0.2, -0.1, -0.3])
    w = 50.0

    # CM1: default (no mask) == old behaviour
    base = T.joint_training_sequence(g_ext, c, w)
    assert np.array_equal(base, g_ext + np.concatenate([w * c, [0.0]])), "CM1 baseline formula"
    check("CM1_default_matches_old_formula",
          bool(np.allclose(base, [1.0 + w * 0.01, 1.0 + w * 0.02,
                                  1.0 + w * 0.03, 1.0 + w * 0.04, 0.0])))

    # CM2: mask only keeps steps where z < 0
    masked = T.joint_training_sequence(g_ext, c, w, overspend_mask=(z < 0.0))
    expect = g_ext.copy()
    expect[2] += w * 0.03
    expect[3] += w * 0.04
    check("CM2_mask_only_after_overspend", bool(np.allclose(masked, expect)),
          f"got={np.round(masked,4).tolist()}")
    check("CM2_pre_overspend_untouched",
          bool(np.allclose(masked[:2], g_ext[:2])), "steps 0-1 unchanged")
    check("CM2_terminal_untouched", masked[4] == g_ext[4])

    # mask all ones == no mask; mask all zeros == no cost term
    check("CM2_mask_ones_equals_no_mask",
          bool(np.allclose(T.joint_training_sequence(g_ext, c, w, overspend_mask=np.ones(4)),
                           base)))
    check("CM2_mask_zeros_equals_no_cost",
          bool(np.allclose(T.joint_training_sequence(g_ext, c, w, overspend_mask=np.zeros(4)),
                           g_ext)))

    # CM4: length guard
    try:
        T.joint_training_sequence(g_ext, c, w, overspend_mask=np.ones(3))
        ok = False
    except ValueError:
        ok = True
    check("CM4_length_guard", ok)

    # CM1 at config level: default flag is False and w=0 short-circuits
    cfg = S.ObjectiveConfig  # dataclass; check the default field value
    import inspect
    defaults = {f.name: f.default for f in __import__("dataclasses").fields(cfg)}
    check("CM1_config_default_false", defaults.get("joint_cost_on_overspend_only") is False,
          str(defaults.get("joint_cost_on_overspend_only")))
    check("CM1_w_zero_returns_copy",
          bool(np.array_equal(T.joint_training_sequence(g_ext, c, 0.0), g_ext)))

    # CM5: prereg pinned
    p = os.path.join(NAV, "config", "objective_cost_mask_prereg_v1.json")
    pinned = open(p + ".sha256", encoding="utf-8").read().strip()
    import hashlib
    check("CM5_prereg_pin", hashlib.sha256(open(p, "rb").read()).hexdigest() == pinned)

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(json.dumps({"task": "NAV-064", "test": "cost_mask",
                      "verdict": "PASS" if n_fail == 0 else "FAIL",
                      "n_checks": len(checks), "n_failed": n_fail}, ensure_ascii=False))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
