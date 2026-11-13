"""Acceptance test for the frozen action_init_fix preregistration (v1.1.0).

Run on tiger11 from CaMRA/navigation:
    <frozen python> tests/test_action_init_fix.py
Covers AI1 (default-off regression), AI2, AI3, AI4 and AI6.  AI5 (fresh-policy
arrival examples) is exercised by scripts/t_verify_init_fix.py because it needs
the full development rollout; AI7 is the plan/ledger/docs records.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(NAV, "..", "calibration", "src"))

from camra_navigation.algorithm import networks as NW  # noqa: E402

NAV_CONTRACT = os.path.join(NAV, "config", "navigation_contract_v1.json")
PREREG = os.path.join(NAV, "config", "action_init_fix_prereg_v1.json")
PIN = PREREG + ".sha256"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))
    print(("  OK   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))


def main() -> int:
    nav = json.load(open(NAV_CONTRACT, encoding="utf-8"))
    prereg = json.load(open(PREREG, encoding="utf-8"))
    pinned = open(PIN, encoding="utf-8").read().strip()
    import hashlib
    actual = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    check("prereg_sha256_matches_pin", actual == pinned, actual[:16])
    p = prereg["change"]["parameters"]
    check("prereg_version", prereg["version"] == "1.1.0", prereg["version"])

    # AI1: default-off is bitwise identical to the old signature
    torch.manual_seed(7)
    a = NW.build(nav, action_bounded=True)
    torch.manual_seed(7)
    b = NW.build(nav, action_bounded=True, initial_forward_speed=0.0,
                 log_std_min=None)
    same = all(torch.equal(x, y) for x, y in
               zip(a.state_dict().values(), b.state_dict().values()))
    o = torch.zeros(1, a.policy.cfg.obs_dim)
    with torch.no_grad():
        same_out = torch.equal(a.policy.forward(o).mean,
                               b.policy.forward(o).mean)
    check("AI1_default_off_bitwise", same and same_out)
    check("AI1_default_bias_zero", abs(float(a.policy.head.bias.data[0])) < 1e-9)
    check("AI1_default_floor_is_contract",
          abs(a.policy.log_std_min - NW.LOG_STD_MIN) < 1e-12,
          str(a.policy.log_std_min))

    # AI2: calibration
    torch.manual_seed(0)
    on = NW.build(nav, action_bounded=True,
                  initial_forward_speed=p["initial_forward_speed"],
                  log_std_min=p["log_std_min"])
    b0 = float(on.policy.head.bias.data[0])
    check("AI2_bias_calibration", abs(math.tanh(b0) - 0.465) < 1e-6,
          f"tanh(bias)={math.tanh(b0):.8f}")
    check("AI2_default_bias_unchanged",
          abs(float(NW.build(nav, action_bounded=True).policy.head.bias.data[0])) < 1e-9)

    # AI3: exploration floor
    with torch.no_grad():
        on.policy.log_std.fill_(-4.0)
    on.policy.clamp_log_std()
    check("AI3_floor_enforced", float(on.policy.log_std.min()) >= p["log_std_min"] - 1e-6,
          f"min={float(on.policy.log_std.min()):.4f}")
    with torch.no_grad():
        a.policy.log_std.fill_(-4.0)
    a.policy.clamp_log_std()
    check("AI3_default_floor_unchanged", float(a.policy.log_std.min()) < -1.5,
          f"min={float(a.policy.log_std.min()):.4f}")

    # AI4: distribution self-consistency
    torch.manual_seed(0)
    c = NW.build(nav, action_bounded=True,
                 initial_forward_speed=p["initial_forward_speed"],
                 log_std_min=p["log_std_min"])
    dist = c.policy.forward(torch.zeros(1, c.policy.cfg.obs_dim))
    smp = dist.sample((4000,))
    lp = dist.log_prob(smp)
    check("AI4_log_prob_finite", bool(torch.isfinite(lp).all()))
    check("AI4_actions_strictly_bounded", bool((smp.abs() < 1.0).all()))
    check("AI4_entropy_finite", math.isfinite(float(dist.entropy().mean())))

    # AI6: existing switches untouched (consumer reads from prereg, default off)
    cpath = os.path.join(NAV, "scripts", "train_formal.py")
    esrc = open(cpath, encoding="utf-8").read()
    check("AI6_consumer_reads_prereg",
          'prereg.get("action_init_fix", {})' in esrc
          and 'prereg.get("action_bounded", False)' in esrc
          and 'prereg.get("budget_accounting", "nominal")' in esrc)

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(json.dumps({"task": "NAV-064", "test": "action_init_fix",
                      "verdict": "PASS" if n_fail == 0 else "FAIL",
                      "n_checks": len(checks), "n_failed": n_fail},
                     ensure_ascii=False))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
