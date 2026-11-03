#!/usr/bin/env python3
"""CAL-011 演示与留痕：四个命令族的名义轨迹、散列与对账指标。

输出一个 JSON，两次独立进程运行应产生完全相同的文件（逐位可复现）。
只用标准库。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from camra_calibration.commands import FAMILIES, build_program  # noqa: E402
from camra_calibration.nominal import (  # noqa: E402
    CROSSCHECK_TOL,
    PlanarState,
    integrate,
    integrate_rk4,
    max_planar_deviation,
    wrap_angle,
)
from camra_calibration.protocol import Protocol  # noqa: E402
from camra_calibration.timeline import Timeline  # noqa: E402

SEED = 6000
INIT = PlanarState(0.25, -0.30, 0.40)


def trajectory_digest(traj) -> str:
    blob = json.dumps([[repr(s.px), repr(s.py), repr(s.yaw)] for s in traj], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    proto = Protocol.load()
    tl = Timeline.from_protocol(proto)
    problems = tl.check()

    report = {
        "protocol_version": proto.version,
        "protocol_sha256": proto.sha256,
        "timeline": {
            "physics_intervals": tl.physics_intervals,
            "policy_intervals": tl.policy_intervals,
            "command_intervals": tl.command_intervals,
            "pairing_points": tl.pairing_points,
            "warmup_intervals": tl.warmup_intervals,
            "warmup_s": tl.warmup_s,
            "window_s": tl.measured_window_s,
            "ratios": tl.ratios(),
            "problems": problems,
        },
        "initial_state": list(INIT.as_tuple()),
        "seed": args.seed,
        "families": {},
    }
    for fam in FAMILIES:
        prog = build_program(fam, args.seed)
        exact = integrate(INIT, prog.commands, tl.command_dt)
        rk4 = integrate_rk4(INIT, prog.commands, tl.command_dt, substeps=50)
        dev = max_planar_deviation(exact, rk4)
        report["families"][fam] = {
            "program_sha256": prog.sha256(),
            "n_command_points": len(prog.commands),
            "low_speed_share_strict": round(prog.low_speed_share(True), 6),
            "low_speed_share_literal": round(prog.low_speed_share(False), 6),
            "trajectory_sha256": trajectory_digest(exact),
            "final": {
                "px": exact[-1].px, "py": exact[-1].py,
                "yaw_raw": exact[-1].yaw, "yaw_wrapped": wrap_angle(exact[-1].yaw),
                "path_length_m": round(sum(
                    math.hypot(exact[i + 1].px - exact[i].px, exact[i + 1].py - exact[i].py)
                    for i in range(len(exact) - 1)), 6),
            },
            "crosscheck": {"rk4_max_planar_dev": dev, "tol": CROSSCHECK_TOL,
                           "pass": dev < CROSSCHECK_TOL},
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"timeline problems: {problems}")
    for fam, d in report["families"].items():
        print(f"  {fam:<17} prog {d['program_sha256'][:12]}  traj {d['trajectory_sha256'][:12]} "
              f"rk4dev {d['crosscheck']['rk4_max_planar_dev']:.2e}  low {d['low_speed_share_strict']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
