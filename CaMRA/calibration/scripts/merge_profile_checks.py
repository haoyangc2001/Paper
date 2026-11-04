#!/usr/bin/env python3
"""合并各 profile 的检查 JSON 为 summary.json（纯标准库，不进仿真）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from camra_calibration.profiles import PROFILE_IDS, knob_diff  # noqa: E402


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/pub/data/caohy/logs/camra/profile_checks")
    results = {}
    for pid in PROFILE_IDS:
        f = out_dir / f"{pid}.json"
        if not f.is_file():
            print(f"MISSING {f}")
            return 4
        results[pid] = json.loads(f.read_text(encoding="utf-8"))
    clean_cfg = results["clean"]["cfg_report"]
    cfg_diff = {pid: {k: {"clean": clean_cfg[k], pid: results[pid]["cfg_report"][k]}
                      for k in results[pid]["cfg_report"]
                      if results[pid]["cfg_report"][k] != clean_cfg[k]}
                for pid in PROFILE_IDS}
    per_ok = {pid: results[pid]["readback"]["all_ok"] for pid in PROFILE_IDS}
    summary = {
        "task": results["clean"].get("task", "CaMRA-Go2-Velocity"),
        "seeds": [results["clean"]["applied"][0]["seed"], results["clean"]["applied"][1]["seed"]],
        "knob_diff_matrix": knob_diff(),
        "cfg_level_diff_vs_clean": cfg_diff,
        "per_profile_readback_ok": per_ok,
        "overall_ok": all(per_ok.values()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True))
    return 0 if summary["overall_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
