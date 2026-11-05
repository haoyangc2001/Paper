"""静态资产审计（HIRA-012，不启动仿真）。

    python tests/audit_asset_static.py --out reports/asset_actuator_audit_static.json

同时是 ``check_asset_runtime.py`` 的前置门禁：静态检查不过就没必要开 App，
省得把"URDF 本身就不对"误当成"导入出错"。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
ADAPTER = REPO / "CaMRA" / "locomotion" / "adapters"
sys.path.insert(0, str(ADAPTER))

from hira_go2.asset_audit import static_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="HiRA asset static audit")
    parser.add_argument("--urdf", default=str(REPO / "HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf"))
    parser.add_argument("--contract", default=str(EXP / "config/hira_source_contract_v1.json"))
    parser.add_argument("--out", default=str(EXP / "reports/asset_actuator_audit_static.json"))
    args = parser.parse_args()

    report = static_audit(args.urdf, args.contract)
    checks = report["contract_crosscheck"]["checks"]
    failed = [item["id"] for item in checks if not item["ok"]]
    defaults = report["contract_crosscheck"]["default_angles"]

    payload = {
        "report": "asset_actuator_audit_static",
        "task": "HIRA-012",
        **report,
        "verdict": {
            "all_passed": report["contract_crosscheck"]["all_passed"] and defaults["all_inside"],
            "failed_checks": failed,
            "default_angles_outside": defaults["outside"],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[static] checks={len(checks)} failed={failed}")
    print(f"[static] dof={report['source']['urdf']['n_dof']} total_mass={report['source']['urdf']['total_mass']:.4f}")
    print(f"[static] effort_values={report['source']['distinct_effort_values']} "
          f"velocity_values={report['source']['distinct_velocity_values']}")
    print(f"[static] default_angles_inside_limits={defaults['all_inside']}")
    print("ASSET_STATIC_OK" if payload["verdict"]["all_passed"] else "ASSET_STATIC_FAIL")
    return 0 if payload["verdict"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
