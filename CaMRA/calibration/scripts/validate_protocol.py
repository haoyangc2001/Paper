#!/usr/bin/env python3
"""校准协议的机器校验入口（CAL-009）。

校验 `config/calibration_protocol_v<ver>.json` 的内部一致性，并可选比对冻结散列。
只用标准库，不导入 torch / isaaclab，因此在任何机器上都能跑。

用法：
    python scripts/validate_protocol.py                       # 校验默认协议
    python scripts/validate_protocol.py --protocol <path>      # 校验指定文件
    python scripts/validate_protocol.py --write-hash           # 写/刷新 .sha256
    python scripts/validate_protocol.py --check-hash           # 仅比对已冻结散列

退出码：0 通过；2 校验失败；3 散列不匹配；4 文件缺失。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE.parent / "config" / "calibration_protocol_v1.0.0.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks += 1
        if not ok:
            self.failures.append(f"{name}: {detail}" if detail else name)

    def eq(self, name: str, actual, expected) -> None:
        self.check(name, actual == expected, f"expected {expected!r}, got {actual!r}")

    def close(self, name: str, actual, expected, tol: float = 1e-9) -> None:
        try:
            ok = abs(float(actual) - float(expected)) <= tol
        except (TypeError, ValueError):
            ok = False
        self.check(name, ok, f"expected {expected}, got {actual}")


def validate(p: dict, path: Path) -> Checker:
    c = Checker()

    # ---- 1. 标识与状态 ----
    c.eq("schema_version", p.get("schema_version"), "1.0.0")
    c.eq("protocol_id", p.get("protocol_id"), "camra-calibration-protocol")
    c.eq("state", p.get("state"), "frozen")
    c.eq("owner_task", p.get("owner_task"), "CAL-009")
    version = p.get("version", "")
    c.check("version_matches_filename", path.stem.endswith(f"v{version}"),
            f"file {path.name} vs version {version}")
    c.eq("unresolved_empty", p.get("unresolved"), [])

    # ---- 2. 时钟一致性 ----
    k = p.get("clocks", {})
    c.close("policy_hz = 1/policy_dt", k.get("policy_hz"), 1.0 / k.get("policy_dt_s", 0), 1e-6)
    c.close("command_hz = 1/command_dt", k.get("command_hz"), 1.0 / k.get("command_dt_s", 0), 1e-6)
    c.close("pairing_hz = 1/pairing_dt", k.get("pairing_hz"), 1.0 / k.get("pairing_dt_s", 0), 1e-6)
    c.close("sim_hz = 1/sim_dt", k.get("sim_hz"), 1.0 / k.get("sim_dt_s", 0), 1e-6)
    c.eq("decimation = policy_dt/sim_dt",
         round(k.get("policy_dt_s", 0) / k.get("sim_dt_s", 1)), k.get("decimation"))
    c.eq("measured_steps = window/sim_dt",
         round(k.get("measured_window_s", 0) / k.get("sim_dt_s", 1)), k.get("measured_steps"))
    c.eq("pairing_points = window/pairing_dt + 1",
         round(k.get("measured_window_s", 0) / k.get("pairing_dt_s", 1)) + 1,
         k.get("pairing_points_including_endpoints"))
    c.eq("energy_hz == sim_hz", k.get("energy_integration_hz"), k.get("sim_hz"))
    c.eq("warmup_positive", float(k.get("warmup_s", 0)) > 0, True)

    # ---- 3. 命令族 / profile / cell 数 ----
    fam = p.get("command_program", {}).get("families", [])
    prof = p.get("profiles", [])
    fam_ids = [f.get("id") for f in fam]
    prof_ids = [x.get("id") for x in prof]
    c.eq("family_count", len(fam_ids), 4)
    c.eq("family_ids_unique", len(set(fam_ids)), len(fam_ids))
    c.eq("profile_count", len(prof_ids), 5)
    c.eq("profile_ids_unique", len(set(prof_ids)), len(prof_ids))
    c.eq("profile_ids_expected", prof_ids, ["clean", "friction_only", "mass_only", "push_only", "combined"])
    split = p.get("split", {})
    c.eq("cells = families x profiles", split.get("cells"), len(fam_ids) * len(prof_ids))
    c.eq("cells == 20", split.get("cells"), 20)
    for layer in fam:
        c.check(f"family[{layer.get('id')}] has description", bool(layer.get("description")))

    # ---- 4. 配额自洽 ----
    for name, total in (("baseline", 400), ("calibration", 800), ("heldout", 400), ("near_obstacle", 200)):
        d = split.get(name, {})
        c.eq(f"{name}.total", d.get("total"), total)
        c.eq(f"{name}.per_cell x cells", d.get("per_cell", 0) * split.get("cells", 0), d.get("total"))
    base = split.get("baseline", {})
    sub = base.get("subsets", {})
    c.eq("baseline.subsets sum", sum(sub.values()) if sub else -1, base.get("total"))
    c.eq("baseline.profile clean_only", base.get("profile"), "clean only")
    c.eq("pilot.total", split.get("pilot", {}).get("total"), 20)
    for name in ("calibration", "heldout", "near_obstacle"):
        c.check(f"{name}.consumed_by non-empty", bool(split.get(name, {}).get("consumed_by")))
    c.eq("heldout.may_tune", split.get("heldout", {}).get("may_tune"), False)
    c.eq("near_obstacle.may_tune", split.get("near_obstacle", {}).get("may_tune"), False)

    # ---- 5. 种子：覆盖配额、互斥、避开保留值 ----
    seeds = p.get("seeds", {})
    blocks = seeds.get("blocks", {})
    reserved = set(seeds.get("reserved_excluded", []))
    used: dict[int, str] = {}
    overlap_blocks: list[str] = []
    hit_reserved: list[str] = []
    for bid, spec in blocks.items():
        start, count = spec.get("start"), spec.get("count")
        c.check(f"seed_block[{bid}] wellformed", isinstance(start, int) and isinstance(count, int) and count > 0)
        for s in range(start, start + count):
            if s in used:
                overlap_blocks.append(f"{bid}@{s} clashes {used[s]}")
            used[s] = bid
            if s in reserved:
                hit_reserved.append(f"{bid}@{s}")
    c.eq("seed_blocks_disjoint", overlap_blocks[:5], [])
    c.eq("seed_blocks_avoid_reserved", hit_reserved[:5], [])
    expected_blocks = {
        "baseline_fit": sub.get("fit"), "baseline_dev": sub.get("dev"),
        "baseline_validate": sub.get("validate"), "pilot": split.get("pilot", {}).get("total"),
        "calibration": split.get("calibration", {}).get("total"),
        "heldout": split.get("heldout", {}).get("total"),
        "near_obstacle": split.get("near_obstacle", {}).get("total"),
    }
    for bid, want in expected_blocks.items():
        c.eq(f"seed_block[{bid}].count == quota", blocks.get(bid, {}).get("count"), want)
    c.eq("total_seeds == 1820", len(used), 1820)

    # ---- 6. 能量口径 ----
    e = p.get("energy", {})
    c.eq("energy.metric_id", e.get("metric_id"), "D-ENERGY-001")
    c.eq("energy.family", e.get("family"), "positive_joint_mechanical_work")
    c.eq("energy.unit", e.get("unit"), "J")
    c.check("energy.exec_formula has max(...,0)", "max(" in e.get("exec_formula", ""))
    c.check("energy.exec_formula uses 6000 steps", "6000" in e.get("exec_formula", ""))
    c.check("energy.qdot_source mentions robot.update",
            "robot.update" in e.get("qdot_source", ""), "必须声明逐子步刷新，否则复现 50Hz 旧值")
    c.check("energy.qdot_timing post", e.get("qdot_timing") == "post")
    c.check("energy.tau_timing post", e.get("tau_timing") == "post")
    nz = e.get("near_zero", {})
    c.eq("near_zero.rule", nz.get("rule"), "absolute_threshold_single_column")
    c.eq("near_zero.j_min_J", nz.get("j_min_J"), 1.0)
    c.check("near_zero forbids epsilon", "epsilon" in nz.get("forbidden", ""))
    c.check("near_zero reopen trigger", bool(nz.get("reopen_trigger")))
    eps = e.get("epsilon", {})
    c.check("epsilon_c references j_min", "1.0 J" in eps.get("epsilon_c", ""))
    c.check("epsilon_d on 10 Hz grid", "10 Hz" in eps.get("epsilon_d", ""))
    c.check("epsilon_d sensitivity kept", bool(eps.get("epsilon_d_sensitivity")))

    # ---- 7. 统计口径 ----
    st = p.get("statistics", {})
    c.eq("quantile_level", st.get("quantile_level"), 0.9)
    c.check("quantile nearest-rank", "nearest-rank" in st.get("quantile_method", ""))
    c.check("alpha non-negative floor", "max(0" in st.get("alpha", ""))
    iv = st.get("intervals", {})
    c.check("bootstrap by trajectory", "轨迹" in iv.get("method", "") or "命令簇" in iv.get("method", ""))
    c.check("bootstrap replicates", int(iv.get("replicates", 0)) >= 1000)
    cd = st.get("coverage_denominator", {})
    c.check("two denominators reported", bool(cd.get("overall")) and bool(cd.get("conditional")))
    c.check("joint coverage not implied", "不推出" in st.get("joint_coverage", ""))

    # ---- 7b. 终止规则与簇相关性 ----
    tr = p.get("termination", {})
    for key in ("env_terminations", "trial_termination", "truncation",
                "energy_partial_rule", "no_resume", "denominator_rule"):
        c.check(f"termination.{key} present", bool(tr.get(key)))
    cl = st.get("cluster_handling", {})
    for key in ("definition", "resampling_unit", "reporting", "independence"):
        c.check(f"statistics.cluster_handling.{key} present", bool(cl.get(key)))

    # ---- 8. 容差 ----
    tol = p.get("tolerances", {})
    for key in ("energy_recompute_rel", "online_vs_offline_rel", "position_pairing_abs_m",
                "time_sync_abs_s", "obs_dim_expected", "action_dim_expected", "joint_count"):
        c.check(f"tolerances.{key} present", key in tol)
    c.eq("obs_dim == 45", tol.get("obs_dim_expected"), 45)
    c.eq("action_dim == 12", tol.get("action_dim_expected"), 12)

    # ---- 9. 失败分类三条边界 ----
    inv = p.get("invalid_rules", {})
    for key in ("business_failure", "data_invalid", "infra_retry"):
        c.check(f"invalid_rules.{key} present", key in inv and bool(inv[key].get("treatment")))
    c.check("invalid_rules boundary declared", bool(inv.get("boundary")))

    # ---- 10. 统计准入 ----
    acc = p.get("statistical_acceptance", {})
    c.check("acceptance.calibration.require", bool(acc.get("calibration", {}).get("require")))
    c.check("acceptance.heldout.may_not", bool(acc.get("heldout", {}).get("may_not")))
    c.close("min_complete_share_per_cell", acc.get("min_complete_share_per_cell"), 0.9, 1e-9)
    c.check("near_obstacle acceptance", bool(acc.get("near_obstacle", {}).get("require")))

    # ---- 11. 资源预算与实测一致 ----
    rb = p.get("resource_budget", {})
    m = rb.get("measurement", {})
    c.eq("budget.measurement.trials", m.get("trials"), 32)
    c.eq("budget.measurement.substeps", m.get("physics_substeps_recorded"), k.get("measured_steps"))
    c.check("budget has estimates for all domains",
            all(x in rb.get("estimates", {}) for x in
                ("pilot_20", "baseline_400", "calibration_800", "heldout_400", "near_obstacle_200", "total_1820")))

    # ---- 12. 契约绑定 ----
    a = p.get("applies_to", {})
    c.eq("applies_to.controller", a.get("controller"), "camra_go2_velocity_it2999_v2")
    c.eq("applies_to.decision", a.get("decision"), "D-005")
    c.check("applies_to bundle hash present", len(a.get("bundle_tree_sha256", "")) == 64)

    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--write-hash", action="store_true")
    ap.add_argument("--check-hash", action="store_true")
    args = ap.parse_args()
    path: Path = args.protocol
    if not path.exists():
        print(f"FATAL: protocol not found: {path}")
        return 4
    digest = sha256_file(path)

    if args.check_hash and not args.write_hash:
        hf = path.with_suffix(path.suffix + ".sha256")
        if not hf.exists():
            print(f"FATAL: hash file not found: {hf}")
            return 4
        recorded = hf.read_text(encoding="utf-8").split()[0]
        if recorded != digest:
            print(f"HASH MISMATCH\n  recorded {recorded}\n  actual   {digest}")
            return 3
        print(f"HASH OK {digest}")
        return 0

    payload = json.loads(path.read_text(encoding="utf-8"))
    chk = validate(payload, path)
    if chk.failures:
        print(f"FAIL: {len(chk.failures)}/{chk.checks} checks failed")
        for f in chk.failures:
            print(f"  - {f}")
        return 2

    if args.write_hash:
        hf = path.with_suffix(path.suffix + ".sha256")
        hf.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        print(f"wrote {hf.name}")
    print(f"PASS: {chk.checks}/{chk.checks} checks, protocol {payload['version']}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
