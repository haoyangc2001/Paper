"""校验数据锁（CAL-023 验收）：散列、配额、互斥、近障地图与门禁行为。

用法：
    python scripts/check_data_lock.py --lock config/data_lock_v1.json \
        --report /pub/data/caohy/logs/camra/lock/data_lock_check.json

退出码：0 通过；2 失败；4 文件缺失。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from camra_calibration.lock import (  # noqa: E402
    DataLock, LockError, enforce_collection, sha256_file,
)
from camra_calibration.obstacles import CLEARANCE_TIERS  # noqa: E402
from camra_calibration.protocol import Protocol  # noqa: E402

FORMAL = ("calibration", "heldout", "near_obstacle")
NON_FORMAL = ("baseline_fit", "baseline_dev", "baseline_validate", "pilot")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-023 数据锁校验")
    ap.add_argument("--lock", default=None)
    ap.add_argument("--report", required=True)
    ap.add_argument("--margins", default=None, help="用于验证门禁（缺省按不存在处理）")
    args = ap.parse_args(argv)

    cal = Path(__file__).resolve().parent.parent
    lock_p = Path(args.lock) if args.lock else cal / "config" / "data_lock_v1.json"
    if not lock_p.is_file():
        print(f"FATAL: 锁文件缺失 {lock_p}")
        return 4

    problems: list[str] = []
    checks: dict[str, dict] = {}
    try:
        lock = DataLock.load(lock_p)
    except LockError as exc:
        print(f"FATAL: {exc}")
        return 2

    # ---- 1) 散列 ----
    bad = lock.verify_artifacts(cal)
    checks["artifacts"] = {"ok": not bad, "mismatches": bad}
    if bad:
        problems.extend(bad)

    # ---- 2) 配额与条数 ----
    proto = Protocol.load(Path(lock.protocol["path"]))
    if proto.sha256 != lock.protocol["sha256"]:
        problems.append("协议散列与锁不一致")
    quota_rows = {}
    for split, entry in lock.payload["trial_lists"]["files"].items():
        want = proto.quota(split)
        start, count = proto.seed_block(split)
        got = entry["count"]
        ok = (want == entry["quota"] == got and entry["seed_block"] == [start, count])
        quota_rows[split] = {"quota": want, "count": got, "seed_block": entry["seed_block"],
                             "protocol_seed_block": [start, count], "ok": ok}
        if not ok:
            problems.append(f"{split} 配额/seed 与协议不符: {quota_rows[split]}")
    checks["quotas"] = {"ok": all(r["ok"] for r in quota_rows.values()), "rows": quota_rows}

    # ---- 3) 互斥 ----
    seeds: dict[str, set[int]] = {}
    tids: dict[str, set[str]] = {}
    for split, entry in lock.payload["trial_lists"]["files"].items():
        rows = [json.loads(l) for l in Path(entry["path"]).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        seeds[split] = {r["seed"] for r in rows}
        tids[split] = {r["trial_id"] for r in rows}
    overlaps = {}
    names = sorted(seeds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter_s = seeds[a] & seeds[b]
            inter_t = tids[a] & tids[b]
            if inter_s or inter_t:
                overlaps[f"{a}|{b}"] = {"seeds": len(inter_s), "trial_ids": len(inter_t)}
    checks["mutual_exclusion"] = {"ok": not overlaps, "overlaps": overlaps}
    if overlaps:
        problems.append(f"split 之间存在 seed/trial_id 交叉: {list(overlaps)[:3]}")

    # ---- 4) 近障地图 ----
    maps_p = Path(lock.payload["near_obstacle_maps"]["path"])
    maps = json.loads(maps_p.read_text(encoding="utf-8"))
    cases = maps["cases"]
    tiers = Counter(c["clearance_target_m"] for c in cases)
    near_rows = [json.loads(l) for l in
                 Path(lock.payload["trial_lists"]["files"]["near_obstacle"]["path"])
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
    near_ids = {r["trial_id"] for r in near_rows}
    map_ids = {c["trial_id"] for c in cases}
    bad_clear = [c["trial_id"] for c in cases
                 if c["clearance_achieved_m"] < c["clearance_target_m"] - 2e-3]
    bad_arena = [c["trial_id"] for c in cases
                 if not (c["path_in_arena"] and c["obstacles_in_arena"])]
    # 场地按"路径 + 半宽 + 净距 + 1 m"定尺寸，因此路径到墙必然 ≥1 m；
    # 障碍被外推后可能更靠近墙，这里只要求不贴墙（≥0.5 m）。
    bad_wall = [c["trial_id"] for c in cases if c["wall_margin_m"] < 0.5 - 1e-9]
    tier_ok = all(tiers.get(float(t), 0) == len(cases) / len(CLEARANCE_TIERS) for t in CLEARANCE_TIERS)
    fp_entry = lock.payload["footprint"]
    fp_bind = maps["footprint"]["sha256"] in (fp_entry.get("content_sha256"),
                                              fp_entry.get("sha256"))
    near_ok = (len(cases) == 200 and map_ids == near_ids and not bad_clear
               and not bad_arena and not bad_wall and tier_ok and fp_bind)
    checks["near_obstacle_maps"] = {
        "ok": bool(near_ok), "n_cases": len(cases), "tiers": {str(k): v for k, v in tiers.items()},
        "tier_balanced": bool(tier_ok), "ids_match_list": map_ids == near_ids,
        "footprint_bound": bool(fp_bind),
        "clearance_violations": bad_clear[:5], "arena_violations": bad_arena[:5],
        "wall_margin_violations": bad_wall[:5],
        "arena_extent": [min(c["arena"]["width_m"] for c in cases),
                         max(c["arena"]["width_m"] for c in cases),
                         min(c["arena"]["height_m"] for c in cases),
                         max(c["arena"]["height_m"] for c in cases)],
    }
    if not near_ok:
        problems.append(f"近障地图未通过: {checks['near_obstacle_maps']}")

    # ---- 5) 门禁行为 ----
    gates = {}
    margins = Path(args.margins) if args.margins else cal / "config" / "margins_v1.json"
    if margins.is_file():
        gates["margins_present"] = {"ok": True, "note": "margin 已冻结，检验/近障可采",
                                    "path": str(margins)}
    else:
        for split in ("heldout", "near_obstacle"):
            try:
                enforce_collection(lock, split, cal,
                                   manifest_dir=Path(lock.payload["trial_lists"]["dir"]),
                                   maps_path=maps_p, margins_path=margins)
                gates[f"{split}_without_margin"] = {"ok": False, "note": "本应被拒绝却放行"}
                problems.append(f"{split} 在 margin 未冻结时未被拒绝")
            except LockError as exc:
                gates[f"{split}_without_margin"] = {"ok": True, "refused": str(exc)[:120]}
    # 正式 split 不允许覆盖
    for split in FORMAL:
        try:
            enforce_collection(lock, split, cal,
                               manifest_dir=Path(lock.payload["trial_lists"]["dir"]),
                               maps_path=maps_p, margins_path=margins,
                               allow_overwrite=True)
            gates[f"{split}_overwrite_refused"] = {"ok": False}
            problems.append(f"{split} 允许 --allow-overwrite，未被拒绝")
        except LockError:
            gates[f"{split}_overwrite_refused"] = {"ok": True}
    # 不存在的 split 必须拒绝
    try:
        enforce_collection(lock, "not_a_split", cal)
        gates["unknown_split_refused"] = {"ok": False}
        problems.append("未知 split 未被拒绝")
    except LockError:
        gates["unknown_split_refused"] = {"ok": True}
    checks["gates"] = {"ok": all(v["ok"] for v in gates.values()), "rows": gates}
    if not checks["gates"]["ok"]:
        problems.append("采集门禁行为不符合预期")

    report = {
        "cal_task": "CAL-023",
        "lock": {"path": str(lock_p), "sha256": lock.sha256,
                 "protocol_version": lock.protocol.get("version")},
        "checks": checks,
        "problems": problems,
        "verdict": "pass" if not problems else "fail",
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    rep_p.with_suffix(rep_p.suffix + ".sha256").write_text(
        f"{sha256_file(rep_p)}  {rep_p.name}\n", encoding="utf-8")

    print(f"lock {lock_p} sha {lock.sha256}")
    print(f"  artifacts ok={checks['artifacts']['ok']}  quotas ok={checks['quotas']['ok']}  "
          f"exclusion ok={checks['mutual_exclusion']['ok']}")
    nm = checks["near_obstacle_maps"]
    print(f"  near maps ok={nm['ok']} n={nm['n_cases']} tiers={nm['tiers']} "
          f"arena={nm['arena_extent']}")
    print(f"  gates ok={checks['gates']['ok']} ({len(checks['gates']['rows'])} 项)")
    for p in problems:
        print(f"  ERR {p}")
    print(f"report {rep_p}")
    print(f"VERDICT: {report['verdict']}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
