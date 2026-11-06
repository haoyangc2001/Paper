"""HIRA-027：重验 HiRA 执行系统的 Go2 占地轮廓。

流程（全部可从落盘文件复算）：
1. 读 `config/go2_asset_geometry_hira_v1.json`（由 extract_geometry_hira.py 从本候选 URDF 抽取）并校验散列；
2. `rest_consistency`：FK(全关节=0) 必须复现每个链接的静止平移；
3. 用**已采到的独立数据**（baseline_fit/dev/validate + pilot 的 50 Hz 关节记录）给出腿部运动包络，
   据此算凸多边形轮廓；
4. `coverage`：把同一批帧的**全部投影点**逐点检查是否落在轮廓内（不允许越界）；
5. 另算一份"关节限位角点"轮廓作量级对照（腿间独立 8 角点组合 → 8⁴=4096 帧）。**注意**：
   角点组合在 x/y 方向更长但面积更小（缺"单腿前伸"这类不对称姿态），因此它**不是**实测包络的超集，
   只能当参照；若实测姿态超出包络，必须重新采集并重算轮廓，不得用角点轮廓顶替；
6. 冻结 `config/footprint_hira_v1.json`（含 sha256 侧车）并写 `reports/footprint_validation.json`。

几何只在这里进入一次；δ 的扩张由 `Footprint.obstacle_dilation` 在危险集侧单独施加，
不允许在轮廓里预先加安全裕度（否则 δ 会被重复计入）。

    python tests/verify_footprint_hira.py            # 全量重算（outline 3 min + coverage 17 min）
    python tests/verify_footprint_hira.py --reuse    # 只重做判定（多边形取冻结文件、coverage 取旧报告）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
for path in (str(REPO / "CaMRA" / "locomotion" / "adapters"), str(REPO / "CaMRA" / "calibration" / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from camra_calibration.footprint import (  # noqa: E402
    POLICY_JOINT_ORDER,
    AssetGeometry,
    Footprint,
    coverage,
    joint_ranges,
    outline_from_q,
)
from hira_go2.env_runner import SOURCE_ORDER  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--geometry", default=str(EXP / "config/go2_asset_geometry_hira_v1.json"))
parser.add_argument("--runs-root", default="/pub/data/caohy/logs/camra_hira/runs")
parser.add_argument("--runs", default="baseline_20261106_fit,baseline_20261106_dev,"
                                     "baseline_20261106_validate,pilot_20261106_protocolv1")
parser.add_argument("--stride", type=int, default=1, help="关节记录降采样步长（1=全部 50 Hz 帧）")
parser.add_argument("--out-config", default=str(EXP / "config/footprint_hira_v1.json"))
parser.add_argument("--out-report", default=str(EXP / "reports/footprint_validation.json"))
parser.add_argument("--reuse", action="store_true",
                    help="只重做判定：多边形取自已冻结的 footprint 文件，coverage/limit 取自旧报告")
args_cli = parser.parse_args()

T0 = time.time()


def log(text: str) -> None:
    print(f"[fp] {time.time() - T0:6.1f}s {text}", flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_frames() -> tuple[np.ndarray, dict]:
    """把各 run 的 50 Hz 关节记录按名字重排成 POLICY_JOINT_ORDER。"""
    index = [SOURCE_ORDER.index(name) for name in POLICY_JOINT_ORDER]
    blocks, per_run = [], {}
    for run_id in [x for x in args_cli.runs.split(",") if x]:
        run_dir = Path(args_cli.runs_root) / run_id
        if not run_dir.is_dir():
            per_run[run_id] = {"trials": 0, "frames": 0, "missing": True}
            continue
        n_trials = n_frames = 0
        for trial_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            attempts = sorted(a for a in trial_dir.iterdir() if a.is_dir() and a.name.startswith("attempt_"))
            if not attempts:
                continue
            path = attempts[0] / "state_50hz.npz"
            if not path.is_file():
                continue
            payload = np.load(path)
            q = payload["joint_pos_source"].astype(np.float64)[:: args_cli.stride]
            if q.shape[1] != 12:
                raise SystemExit(f"{path}: joint_pos_source 列数 {q.shape[1]} ≠ 12")
            blocks.append(q[:, index])
            n_trials += 1
            n_frames += q.shape[0]
        per_run[run_id] = {"trials": n_trials, "frames": n_frames, "missing": False}
    if not blocks:
        raise SystemExit("没有可用的关节记录")
    return np.vstack(blocks), per_run


def limit_corner_frames(geom: AssetGeometry) -> np.ndarray:
    """关节限位内的**腿间独立** 8 角点组合（8⁴=4096 帧），仅作量级对照。"""
    limits = {j.name: (j.lower, j.upper) for j in geom.joints if j.kind == "revolute"}
    legs = ("FL", "FR", "RL", "RR")
    per_leg = []
    for leg in legs:
        names = [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"]
        per_leg.append(np.asarray(
            [[limits[n][(mask >> k) & 1] for k, n in enumerate(names)] for mask in range(8)],
            dtype=np.float64,
        ))
    frames = []
    for a in per_leg[0]:
        for b in per_leg[1]:
            for c in per_leg[2]:
                for d in per_leg[3]:
                    frames.append(np.concatenate([a, b, c, d]))
    return np.asarray(frames, dtype=np.float64)


def main() -> int:
    geom_path = Path(args_cli.geometry)
    geom = AssetGeometry.load(geom_path, verify_hash=True)
    log(f"geometry: {geom_path.name} links={len(geom.links)} joints={len(geom.joints)} "
        f"colliders={len(geom.colliders)} sha={sha256_file(geom_path)[:16]}")

    rest = geom.rest_consistency()
    log(f"rest_consistency max_abs_err={rest['max_abs_err']:.3e}")

    out_config = Path(args_cli.out_config)
    out_report = Path(args_cli.out_report)

    if args_cli.reuse:
        previous = json.loads(out_report.read_text(encoding="utf-8"))
        frozen = Footprint.load(out_config, verify_hash=True)
        polygon = frozen.polygon
        cov = previous["outline"]["coverage"]
        limit_stats = previous["outline"]["limit_envelope"]
        ranges = previous["joint_ranges_rad"]
        per_run = previous["frames"]["per_run"]
        n_frames = previous["frames"]["total"]
        log(f"reuse: polygon {polygon.n} 顶点（来自已冻结 footprint），coverage/limit 取自旧报告")
    else:
        Q, per_run = collect_frames()
        detail = ", ".join(f"{k}:{v['frames']}" for k, v in per_run.items())
        n_trials = sum(v["trials"] for v in per_run.values())
        log(f"frames={Q.shape[0]} from {n_trials} trials ({detail})")

        polygon = outline_from_q(geom, Q, POLICY_JOINT_ORDER, chunk=2048)
        bb = polygon.bounds()
        log(f"outline: {polygon.n} 顶点, 面积 {polygon.area:.4f} m², 包围盒 "
            f"{bb[1] - bb[0]:.4f} × {bb[3] - bb[2]:.4f} m")

        cov = coverage(polygon, geom, Q, POLICY_JOINT_ORDER, chunk=512)
        log(f"coverage: max_outside={cov['max_outside_m']:.3e} m, 越界点 {cov['n_points_outside']}, "
            f"最差帧 {cov['worst_frame']}/{cov['n_configs']}")

        limit_Q = limit_corner_frames(geom)
        limit_poly = outline_from_q(geom, limit_Q, POLICY_JOINT_ORDER, chunk=1024)
        limit_cov = coverage(limit_poly, geom, limit_Q, POLICY_JOINT_ORDER, chunk=512)
        lb = limit_poly.bounds()
        limit_stats = {
            "n_vertices": limit_poly.n,
            "area_m2": limit_poly.area,
            "bbox_m": [lb[1] - lb[0], lb[3] - lb[2]],
            "coverage": limit_cov,
            "construction": "每条腿 3 关节取限位 8 角点、腿间独立组合（8⁴=4096 帧）",
        }
        log(f"limit-envelope: {limit_poly.n} 顶点, 面积 {limit_poly.area:.4f}, "
            f"包围盒 {lb[1] - lb[0]:.4f} × {lb[3] - lb[2]:.4f}")
        ranges = joint_ranges(Q, POLICY_JOINT_ORDER)
        n_frames = int(Q.shape[0])

    bb = polygon.bounds()
    bbox = [bb[1] - bb[0], bb[3] - bb[2]]
    limit_bbox = limit_stats["bbox_m"]
    limit_rel = {
        "area_ratio_limit_over_measured": limit_stats["area_m2"] / polygon.area,
        "bbox_ratio_limit_over_measured": [limit_bbox[0] / bbox[0], limit_bbox[1] / bbox[1]],
        "area_superset": bool(limit_stats["area_m2"] >= polygon.area - 1e-12),
        "bbox_superset": bool(limit_bbox[0] >= bbox[0] - 1e-12 and limit_bbox[1] >= bbox[1] - 1e-12),
        "is_superset": bool(
            limit_stats["area_m2"] >= polygon.area - 1e-12
            and limit_bbox[0] >= bbox[0] - 1e-12
            and limit_bbox[1] >= bbox[1] - 1e-12
        ),
        "interpretation": "角点组合在 x/y 方向更长，但面积更小：它缺少「单腿前伸/后蹬」这类不对称姿态，"
        "所以**不是**实测包络的超集。仅作量级对照；实测姿态若超出包络必须重采重算，"
        "禁止用角点轮廓顶替（也不得靠放大 δ 掩盖）",
    }

    source = {
        "geometry_file": geom_path.name,
        "geometry_sha256": sha256_file(geom_path),
        "urdf_sha256": geom.source.get("urdf_sha256"),
        "bundle": "hira_go2_18500",
        "joint_record_runs": per_run,
        "n_frames": n_frames,
        "stride": args_cli.stride,
        "data_note": "只用本候选的 clean baseline（fit/dev/validate）与 pilot 的 50 Hz 关节记录；"
        "未使用 heldout / near_obstacle（results_guard 封锁）",
    }
    stats = {
        "n_vertices": polygon.n,
        "area_m2": polygon.area,
        "bbox_m": bbox,
        "bbox_xy_m": {"x": [bb[0], bb[1]], "y": [bb[2], bb[3]]},
        "perimeter_m": polygon.perimeter,
        "circumscribed_radius_m": polygon.circumscribed_radius,
        "coverage": cov,
        "limit_envelope": dict(limit_stats, relation=limit_rel),
        "delta_note": "轮廓只含机器人自身碰撞几何与腿部摆动包络，不含任何安全裕度；"
        "δ 由 Footprint.obstacle_dilation 在危险集侧施加一次",
    }
    footprint = Footprint(
        polygon=polygon,
        version="footprint-hira-v1",
        frame="base（机身系，x 前 y 左 z 上，单位 m）",
        method="对全部实测帧的碰撞体投影点取凸包（outline_from_q），再逐点覆盖校验（coverage）",
        source=source,
        joint_ranges=ranges,
        stats=stats,
        notes=(
            "由本候选 URDF 重新抽取的碰撞几何计算，未复用 v2 的 0.889×0.585 m 数值",
            "轮廓不含 δ；危险集扩张时 footprint 与 δ 各只进入一次",
            "限位角点轮廓不是超集（面积更小），只能作参照",
        ),
    )
    fp_sha = footprint.save(out_config)
    reloaded = Footprint.load(out_config, verify_hash=True)
    log(f"frozen footprint sha={fp_sha[:16]} -> {out_config.name} (reload ok={reloaded.sha256() == fp_sha})")

    checks = {
        "geometry_hash_verified": True,
        "rest_consistency": rest["max_abs_err"] <= 1e-9,
        "n_colliders_matches_source": len(geom.colliders) == geom.source.get("n_colliders"),
        "coverage_no_outside": cov["max_outside_m"] <= 1e-9 and cov["n_points_outside"] == 0,
        "footprint_reload_matches": reloaded.sha256() == fp_sha,
        "geometry_bound_to_urdf": bool(geom.source.get("urdf_sha256")),
        "no_safety_margin_inside": True,
        "every_revolute_joint_in_ranges": set(ranges) == {
            j.name for j in geom.joints if j.kind == "revolute"
        },
    }
    report = {
        "report": "footprint_validation",
        "task": "HIRA-027",
        "geometry": {"file": geom_path.name, "sha256": source["geometry_sha256"],
                     "urdf_sha256": source["urdf_sha256"],
                     "n_links": len(geom.links), "n_joints": len(geom.joints),
                     "n_colliders": len(geom.colliders),
                     "collider_kinds": geom.source.get("collider_kinds"),
                     "conventions": geom.source.get("conventions")},
        "rest_consistency": rest,
        "frames": {"total": n_frames, "per_run": per_run, "stride": args_cli.stride},
        "outline": stats,
        "joint_ranges_rad": ranges,
        "footprint": {"path": str(out_config), "sha256": fp_sha, "version": footprint.version,
                      "reload_matches": reloaded.sha256() == fp_sha},
        "comparison_with_v2": {
            "note": "v2 的轮廓（0.889×0.585 m）来自另一份 USD 资产，不能直接沿用；此处仅作量级对照",
            "v2_reference_bbox_m": [0.889, 0.585],
            "hira_bbox_m": bbox,
            "hira_over_v2_ratio": [bbox[0] / 0.889, bbox[1] / 0.585],
        },
        "checks": checks,
        "verdict": {"all_passed": all(checks.values()),
                    "failed": [k for k, v in checks.items() if not v]},
        "delta_relation": {
            "footprint_contains_robot_only": True,
            "delta_applied_once_in": "Footprint.obstacle_dilation(obstacle, delta) = obstacle ⊕ (footprint ⊕ disc(delta))",
            "no_double_counting": "轮廓内不含任何裕度；δ 不在轮廓里预置",
        },
        "reused_previous_computation": bool(args_cli.reuse),
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"checks {sum(checks.values())}/{len(checks)} failed={report['verdict']['failed']}")
    print("FOOTPRINT_OK" if report["verdict"]["all_passed"] else "FOOTPRINT_FAIL", flush=True)
    return 0 if report["verdict"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
