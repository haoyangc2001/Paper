"""构造并冻结 Go2 物理占地轮廓（CAL-021）。

输入只有两类（都不含正式校准/检验结果）：
  * ``config/go2_asset_geometry_v1.json``：从冻结 USD 只读提取的链接/关节/碰撞体；
  * ``baseline_{fit,dev,validate}`` 与 pilot 的 **clean** 试验的 50 Hz 关节记录。

方法（可复算）：
  1. 用资产里的关节链做正运动学，把每个碰撞体采样点映射到机身系；
  2. 对全部 clean 帧的投影点取凸包 → 占地轮廓（机身系 xy 平面上的凸多边形）；
  3. 独立覆盖检查：每个观测帧的每个碰撞体点都必须落在轮廓内；
  4. 同时报告三种参考量（机身静止 / 全链接名义站姿 / 全观测包络），便于判断保守程度。

用法：
    python scripts/build_footprint.py \
        --manifest-dir /pub/data/caohy/logs/camra/trial_lists_v1.0.1 \
        --out-root /pub/data/caohy/logs/camra/trials --out config/footprint_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from camra_calibration.footprint import (  # noqa: E402
    AssetGeometry, BODY_LINKS, Footprint, FootprintError, POLICY_JOINT_ORDER,
    coverage, init_joint_pose, joint_ranges, outline_from_q,
)

BASE_SPLITS = ("baseline_fit", "baseline_dev", "baseline_validate")
PILOT_SPLIT = "pilot"
INIT_TOL_RAD = 0.15      # 关节顺序自检容差：每条试验首帧应接近 init_state
REST_TOL_M = 1e-6        # FK(0) 必须复现资产里作者写的链接静止平移


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"FATAL: 清单不存在: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_done_dir(base: Path, trial_id: str) -> Path | None:
    """优先正式目录，否则取含 DONE 的最大 attempt 目录。"""
    d = base / trial_id
    if (d / "DONE").is_file():
        return d
    best: tuple[int, Path] | None = None
    for p in base.glob(f"{trial_id}__a*"):
        if not (p / "DONE").is_file():
            continue
        try:
            n = int(p.name.split("__a")[-1])
        except ValueError:
            n = 0
        if best is None or n > best[0]:
            best = (n, p)
    return None if best is None else best[1]


def collect_joint_samples(manifest_dir: Path, out_root: Path, stride: int
                          ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """收集 clean 试验的关节角。

    返回 (Q, 首帧矩阵, 来源明细)。只接受 outcome=completed 的完整窗口；
    pilot 只取 profile=clean（其余四个 profile 是扰动，不得抬高几何包络）。
    """
    import torch  # 延迟导入：仅读 trial 产物时需要

    blocks: list[np.ndarray] = []
    firsts: list[np.ndarray] = []
    used: list[dict] = []
    for split in BASE_SPLITS + (PILOT_SPLIT,):
        for spec in read_jsonl(manifest_dir / f"{split}.jsonl"):
            if split == PILOT_SPLIT and spec["profile"] != "clean":
                continue
            d = find_done_dir(out_root / split, spec["trial_id"])
            if d is None:
                continue
            meta_p = d / "meta.json"
            state_p = d / "state_50hz.npz"
            if not (meta_p.is_file() and state_p.is_file()):
                continue
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("outcome") != "completed":
                continue
            q = torch.load(state_p, map_location="cpu")["joint_pos"].numpy()
            if q.ndim != 2 or q.shape[1] != len(POLICY_JOINT_ORDER):
                raise SystemExit(f"FATAL: {spec['trial_id']} 关节形状异常 {q.shape}")
            blocks.append(q[::stride])
            firsts.append(q[0])
            used.append({"split": split, "trial_id": spec["trial_id"], "dir": str(d),
                         "profile": meta.get("profile"), "n_rows": int(q[::stride].shape[0])})
    if not blocks:
        raise SystemExit("FATAL: 没有可用的 clean 关节记录")
    return np.vstack(blocks), np.vstack(firsts), used


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-021 占地轮廓构造与冻结")
    ap.add_argument("--asset", default=None)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help=("50 Hz 抽样步长。**默认必须为 1**：抽样建包络时，未采样的中间帧"
                          "可能略微超出（实测 stride=5 会有 7 个点越界、最多 1.81 mm），"
                          "逐帧检查会立刻抓到。只有做对照实验时才用 >1，且必须同步放宽检查口径。"))
    args = ap.parse_args(argv)

    cal = Path(__file__).resolve().parent.parent
    asset_p = Path(args.asset) if args.asset else cal / "config" / "go2_asset_geometry_v1.json"
    geom = AssetGeometry.load(asset_p)

    # ---- 1) 资产自洽 ----
    rest = geom.rest_consistency()
    if rest["max_abs_err"] > REST_TOL_M:
        bad = [r for r in rest["rows"] if r["max_abs_err"] > REST_TOL_M]
        raise SystemExit(f"FATAL: FK(0) 与资产静止位姿不一致: {bad[:3]}")
    print(f"asset ok: links {len(geom.links)} joints {len(geom.joints)} "
          f"colliders {len(geom.colliders)}  rest_err {rest['max_abs_err']:.2e} m")

    # ---- 2) 关节记录 + 顺序自检 ----
    Q, firsts, used = collect_joint_samples(Path(args.manifest_dir), Path(args.out_root),
                                            args.stride)
    init = init_joint_pose()
    worst_first = float(np.abs(firsts - init).max())
    print(f"joint-order check: {firsts.shape[0]} 条首帧 vs init_state 最大偏差 "
          f"{worst_first:.3f} rad（容差 {INIT_TOL_RAD}）")
    if worst_first > INIT_TOL_RAD:
        raise SystemExit("FATAL: 关节顺序与冻结清单不一致，禁止继续")
    print(f"samples: {Q.shape[0]} 帧（stride {args.stride}），{len(used)} 条 trial")

    # ---- 3) 轮廓与覆盖 ----
    env = outline_from_q(geom, Q, POLICY_JOINT_ORDER)
    nominal = outline_from_q(geom, init[None, :], POLICY_JOINT_ORDER)
    body = outline_from_q(geom, init[None, :], POLICY_JOINT_ORDER, links=BODY_LINKS)
    cov = coverage(env, geom, Q, POLICY_JOINT_ORDER)
    if cov["n_points_outside"] != 0:
        raise SystemExit(f"FATAL: 自覆盖检查失败: {cov}")

    b = env.bounds()
    stats = {
        "n_trials": len(used),
        "n_configs": int(Q.shape[0]),
        "stride": args.stride,
        "nominal_stance": nominal.as_dict(),
        "body_only": body.as_dict(),
        "coverage": cov,
        "area_ratio_vs_nominal": env.area / nominal.area,
        "x_span_m": b[1] - b[0],
        "y_span_m": b[3] - b[2],
        "joint_order_check_max_dev_rad": worst_first,
        "rest_consistency_max_err_m": rest["max_abs_err"],
    }
    fp = Footprint(
        polygon=env, version="footprint-v1", frame="base_link_xy",
        # 注意：method 字符串会进散列，改动它等于改动冻结产物的身份；
        # 已冻结的 footprint_v1.json 由 stride=1 生成，对应文字为 "sampled at 1×50 Hz"。
        method=("hull over the xy projection of all collision primitives (frozen asset) at "
                f"every clean baseline+pilot joint configuration, sampled at {args.stride}×50 Hz"),
        source={
            "asset_geometry": str(asset_p),
            "asset_geometry_sha256": sha256_file(asset_p),
            "asset_usd": geom.source["usd"],
            "asset_usd_sha256": geom.source["usd_sha256"],
            "asset_physics_usd": geom.source["physics_usd"],
            "asset_physics_usd_sha256": geom.source["physics_usd_sha256"],
            "trials": used,
            "splits_used": list(BASE_SPLITS) + [f"{PILOT_SPLIT} (clean only)"],
            "excluded": ("pilot 的 friction_only/mass_only/push_only/combined 四个扰动 profile；"
                         "正式 calibration/heldout/near_obstacle 尚未采集，也未使用"),
        },
        joint_ranges=joint_ranges(Q, POLICY_JOINT_ORDER),
        stats=stats,
        notes=(
            "轮廓是机身系 xy 平面上的凸多边形，单位米；**不含** delta 膨胀。",
            "delta = Q_0.90(ε_d) 由 CAL-025 给出；膨胀只做一次：obstacle ⊕ (footprint ⊕ disc(delta))。",
            "不修改冻结机器人的碰撞几何，本轮廓只是其二维投影包络。",
            "站立/低速等命令条件不影响本轮廓：它只由关节运动包络决定。",
        ),
    )
    digest = fp.save(args.out)
    print(f"envelope: area {env.area:.4f} m2  min_width {env.min_width():.3f}  "
          f"radius {env.circumscribed_radius:.3f}")
    print(f"  bounds x[{b[0]:+.3f},{b[1]:+.3f}] y[{b[2]:+.3f},{b[3]:+.3f}]  "
          f"span {stats['x_span_m']:.3f}×{stats['y_span_m']:.3f} m")
    print(f"  nominal stance: area {nominal.area:.4f}  "
          f"bounds {[round(v, 3) for v in nominal.bounds()]}")
    print(f"  body only: area {body.area:.4f}  bounds {[round(v, 3) for v in body.bounds()]}")
    print(f"  coverage: {cov}")
    print(f"wrote {args.out}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FootprintError as exc:
        print(f"FOOTPRINT ERROR: {exc}")
        sys.exit(2)
