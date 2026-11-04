"""独立检查冻结的占地轮廓（CAL-021 验收项）。

与 ``build_footprint.py`` 分开实现校验路径，不复用其判定结果：

1. 资产与轮廓文件的散列必须自洽（两个侧车都要通过）；
2. 轮廓必须是凸多边形、面积/宽度/半径自洽、min_width > 0；
3. **重算**一遍轮廓（从资产 + 关节记录独立重算）并与冻结顶点逐个比对；
4. 覆盖检查：对全部 clean 试验按 stride 抽样 + 对若干试验**逐帧**（stride=1）检查，
   任何投影点越界即失败；
5. 输入范围检查：只允许 baseline_{fit,dev,validate} 与 pilot(clean)，
   出现正式 calibration / heldout / near_obstacle 即失败；
6. 名义站姿轮廓必须被冻结轮廓包含（保守性下限）。

用法：
    python scripts/check_footprint.py --footprint config/footprint_v1.json \
        --manifest-dir /pub/data/caohy/logs/camra/trial_lists_v1.0.1 \
        --out-root /pub/data/caohy/logs/camra/trials \
        --report /pub/data/caohy/logs/camra/footprint_check.json

退出码：0 通过；2 失败；4 文件缺失。
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
    AssetGeometry, FootprintError, Footprint, POLICY_JOINT_ORDER, coverage,
    init_joint_pose, outline_from_q,
)
from camra_calibration.geometry import Polygon  # noqa: E402

FORBIDDEN = ("calibration", "heldout", "near_obstacle")
ALLOWED = ("baseline_fit", "baseline_dev", "baseline_validate", "pilot")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def find_done_dir(base: Path, trial_id: str) -> Path | None:
    d = base / trial_id
    if (d / "DONE").is_file():
        return d
    best: tuple[int, Path] | None = None
    for p in base.glob(f"{trial_id}__a*"):
        if (p / "DONE").is_file():
            n = int(p.name.split("__a")[-1]) if "__a" in p.name else 0
            if best is None or n > best[0]:
                best = (n, p)
    return None if best is None else best[1]


def load_frames(manifest_dir: Path, out_root: Path, stride: int, per_trial_stride1: int
                ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    import torch
    blocks, dense_blocks, used = [], [], []
    n_trial = 0
    for split in ALLOWED:
        for spec in read_jsonl(manifest_dir / f"{split}.jsonl"):
            if split == "pilot" and spec["profile"] != "clean":
                continue
            d = find_done_dir(out_root / split, spec["trial_id"])
            if d is None:
                continue
            meta_p, state_p = d / "meta.json", d / "state_50hz.npz"
            if not (meta_p.is_file() and state_p.is_file()):
                continue
            if json.loads(meta_p.read_text(encoding="utf-8")).get("outcome") != "completed":
                continue
            q = torch.load(state_p, map_location="cpu")["joint_pos"].numpy()
            blocks.append(q[::stride])
            if n_trial < per_trial_stride1:      # 前若干条逐帧全查（不抽样）
                dense_blocks.append(q)
            n_trial += 1
            used.append({"split": split, "trial_id": spec["trial_id"]})
    if not blocks:
        raise SystemExit("FATAL: 没有可用记录")
    dense = np.vstack(dense_blocks) if dense_blocks else np.zeros((0, len(POLICY_JOINT_ORDER)))
    return np.vstack(blocks), dense, used


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAL-021 占地轮廓独立检查")
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dense-trials", type=int, default=24, help="逐帧全查的试验条数")
    args = ap.parse_args(argv)

    cal = Path(__file__).resolve().parent.parent
    fp_p = Path(args.footprint) if args.footprint else cal / "config" / "footprint_v1.json"
    asset_p = Path(args.asset) if args.asset else cal / "config" / "go2_asset_geometry_v1.json"
    if not fp_p.is_file() or not asset_p.is_file():
        print(f"FATAL: 文件缺失 {fp_p} / {asset_p}")
        return 4

    checks: dict[str, dict] = {}
    problems: list[str] = []

    # ---- 1) 散列 ----
    try:
        fp = Footprint.load(fp_p, verify_hash=True)
        geom = AssetGeometry.load(asset_p, verify_hash=True)
        checks["hash_sidecars"] = {"ok": True, "footprint_sha256": fp.sha256(),
                                   "asset_sha256": sha256_file(asset_p)}
    except FootprintError as exc:
        print(f"FATAL: 散列校验失败: {exc}")
        return 2

    # 轮廓记录里的资产散列必须与当前资产文件一致
    rec_asset = fp.source.get("asset_geometry_sha256")
    same_asset = rec_asset == sha256_file(asset_p)
    checks["asset_binding"] = {"ok": same_asset, "recorded": rec_asset,
                               "actual": sha256_file(asset_p)}
    if not same_asset:
        problems.append("轮廓记录的资产几何散列与当前文件不一致")

    # ---- 2) 轮廓几何自洽 ----
    poly = fp.polygon
    ccw_ok = poly.area > 0
    convex_ok = _is_convex(poly)
    width_ok = poly.min_width() > 0.05
    checks["polygon"] = {"ok": bool(ccw_ok and convex_ok and width_ok),
                         "area_m2": poly.area, "min_width_m": poly.min_width(),
                         "circumscribed_radius_m": poly.circumscribed_radius,
                         "n_vertices": poly.n, "convex": convex_ok, "ccw": ccw_ok}
    if not checks["polygon"]["ok"]:
        problems.append("轮廓不是合法凸多边形或最小宽度异常")

    # ---- 3) 来源范围 ----
    splits_used = [s["split"] for s in fp.source.get("trials", [])]
    bad = [s for s in splits_used if s.startswith(FORBIDDEN)]
    checks["input_scope"] = {"ok": not bad, "splits": sorted(set(splits_used)),
                             "forbidden_hits": bad}
    if bad:
        problems.append(f"轮廓输入含正式数据: {bad[:3]}")

    # ---- 4) 重算并比对 ----
    Q, dense, used = load_frames(Path(args.manifest_dir), Path(args.out_root),
                                 args.stride, args.dense_trials)
    rebuilt = outline_from_q(geom, Q, POLICY_JOINT_ORDER)
    same_vertices = (rebuilt.n == poly.n
                     and np.allclose(np.sort(rebuilt.vertices, axis=0),
                                     np.sort(poly.vertices, axis=0), atol=1e-9))
    checks["recompute"] = {"ok": bool(same_vertices), "n_frames": int(Q.shape[0]),
                           "rebuilt_area_m2": rebuilt.area, "frozen_area_m2": poly.area,
                           "max_vertex_delta_m": float(np.abs(
                               np.sort(rebuilt.vertices, axis=0)
                               - np.sort(poly.vertices, axis=0)).max())
                           if rebuilt.n == poly.n else None}
    if not same_vertices:
        problems.append("独立重算的轮廓与冻结顶点不一致")

    # ---- 5) 覆盖（抽样 + 逐帧）----
    cov_sparse = coverage(poly, geom, Q, POLICY_JOINT_ORDER)
    cov_dense = coverage(poly, geom, dense, POLICY_JOINT_ORDER) if dense.size else None
    ok_cov = (cov_sparse["n_points_outside"] == 0
              and (cov_dense is None or cov_dense["n_points_outside"] == 0))
    checks["coverage"] = {"ok": ok_cov, "sparse": cov_sparse,
                          "dense_stride1": cov_dense, "dense_trials": args.dense_trials}
    if not ok_cov:
        problems.append("覆盖检查出现越界点")

    # ---- 6) 名义站姿必须被包含 ----
    nominal_poly = outline_from_q(geom, init_joint_pose()[None, :], POLICY_JOINT_ORDER)
    inside = bool(poly.contains(nominal_poly.vertices, eps=1e-9).all())
    checks["nominal_contained"] = {"ok": inside,
                                  "nominal_area_m2": nominal_poly.area,
                                  "area_ratio": float(poly.area / nominal_poly.area)}
    if not inside:
        problems.append("名义站姿轮廓未被冻结轮廓包含")

    report = {
        "cal_task": "CAL-021",
        "footprint": {"path": str(fp_p), "sha256": fp.sha256(), "version": fp.version,
                      "frame": fp.frame, "method": fp.method},
        "checks": checks,
        "trials_used": len(used),
        "problems": problems,
        "verdict": "pass" if not problems else "fail",
    }
    rep_p = Path(args.report)
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    rep_p.with_suffix(rep_p.suffix + ".sha256").write_text(
        f"{sha256_file(rep_p)}  {rep_p.name}\n", encoding="utf-8")

    print(f"footprint {fp_p}  sha {fp.sha256()}")
    print(f"  area {poly.area:.4f} m2  min_width {poly.min_width():.3f}  "
          f"radius {poly.circumscribed_radius:.3f}  vertices {poly.n}")
    print(f"  recompute deltas: {checks['recompute']['max_vertex_delta_m']}")
    print(f"  coverage sparse: outside={cov_sparse['n_points_outside']} "
          f"max={cov_sparse['max_outside_m']:.2e} over {cov_sparse['n_configs']} frames")
    if cov_dense:
        print(f"  coverage dense : outside={cov_dense['n_points_outside']} "
              f"max={cov_dense['max_outside_m']:.2e} over {cov_dense['n_configs']} frames")
    for p in problems:
        print(f"  ERR {p}")
    print(f"report {rep_p}")
    print(f"VERDICT: {report['verdict']}")
    return 0 if not problems else 2


def _is_convex(poly: Polygon) -> bool:
    v = poly.vertices
    nxt = np.roll(v, -1, axis=0)
    e1 = nxt - v
    e2 = np.roll(nxt, -1, axis=0) - nxt
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return bool((cross >= -1e-12).all())


if __name__ == "__main__":
    sys.exit(main())
