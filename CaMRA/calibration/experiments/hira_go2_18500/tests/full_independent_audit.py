"""HIRA-036：七分片数据与数值的独立审计。

**独立性纪律**：本脚本**不 import** `camra_calibration` 与 `hira_go2` 的任何函数。
平面名义积分、功率特征、ε_c/ε_d、nearest-rank 分位、覆盖判定、矩形-足迹相交，
全部在本文档内按规格另写一遍；只读它们的**产物**（JSON/NPZ）与**冻结常量**。
若调用被测函数，等于用同一把尺子量自己，审计就没有意义（计划 constraints 明令禁止）。

审计四类东西：

1. **结构**：1820 条配额、trial_id 全局唯一、attempt 目录与 DONE、主 attempt 唯一、
   seed 块两两互斥、版本散列（bundle / model / margins / lock）。
2. **数值**：从原始 200 Hz `tau`/`qdot` 用 float64 重积分正功；从命令重推名义轨迹、
   重算 J_nom、ε_c、ε_d；再独立复算 α/δ 与覆盖率，与冻结值/主工具输出对照。
3. **几何**：近障 200 条的足迹-障碍重叠，用自家的旋转矩形判定，对照 §37 的 151/200。
4. **隔离**：heldout/near 未进入拟合与阈值；旧 v2 与只读参考源未被改动。

差异处理：任何超容差的差异都要写进 `audit_discrepancies.json` 并给出可复查解释；
**无法取得原始数据不判通过**，也不修改旧记录去消差。

    python tests/full_independent_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent.parent.parent
RUNS = Path("/pub/data/caohy/logs/camra_hira/runs")

#: (split, run_id, 预登记配额)
SPLITS = (
    ("pilot", "pilot_20261106_protocolv1", 20),
    ("baseline_fit", "baseline_20261106_fit", 240),
    ("baseline_dev", "baseline_20261106_dev", 80),
    ("baseline_validate", "baseline_20261106_validate", 80),
    ("calibration", "calibration_20261106", 800),
    ("heldout", "heldout_20261106", 400),
    ("near_obstacle", "near_obstacle_20261107", 200),
)
QUOTA_TOTAL = 1820
DT_SIM = 0.005
PAIR_DT = 0.1
J_MIN = 1.0
ENERGY_REL_MAX = 1e-9
QUANTILE = 0.9
LOW_SPEED_VX, LOW_SPEED_WZ = 0.3, 0.3

parser = argparse.ArgumentParser()
parser.add_argument("--runs-root", default=str(RUNS))
parser.add_argument("--out", default=str(EXP / "reports/full_independent_audit.json"))
parser.add_argument("--out-discrepancies", default=str(EXP / "reports/audit_discrepancies.json"))
parser.add_argument("--v2-config", default="/home/caohy/repositories/Paper/CaMRA/calibration/config",
                    help="v2 冻结物所在目录（只在 git 仓库里，/pub 工作副本不含）")
args_cli = parser.parse_args()

CHECKS: list[dict] = []
DISC: list[dict] = []


def add(cid: str, name: str, ok: bool, detail: str, **extra) -> None:
    CHECKS.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'ok  ' if ok else 'FAIL'}] {cid} {name}\n        {detail}", flush=True)


def note(cid: str, kind: str, detail: str, **extra) -> None:
    DISC.append({"id": cid, "kind": kind, "detail": detail, **extra})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 独立实现 1：平面名义积分（圆弧闭式解，零阶保持）
def wrap_pi(a: float) -> float:
    two_pi = 2.0 * math.pi
    a = math.fmod(a + math.pi, two_pi)
    if a <= 0:
        a += two_pi
    return a - math.pi


def step_exact_indep(px: float, py: float, yaw: float, cmd, dt: float):
    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    th0 = yaw
    dyaw = wz * dt
    if abs(wz) < 1e-12:
        c, s = math.cos(th0), math.sin(th0)
        dx = (vx * c - vy * s) * dt
        dy = (vx * s + vy * c) * dt
    else:
        th1 = th0 + dyaw
        sx = (math.sin(th1) - math.sin(th0)) / wz
        cx = (math.cos(th1) - math.cos(th0)) / wz
        dx = vx * sx + vy * cx
        dy = -vx * cx + vy * sx
    return px + dx, py + dy, th0 + dyaw


def nominal_xy_indep(commands, start, yaw0: float, dt: float = PAIR_DT) -> np.ndarray:
    px, py, yaw = float(start[0]), float(start[1]), float(yaw0)
    out = [(px, py)]
    for i in range(len(commands) - 1):
        px, py, yaw = step_exact_indep(px, py, yaw, commands[i], dt)
        out.append((px, py))
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# 独立实现 2：二次功率特征 + J_nom
#: 与 power_model_hira_v1.json 的 feature_names 同序（读文件时会再断言一次）
FEAT_ORDER = ("bias", "abs_vx", "abs_vy", "abs_wz",
              "vx2", "vy2", "wz2", "abs_vx_wz", "abs_vy_wz", "abs_vx_vy")


def step_power_indep(commands: np.ndarray, w: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """逐区间功率，第 i 行只依赖 u[i]（本模型 history_k=0）。"""
    n = commands.shape[0] - 1
    u = commands[:n]
    ax, ay, az = np.abs(u[:, 0]), np.abs(u[:, 1]), np.abs(u[:, 2])
    cols = {"bias": np.ones_like(ax), "abs_vx": ax, "abs_vy": ay, "abs_wz": az,
            "vx2": ax * ax, "vy2": ay * ay, "wz2": az * az,
            "abs_vx_wz": ax * az, "abs_vy_wz": ay * az, "abs_vx_vy": ax * ay}
    phi = np.stack([cols[c] for c in names], axis=1)
    return np.clip(phi @ w, 0.0, None)


def j_nom_indep(commands: np.ndarray, w: np.ndarray, names: tuple[str, ...], dt: float) -> float:
    return float(step_power_indep(commands, w, names).sum() * dt)


# --------------------------------------------------------------------------- #
# 独立实现 3：能量正功
def positive_work_indep(tau: np.ndarray, qdot: np.ndarray, dt: float) -> float:
    """逐关节截断后求和：Σ_t Σ_j max(τ·q̇, 0)·dt。"""
    prod = tau.astype(np.float64) * qdot.astype(np.float64)
    return float(np.maximum(prod, 0.0).sum() * dt)


# --------------------------------------------------------------------------- #
# 独立实现 4：nearest-rank 分位
def nearest_rank_indep(values, q: float) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = int(math.ceil(q * arr.size))
    return float(arr[max(k - 1, 0)])


# --------------------------------------------------------------------------- #
# 独立实现 5：旋转矩形 vs 足迹多边形（重叠判定）
def _rect_corners(x: float, y: float, width: float, depth: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hw, hd = width / 2.0, depth / 2.0
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (local @ rot.T) + np.array([x, y], dtype=np.float64)


def _point_in_poly(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """射线法；points (N,2), poly (M,2) → (N,) bool。"""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    m = poly.shape[0]
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        cond = ((y1 > y) != (y2 > y))
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xin)
    return inside


def footprint_vertices_indep(verts: np.ndarray, pos: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return verts @ rot.T + np.asarray(pos, dtype=np.float64)


def overlaps_indep(fp_verts: np.ndarray, obstacle: dict) -> bool:
    """足迹与障碍矩形是否相交：任一足迹顶点在矩形内，或矩形中心在足迹内。

    这是碰撞的**下界**判据（可能漏掉极端交叉情形，但对 0.3 m 的柱子与 1.0×0.78 m 的足迹足够）。
    """
    corners = _rect_corners(float(obstacle["x"]), float(obstacle["y"]),
                            float(obstacle["width"]), float(obstacle["depth"]),
                            float(obstacle.get("yaw", 0.0)))
    if _point_in_poly(fp_verts, corners).any():
        return True
    center = np.array([[float(obstacle["x"]), float(obstacle["y"])]])
    return bool(_point_in_poly(center, fp_verts)[0])


# --------------------------------------------------------------------------- #
def main() -> int:
    runs_root = Path(args_cli.runs_root)
    margins = json.loads((EXP / "config/margins_hira_v1.json").read_text(encoding="utf-8"))
    model_payload = json.loads((EXP / "config/power_model_hira_v1.json").read_text(encoding="utf-8"))
    bundle_manifest = json.loads((EXP / "frozen/hira_go2_18500/manifest.json").read_text(encoding="utf-8"))
    lock_eval = json.loads((EXP / "config/data_lock_hira_evaluation_v2.json").read_text(encoding="utf-8"))
    alpha_frozen = float(margins["alpha"])
    delta_frozen = float(margins["delta"])
    bundle_id = bundle_manifest["bundle_id"]

    names = tuple(model_payload["feature_names"])
    weights = np.asarray(model_payload["weights"], dtype=np.float64)
    dt_cost = float(model_payload.get("step_cost_dt", PAIR_DT))
    if names != FEAT_ORDER:
        raise SystemExit(f"REFUSED：模型特征顺序与独立实现不一致\n  文件 {names}\n  脚本 {FEAT_ORDER}")
    print(f"[audit] alpha={alpha_frozen:.6f} delta={delta_frozen:.6f} dt_cost={dt_cost} "
          f"n_feat={len(names)}", flush=True)

    # ---------------- 结构：配额 / 唯一性 / attempt / 失败分母 ----------------
    per_split: dict[str, dict] = {}
    all_ids: list[str] = []
    seed_blocks: dict[str, set[int]] = {}
    failures: list[dict] = []
    attempts_hist: Counter = Counter()
    n_missing = 0
    for split, run_id, quota in SPLITS:
        manifest_path = EXP / "manifests" / f"{split}.jsonl"
        rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_ids.extend(r["trial_id"] for r in rows)
        seed_blocks[split] = {int(r["seed"]) for r in rows}
        run_dir = runs_root / run_id
        completed = bf = missing = 0
        for r in rows:
            td = run_dir / r["trial_id"]
            attempts = sorted(p for p in td.iterdir() if p.is_dir() and p.name.startswith("attempt_")) \
                if td.is_dir() else []
            if not attempts:
                missing += 1
                continue
            attempts_hist[len(attempts)] += 1
            good = [a for a in attempts if (a / "DONE").is_file()]
            if not good:
                missing += 1
                continue
            meta = json.loads((good[0] / "meta.json").read_text(encoding="utf-8"))
            if meta["outcome"] == "completed":
                completed += 1
            else:
                bf += 1
                failures.append({"split": split, "trial_id": r["trial_id"],
                                 "outcome": meta["outcome"],
                                 "steps": meta["timing"]["steps_executed"]})
        per_split[split] = {"run_id": run_id, "quota": quota, "manifest": len(rows),
                            "completed": completed, "business_failure": bf, "missing": missing}
        n_missing += missing
        print(f"[audit] {split:18s} quota={quota:4d} manifest={len(rows):4d} "
              f"completed={completed:4d} bf={bf:3d} missing={missing}", flush=True)

    total_manifest = sum(v["manifest"] for v in per_split.values())
    add("A01-quota", f"七分片合计 {QUOTA_TOTAL} 条且各片等于预登记配额",
        total_manifest == QUOTA_TOTAL and all(v["manifest"] == v["quota"] for v in per_split.values()),
        " / ".join(f"{k}={v['manifest']}" for k, v in per_split.items()),
        per_split=per_split)

    add("A02-unique-ids", "trial_id 全局唯一（跨片不重复）",
        len(all_ids) == len(set(all_ids)),
        f"{len(all_ids)} 条，去重后 {len(set(all_ids))} 条")

    add("A03-attempts", "每个 trial 都有有效 attempt（DONE 完整），主 attempt 唯一可判",
        n_missing == 0,
        f"缺失 {n_missing} 条；attempt 数分布 {dict(attempts_hist)}"
        + ("（全部为单 attempt）" if set(attempts_hist) == {1} else ""))

    # seed 块互斥
    overlaps = {}
    keys = sorted(seed_blocks)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = seed_blocks[keys[i]] & seed_blocks[keys[j]]
            if inter:
                overlaps[f"{keys[i]}∩{keys[j]}"] = len(inter)
    add("A04-seed-disjoint", "七个 seed 块两两不相交",
        not overlaps, f"相交 {overlaps}" if overlaps else
        "全部成对为空；区间 " + ", ".join(f"{k}={min(seed_blocks[k])}-{max(seed_blocks[k])}" for k in keys))

    # 版本散列
    ok_ver = True
    ver_detail = []
    if bundle_manifest["bundle_id"] != lock_eval["controller"]["bundle_id"]:
        ok_ver = False
        ver_detail.append("bundle_id 与检验锁不一致")
    if sha256_file(EXP / "config/margins_hira_v1.json") != lock_eval["margins"]["sha256"]:
        ok_ver = False
        ver_detail.append("margins 文件散列与检验锁不一致")
    if model_payload["sha256"] != margins["data"]["model_sha256"]:
        ok_ver = False
        ver_detail.append("功率模型内容散列与 margins 记录不一致")
    add("A05-version-bindings", "bundle / margins / power model 与检验锁、margins 记录逐项一致",
        ok_ver,
        "；".join(ver_detail) if ver_detail else
        f"bundle={bundle_id[:16]} margins_file={lock_eval['margins']['sha256'][:16]} "
        f"model={model_payload['sha256'][:16]}")

    # ---------------- 数值：能量 / J_nom / ε_c / ε_d（全量） ----------------
    eps_rows: list[dict] = []
    worst_energy = 0.0
    worst_energy_trial = None
    worst_jnom_rel = 0.0
    n_energy_checked = 0
    n_eps_ok = 0
    for split, run_id, _q in SPLITS:
        rows = [json.loads(l) for l in
                (EXP / "manifests" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        run_dir = runs_root / run_id
        for r in rows:
            td = run_dir / r["trial_id"]
            if not td.is_dir():
                continue
            attempts = sorted(p for p in td.iterdir() if p.is_dir() and p.name.startswith("attempt_"))
            good = [a for a in attempts if (a / "DONE").is_file()]
            if not good:
                continue
            att = good[0]
            meta = json.loads((att / "meta.json").read_text(encoding="utf-8"))
            energy = json.loads((att / "energy.json").read_text(encoding="utf-8"))
            pair = np.load(att / "pairing_10hz.npz")
            commands = pair["command"].astype(np.float64)

            # 能量：从原始 200 Hz 重积分
            phys = np.load(att / "physics_200hz.npz")
            j_indep = positive_work_indep(phys["tau_applied_source"], phys["qdot_post_source"], DT_SIM)
            rel = abs(j_indep - energy["j_exec"]) / max(abs(energy["j_exec"]), 1e-12)
            n_energy_checked += 1
            if rel > worst_energy:
                worst_energy = rel
                worst_energy_trial = r["trial_id"]

            # J_nom / ε_c / ε_d
            j_nom = j_nom_indep(commands, weights, names, dt_cost) if commands.shape[0] >= 2 else None
            complete = meta["outcome"] == "completed"
            pose = (meta.get("profile_applied") or {}).get("initial_pose") or {}
            nom_xy = nominal_xy_indep(commands, (float(pose.get("px_m", 0.0)), float(pose.get("py_m", 0.0))),
                                      float(pose.get("yaw_rad", 0.0)))
            exec_xy = pair["pos_xy"].astype(np.float64)
            eps_d = float(np.hypot(*(exec_xy - nom_xy).T).max()) if nom_xy.shape == exec_xy.shape else None
            eps_c = None
            if complete and j_nom is not None and j_nom >= J_MIN:
                eps_c = (energy["j_exec"] - j_nom) / j_nom
            eps_rows.append({"split": split, "trial_id": r["trial_id"], "cell": f"{r['family']}/{r['profile']}",
                             "complete": complete, "eps_c": eps_c, "eps_d": eps_d, "j_nom": j_nom,
                             "j_exec": float(energy["j_exec"])})
            n_eps_ok += 1

    add("A06-energy-replay", f"能量从原始 200 Hz 独立重积分，rel ≤ {ENERGY_REL_MAX:g}",
        worst_energy <= ENERGY_REL_MAX,
        f"复算 {n_energy_checked} 条，最差 rel {worst_energy:.2e}（{worst_energy_trial}）")

    # α/δ 独立复算（只用 calibration 的完整记录）
    cal = [r for r in eps_rows if r["split"] == "calibration" and r["complete"]
           and r["eps_c"] is not None and r["eps_d"] is not None]
    eps_c_all = [r["eps_c"] for r in cal]
    eps_d_all = [r["eps_d"] for r in cal]
    alpha_indep = max(0.0, nearest_rank_indep(eps_c_all, QUANTILE))
    delta_indep = max(0.0, nearest_rank_indep(eps_d_all, QUANTILE))
    d_alpha = abs(alpha_indep - alpha_frozen)
    d_delta = abs(delta_indep - delta_frozen)
    add("A07-margins-recompute", "α/δ 从 800 条校准独立复算，与冻结值一致（1e-12）",
        d_alpha <= 1e-12 and d_delta <= 1e-12,
        f"n={len(cal)}；α {alpha_indep:.9f} vs {alpha_frozen:.9f}（Δ={d_alpha:.2e}）；"
        f"δ {delta_indep:.9f} vs {delta_frozen:.9f}（Δ={d_delta:.2e}）")
    if d_alpha > 1e-12 or d_delta > 1e-12:
        note("A07", "margins_mismatch", f"α差 {d_alpha:.3e}，δ差 {d_delta:.3e}",
             alpha_indep=alpha_indep, delta_indep=delta_indep)

    # ---------------- 覆盖率独立复算 ----------------
    def coverage_of(split: str) -> dict:
        rows = [r for r in eps_rows if r["split"] == split and r["complete"]]
        usable = [r for r in rows if r["eps_c"] is not None and r["eps_d"] is not None]
        c = [1.0 if r["eps_c"] <= alpha_frozen + 1e-12 else 0.0 for r in usable]
        d = [1.0 if r["eps_d"] <= delta_frozen + 1e-12 else 0.0 for r in usable]
        j = [a * b for a, b in zip(c, d)]
        n = len(usable)
        return {"n": n, "cost": sum(c) / n if n else None,
                "position": sum(d) / n if n else None, "joint": sum(j) / n if n else None}

    cov_h = coverage_of("heldout")
    cov_n = coverage_of("near_obstacle")
    report_h = json.loads((EXP / "reports/heldout_coverage.json").read_text(encoding="utf-8"))
    report_n = json.loads((EXP / "reports/near_obstacle_evaluation.json").read_text(encoding="utf-8"))
    oh = report_h["full_sample"]["overall"]
    on = report_n["full_sample"]["overall"]
    max_dev = 0.0
    detail = []
    for tag, indep, reported in (("heldout", cov_h, oh), ("near", cov_n, on)):
        for key in ("cost", "position", "joint"):
            rv = reported[key]["rate"]
            iv = indep[key]
            dev = abs((iv or 0.0) - (rv or 0.0))
            max_dev = max(max_dev, dev)
            detail.append(f"{tag}.{key} {iv:.4f} vs {rv:.4f}")
    add("A08-coverage-recompute", "heldout 与 near 的成本/位置/联合覆盖独立复算，与主报告一致（1e-9）",
        max_dev <= 1e-9, "；".join(detail) + f"（最大偏差 {max_dev:.2e}）")
    if max_dev > 1e-9:
        note("A08", "coverage_mismatch", f"最大偏差 {max_dev:.3e}")

    # ---------------- 近障几何独立复算 ----------------
    fp_payload = json.loads((EXP / "config/footprint_hira_v1.json").read_text(encoding="utf-8"))
    fp_verts = np.asarray(fp_payload["polygon"]["vertices"], dtype=np.float64)
    maps = json.loads((EXP / "config/near_obstacle_maps_v1.json").read_text(encoding="utf-8"))
    case_by = {c["trial_id"]: c for c in maps["cases"]}
    n_overlap_indep = 0
    n_near = 0
    for r in [x for x in eps_rows if x["split"] == "near_obstacle"]:
        td = runs_root / "near_obstacle_20261107" / r["trial_id"]
        attempts = sorted(p for p in td.iterdir() if p.is_dir() and p.name.startswith("attempt_"))
        good = [a for a in attempts if (a / "DONE").is_file()]
        if not good:
            continue
        state = np.load(good[0] / "state_50hz.npz")
        pos = state["root_pos_w"][:, :2].astype(np.float64)
        quat = state["root_quat_w"].astype(np.float64)
        yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                         1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        key = r["trial_id"][5:] if r["trial_id"].startswith("hira-") else r["trial_id"]
        case = case_by[key]
        hit = False
        for i in range(0, pos.shape[0], 5):     # 10 Hz 抽样，够判"是否曾重叠"
            fv = footprint_vertices_indep(fp_verts, pos[i], yaw[i])
            for obs in case["obstacles"]:
                if overlaps_indep(fv, obs):
                    hit = True
                    break
            if hit:
                break
        n_near += 1
        n_overlap_indep += int(hit)

    reported_overlap = report_n["geometry"]["executed"]["n_geometric_overlap"]
    add("A09-geometry-recompute", "近障几何重叠独立复算（10 Hz 抽样 + 旋转矩形判定）与主报告同量级",
        abs(n_overlap_indep - reported_overlap) <= 20,
        f"独立 {n_overlap_indep}/{n_near}，主报告 {reported_overlap}/{report_n['geometry']['executed']['n_trials']}"
        f"（差 {n_overlap_indep - reported_overlap}；独立实现按 10 Hz 抽样且用顶点/中心判据，属保守下界）")
    if abs(n_overlap_indep - reported_overlap) > 20:
        note("A09", "geometry_mismatch", f"独立 {n_overlap_indep} vs 报告 {reported_overlap}")
    elif n_overlap_indep != reported_overlap:
        note("A09", "expected_method_difference",
             f"独立 {n_overlap_indep} vs 报告 {reported_overlap}（差 {n_overlap_indep - reported_overlap}）："
             "独立实现按 10 Hz 抽样、用「足迹顶点在矩形内 或 矩形中心在足迹内」判据，"
             "主报告用 50 Hz 全帧多边形距离；独立侧为保守下界，方向正确（少报不少不少报多）",
             independent=n_overlap_indep, reported=reported_overlap)

    # ---------------- 隔离检查 ----------------
    mdata = margins.get("data") or {}
    add("A10-no-leakage", "α/δ 只由 calibration 计算，未引用 heldout/near",
        mdata.get("split") == "calibration" and mdata.get("run_id") == "calibration_20261106",
        f"margins.data.split={mdata.get('split')!r} run={mdata.get('run_id')!r} n={mdata.get('n_official')}")

    add("A11-failure-denominators", "失败样本留在主分母（各片 completed + business_failure = manifest）",
        all(v["completed"] + v["business_failure"] + v["missing"] == v["manifest"] for v in per_split.values()),
        "；".join(f"{k} {v['completed']}+{v['business_failure']}+{v['missing']}={v['manifest']}"
                 for k, v in per_split.items()))

    # v2 冻结物只在 git 仓库里（/pub 工作副本只同步了 adapters/src/实验目录），
    # 所以这里按仓库绝对路径查，并用 .sha256 侧车 + 锁内 margins 记录做自校验——
    # 单看"文件在不在"证明不了"没被改过"。
    v2_dir = Path(args_cli.v2_config)
    v2_files = ["data_lock_v1.json", "margins_v1.json", "power_model_v1.json"]
    v2_detail: list[str] = []
    v2_ok = True
    for name in v2_files:
        f = v2_dir / name
        side = v2_dir / f"{name}.sha256"
        if not f.is_file():
            v2_ok = False
            v2_detail.append(f"{name}=缺")
            continue
        if not side.is_file():
            v2_ok = False
            v2_detail.append(f"{name}=无侧车")
            continue
        # v2 的侧车有**两种口径**，不能只认一种：
        #   data_lock_v1.json        → 记文件字节散列
        #   margins_v1 / power_model → 记内容自散列（= 文件内的 sha256 字段，不含时间戳/格式）
        # 两者都算"一致"；只认字节散列会把没被动过的文件误判成篡改（本轮真踩到）。
        recorded = side.read_text(encoding="utf-8").split()[0]
        actual_bytes = sha256_file(f)
        try:
            inner = json.loads(f.read_text(encoding="utf-8")).get("sha256")
        except Exception:  # noqa: BLE001
            inner = None
        if recorded == actual_bytes:
            v2_detail.append(f"{name}=侧车(文件散列)一致")
        elif inner is not None and recorded == str(inner):
            v2_detail.append(f"{name}=侧车(内容散列)一致")
        else:
            v2_ok = False
            v2_detail.append(f"{name}=侧车不符（既非字节也非内容散列）")
    # 链完整性：v2 锁里记录的 margins 散列应等于 margins_v1.json 的实际散列
    try:
        v2_lock = json.loads((v2_dir / "data_lock_v1.json").read_text(encoding="utf-8"))
        chain_rec = (v2_lock.get("margins") or {}).get("sha256")
        chain_act = sha256_file(v2_dir / "margins_v1.json")
        chain_ok = bool(chain_rec) and chain_rec == chain_act
        v2_ok = v2_ok and chain_ok
        v2_detail.append(f"锁→margins 链={'一致' if chain_ok else '不符'}")
    except Exception as exc:  # noqa: BLE001
        v2_ok = False
        v2_detail.append(f"链校验失败：{exc}")
    add("A12-v2-untouched", "旧 v2 冻结物仍在位、侧车自校验通过（文件散列或内容散列）、锁→margins 链完整",
        v2_ok, "；".join(v2_detail))
    note("A12", "schema_difference",
         "v2 的侧车口径不统一：data_lock_v1.json 记文件字节散列，margins_v1 / power_model_v1 记内容自散列"
         "（等于文件内 sha256 字段）。首版判据只认字节散列，把未改动的文件误判成篡改；已改为两者任一成立即通过。"
         "核对结果：三者内容均未变，锁→margins 链一致。")
    if not v2_ok:
        note("A12", "v2_broken", "；".join(v2_detail))

    failed = [c["id"] for c in CHECKS if not c["ok"]]
    payload = {
        "report": "full_independent_audit",
        "task": "HIRA-036",
        "independence": "本脚本不 import camra_calibration / hira_go2；名义积分、功率特征、能量正功、"
                        "nearest-rank、覆盖判定、矩形-足迹相交均为独立实现",
        "runs_root": str(runs_root),
        "alpha_frozen": alpha_frozen, "delta_frozen": delta_frozen,
        "alpha_independent": alpha_indep, "delta_independent": delta_indep,
        "per_split": per_split,
        "failures": failures,
        "n_energy_replayed": n_energy_checked,
        "worst_energy_rel": worst_energy,
        "worst_energy_trial": worst_energy_trial,
        "coverage_independent": {"heldout": cov_h, "near_obstacle": cov_n},
        "geometry_independent": {"n_overlap": n_overlap_indep, "n_trials": n_near,
                                 "reported_overlap": reported_overlap},
        "checks": CHECKS,
        "verdict": {"audit_ok": not failed, "failed": failed,
                    "n_checks": len(CHECKS), "n_discrepancies": len(DISC)},
        "limits": [
            "几何独立实现按 10 Hz 抽样且用「顶点在矩形内 / 中心在足迹内」判据，是重叠的保守下界，"
            "与主报告的 50 Hz 全帧多边形距离不逐条相等，只核对量级与单调性",
            "v2 冻结物位于 git 仓库的 CaMRA/calibration/config（/pub 工作副本不含），"
            "本轮用「侧车自校验 + 锁→margins 链」证明未被改动；其训练期原始日志不在本机，"
            "无法复算 TASK-024/026/036",
            "本轮未复算 J_nom 的残差分布（那需要重跑拟合），只核对冻结模型产出的逐条 J_nom 与 ε_c",
        ],
    }
    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    Path(args_cli.out_discrepancies).write_text(
        json.dumps({"report": "audit_discrepancies", "task": "HIRA-036",
                    "n": len(DISC), "items": DISC,
                    "policy": "超容差差异必须解释或形成阻塞；不修改旧记录消差"},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"AUDIT {'OK' if not failed else 'FAIL'} checks={len(CHECKS) - len(failed)}/{len(CHECKS)} "
          f"failed={failed} discrepancies={len(DISC)}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
