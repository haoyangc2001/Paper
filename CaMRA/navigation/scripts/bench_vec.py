"""NAV-036 路线 C：几何批处理的速度原型。

目的：实测"批量几何"能不能把名义环境推到 47.4 步/s（授权配额 102.4 万步 / 6 h）。

替换的两条热点路径（数学上都精确，不只是近似）：
  1) swept-hull 碰撞：frozen 先做 hull(F0∪F1) 再做 SAT。凸包的边法向是 F0/F1 边法向的
     子集，所以直接对 F0、F1 的全部边法向做 SAT 得到的是**同一个布尔结果**（多出来的
     轴只可能更早发现分离，不可能把相交误判成分离）——省掉两次 hull 和一次 inflate。
  2) nominal_violation 的 inflate：inflation_radius 对整个回合是常数，frozen 每步重算，
     这里预膨胀一次缓存。

对照：与 frozen 路径在同样的随机状态上逐例比较布尔结果，统计不一致数。
"""

import os
import sys
import time
import math
import json

import numpy as np

ROOT = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/home/caohy/repositories/Paper/CaMRA/calibration/src")

from camra_navigation.maps import generator as M
from camra_navigation.maps.geometry import SafetyGeometry
from camra_navigation.envs.nominal import wall_polygons
from camra_navigation.maps.geometry import (
    convex_polygons_intersect,
    footprint_at,
    segment_intersects_polygon,
)
from camra_calibration.geometry import Polygon

NAV = ROOT
CAL = "/home/caohy/repositories/Paper/CaMRA/calibration"
design = M.load_design(os.path.join(NAV, "config", "map_design_v1.json"))
geom = SafetyGeometry.from_frozen(os.path.join(CAL, "config", "footprint_v1.json"),
                                  os.path.join(CAL, "config", "margins_v1.json"),
                                  os.path.join(NAV, "config", "geometry_contract_v1.json"))


def verts(poly):
    return np.asarray(poly.vertices, dtype=np.float64)


def rot(v, yaw):
    """v:(k,2) 或 (N,k,2)；yaw 标量或 (N,)"""
    c, s = math.cos(yaw) if np.isscalar(yaw) else np.cos(yaw), None
    if np.isscalar(yaw):
        R = np.array([[c, -math.sin(yaw)], [math.sin(yaw), c]])
        return v @ R.T
    c = np.cos(yaw)[:, None]
    s = np.sin(yaw)[:, None]
    out = np.empty((v.shape[0], v.shape[1], 2))
    out[:, :, 0] = c * v[None, :, 0] - s * v[None, :, 1]
    out[:, :, 1] = s * v[None, :, 0] + c * v[None, :, 1]
    return out


def sat_axes(P0, P1, Q):
    """P0,P1:(N,kP,2) 两个位姿的足迹；Q:(N,kQ,2) 或 (kQ,2) 障碍物。返回逐 env 是否相交。"""
    if Q.ndim == 2:
        Q = np.broadcast_to(Q[None], (P0.shape[0], Q.shape[0], 2))
    def normals(P):
        e = np.roll(P, -1, axis=1) - P
        n = np.stack([e[:, :, 1], -e[:, :, 0]], axis=-1)
        return n
    axes = np.concatenate([normals(P0), normals(P1), normals(Q)], axis=1)  # (N,kA,2)
    projP0 = np.einsum("npk,nak->npa", P0, axes)
    projP1 = np.einsum("npk,nak->npa", P1, axes)
    projQ = np.einsum("nqk,nak->nqa", Q, axes)
    loP = np.minimum(projP0.min(axis=1), projP1.min(axis=1))
    hiP = np.maximum(projP0.max(axis=1), projP1.max(axis=1))
    loQ = projQ.min(axis=1)
    hiQ = projQ.max(axis=1)
    separated = (hiP < loQ) | (hiQ < loP)
    return ~separated.any(axis=1)


def seg_vs_poly(p0, p1, Q):
    """线段与凸多边形相交（batched）：线段方向 + 多边形边法向作为轴。"""
    d = p1 - p0                                   # (N,2)
    e = np.roll(Q, -1, axis=1) - Q                # (N,k,2)
    axes = np.concatenate([d[:, None, :], e], axis=1)
    t = np.einsum("nk,nk->n", d, d)
    projP0 = np.einsum("nk,nak->na", p0, axes)
    projP1 = np.einsum("nk,nak->na", p1, axes)
    projQ = np.einsum("nqk,nak->nqa", Q, axes)
    loP = np.minimum(projP0, projP1)
    hiP = np.maximum(projP0, projP1)
    loQ, hiQ = projQ.min(axis=1), projQ.max(axis=1)
    separated = (hiP < loQ) | (hiQ < loP)
    return ~separated.any(axis=1)


def main():
    N = 32
    rng = np.random.default_rng(2026)
    print("== 准备地形 ==")
    terr = {}
    for level in ("L1_empty", "L2_single_detour", "L3_multi_path"):
        spec = M.build(level, 1001, design)
        polys = [ob.polygon() for ob in spec.obstacles()]
        walls = [verts(w) for w in wall_polygons(spec.extent)]
        # 预膨胀一次（frozen 每步都重算）
        infl = [verts(p.inflate(geom.inflation_radius_m)) for p in polys]
        terr[level] = {"islands_n": len(infl), "walls_n": len(walls),
                       "infl": infl, "walls": walls, "extent": spec.extent}
        print(" ", level, "islands", len(infl), "walls", len(walls))

    # 随机状态批量
    print("== 生成 %d 个随机 env-step ==" % (N * 200))
    pos0 = rng.uniform(-8, 8, size=(N * 200, 2))
    yaw0 = rng.uniform(-math.pi, math.pi, size=N * 200)
    cmd = np.stack([rng.uniform(-1, 1, N * 200), rng.uniform(-0.4, 0.4, N * 200),
                    rng.uniform(-1, 1, N * 200)], axis=1)
    # 用 frozen 积分器推进 0.1 s
    from camra_calibration.nominal import step_exact
    from camra_calibration.nominal import PlanarState
    p1 = np.empty_like(pos0)
    y1 = np.empty_like(yaw0)
    t0 = time.time()
    for i in range(len(pos0)):
        r = step_exact(PlanarState(float(pos0[i, 0]), float(pos0[i, 1]), float(yaw0[i])),
                       tuple(cmd[i]), 0.1)
        p1[i] = (r.px, r.py)
        y1[i] = r.yaw
    t_int = time.time() - t0
    print("  积分 %d 步 %.2fs（单步 %.2f ms）" % (len(pos0), t_int, 1000 * t_int / len(pos0)))

    wall_v = np.stack([np.asarray(w, dtype=np.float64) for w in terr["L3_multi_path"]["walls"]])
    infl_v = [np.asarray(a, dtype=np.float64) for a in terr["L3_multi_path"]["infl"]]

    # ---- 批处理几何：把 N*200 个状态切成 N 一批，逐步走 10 步 ----
    def batched_pass(states0, states1):
        n = states0.shape[0]
        hits = np.zeros(n, dtype=bool)
        viol = np.zeros(n, dtype=bool)
        for s in range(0, n, N):
            sl = slice(s, min(s + N, n))
            p0, y0 = states0[sl, :2], states0[sl, 2]
            p1s, y1s = states1[sl, :2], states1[sl, 2]
            c = np.stack([np.cos(y0), -np.sin(y0), np.sin(y0), np.cos(y0)], axis=1).reshape(-1, 2, 2)
            fp0 = np.einsum("nlj,nkj->nkl", c, np.broadcast_to(geom.footprint.vertices[None], (c.shape[0],) + geom.footprint.vertices.shape)) + p0[:, None, :]
            c1 = np.stack([np.cos(y1s), -np.sin(y1s), np.sin(y1s), np.cos(y1s)], axis=1).reshape(-1, 2, 2)
            fp1 = np.einsum("nlj,nkj->nkl", c1, np.broadcast_to(geom.footprint.vertices[None], (c1.shape[0],) + geom.footprint.vertices.shape)) + p1s[:, None, :]
            for w in wall_v:
                hits[sl] |= sat_axes(fp0, fp1, w)
            for q in infl_v:
                d = p1s - p0
                e = np.roll(q, -1, axis=0) - q
                axes = np.concatenate([d[:, None, :], np.broadcast_to(e[None], (d.shape[0],) + e.shape)], axis=1)
                pr0 = np.einsum("nk,nak->na", p0, axes)
                pr1 = np.einsum("nk,nak->na", p1s, axes)
                prq = np.einsum("qk,nak->nqa", q, axes)
                loP, hiP = np.minimum(pr0, pr1), np.maximum(pr0, pr1)
                sep = (hiP < prq.min(axis=1)) | (prq.max(axis=1) < loP)
                viol[sl] |= (~sep.any(axis=1))
        return hits, viol

    t0 = time.time()
    S0 = np.concatenate([pos0, yaw0[:, None]], axis=1)
    S1 = np.concatenate([p1, y1[:, None]], axis=1)
    h, v = batched_pass(S0, S1)
    dt_batch = time.time() - t0
    per_step = dt_batch / len(pos0) * 1000
    print("== 批处理几何（墙 4 + 预膨胀岛 %d，N=%d）==" % (len(infl_v), N))
    print("  %d 个 env-step 用时 %.3fs -> %.3f ms/步（等效 %.0f 步/s）"
          % (len(pos0), dt_batch, per_step, 1000.0 / per_step))

    # ---- frozen 对照（同样状态，逐个）----
    from camra_navigation.frames import wrap_angle
    walls_poly = [Polygon(w) for w in terr["L3_multi_path"]["walls"]]
    infl_poly = [Polygon(a) for a in infl_v]
    # 逐多边形对照（第一版把"多岛 OR"当成单岛布尔，会制造假不一致）
    h_isl = np.zeros((len(pos0), len(walls_poly)), dtype=bool)
    v_isl = np.zeros((len(pos0), len(infl_poly)), dtype=bool)
    for s in range(0, len(pos0), N):
        sl = slice(s, min(s + N, len(pos0)))
        p0, y0 = S0[sl, :2], S0[sl, 2]
        p1s, y1s = S1[sl, :2], S1[sl, 2]
        c = np.stack([np.cos(y0), -np.sin(y0), np.sin(y0), np.cos(y0)], axis=1).reshape(-1, 2, 2)
        fp0 = np.einsum("nlj,nkj->nkl", c, np.broadcast_to(geom.footprint.vertices[None], (c.shape[0],) + geom.footprint.vertices.shape)) + p0[:, None, :]
        c1 = np.stack([np.cos(y1s), -np.sin(y1s), np.sin(y1s), np.cos(y1s)], axis=1).reshape(-1, 2, 2)
        fp1 = np.einsum("nlj,nkj->nkl", c1, np.broadcast_to(geom.footprint.vertices[None], (c1.shape[0],) + geom.footprint.vertices.shape)) + p1s[:, None, :]
        for wi, w in enumerate(wall_v):
            h_isl[sl, wi] = sat_axes(fp0, fp1, w)
        for qi, q in enumerate(infl_v):
            d = p1s - p0
            e = np.roll(q, -1, axis=0) - q
            axes = np.concatenate([d[:, None, :], np.broadcast_to(e[None], (d.shape[0],) + e.shape)], axis=1)
            pr0 = np.einsum("nk,nak->na", p0, axes)
            pr1 = np.einsum("nk,nak->na", p1s, axes)
            prq = np.einsum("qk,nak->nqa", q, axes)
            loP, hiP = np.minimum(pr0, pr1), np.maximum(pr0, pr1)
            v_isl[sl, qi] = (~((hiP < prq.min(axis=1)) | (prq.max(axis=1) < loP)).any(axis=1))
    t0 = time.time()
    mism_h = mism_v = 0
    for i in range(len(pos0)):
        p0 = (float(pos0[i, 0]), float(pos0[i, 1]))
        p1s = (float(p1[i, 0]), float(p1[i, 1]))
        y0, y1s = float(yaw0[i]), float(y1[i])
        f0 = footprint_at(geom, p0, y0)
        f1 = footprint_at(geom, p1s, y1s)
        for wi, wp in enumerate(walls_poly):
            swept = f0.union(f1)
            dpsi = abs(wrap_angle(y1s - y0))
            if dpsi > 0.0:
                swept = swept.inflate(geom.r_fp_m * dpsi / 2.0)
            if bool(convex_polygons_intersect(swept, wp)) != bool(h_isl[i, wi]):
                mism_h += 1
                break
        for qi, ip in enumerate(infl_poly):
            if bool(segment_intersects_polygon(p0, p1s, ip)) != bool(v_isl[i, qi]):
                mism_v += 1
                break
    dt_frozen = time.time() - t0
    per_frozen = dt_frozen / len(pos0) * 1000
    print("== frozen 对照 ==")
    print("  %d 个 env-step 用时 %.1fs -> %.1f ms/步（等效 %.0f 步/s）"
          % (len(pos0), dt_frozen, per_frozen, 1000.0 / per_frozen))
    print("  布尔不一致：墙碰撞 %d / %d，越界 %d / %d" % (mism_h, len(pos0), mism_v, len(pos0)))
    print("== 结论 ==")
    print("  需要的吞吐 47.4 步/s；当前标量全链路 9.1–15.6 步/s；")
    print("  本原型只测几何段：%.3f ms/步 vs frozen %.1f ms/步（%.0f 倍）"
          % (per_step, per_frozen, per_frozen / max(per_step, 1e-9)))


if __name__ == "__main__":
    main()
