"""只复跑 nominal_violation 的逐岛对照（上一版的对照逻辑把"多岛 OR"当成了单岛布尔）。"""
import os, sys, time, math
import numpy as np
ROOT = "/home/caohy/repositories/Paper/CaMRA/navigation"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, "/home/caohy/repositories/Paper/CaMRA/calibration/src")
from camra_navigation.maps import generator as M
from camra_navigation.maps.geometry import SafetyGeometry, segment_intersects_polygon
from camra_calibration.nominal import step_exact, PlanarState

NAV = ROOT
CAL = "/home/caohy/repositories/Paper/CaMRA/calibration"
design = M.load_design(os.path.join(NAV, "config", "map_design_v1.json"))
geom = SafetyGeometry.from_frozen(os.path.join(CAL, "config", "footprint_v1.json"),
                                  os.path.join(CAL, "config", "margins_v1.json"),
                                  os.path.join(NAV, "config", "geometry_contract_v1.json"))
spec = M.build("L3_multi_path", 1001, design)
infl = [np.asarray(ob.polygon().inflate(geom.inflation_radius_m).vertices, dtype=np.float64)
        for ob in spec.obstacles()]
print("islands:", len(infl), "verts:", [a.shape[0] for a in infl])

N_CASE = 2000
rng = np.random.default_rng(2026)
pos0 = rng.uniform(-8, 8, size=(N_CASE, 2))
yaw0 = rng.uniform(-math.pi, math.pi, size=N_CASE)
cmd = np.stack([rng.uniform(-1, 1, N_CASE), rng.uniform(-0.4, 0.4, N_CASE),
                rng.uniform(-1, 1, N_CASE)], axis=1)
p1 = np.empty_like(pos0)
for i in range(N_CASE):
    r = step_exact(PlanarState(float(pos0[i, 0]), float(pos0[i, 1]), float(yaw0[i])),
                   tuple(cmd[i]), 0.1)
    p1[i] = (r.px, r.py)

def mine_island(p0a, p1a, q):
    d = p1a - p0a
    e = np.roll(q, -1, axis=0) - q
    axes = np.concatenate([d[:, None, :], np.broadcast_to(e[None], (d.shape[0],) + e.shape)], axis=1)
    pr0 = np.einsum("nk,nak->na", p0a, axes)
    pr1 = np.einsum("nk,nak->na", p1a, axes)
    prq = np.einsum("qk,nak->nqa", q, axes)
    loP, hiP = np.minimum(pr0, pr1), np.maximum(pr0, pr1)
    sep = (hiP < prq.min(axis=1)) | (prq.max(axis=1) < loP)
    return ~sep.any(axis=1)

mine = np.zeros((N_CASE, len(infl)), dtype=bool)
for j, q in enumerate(infl):
    mine[:, j] = mine_island(pos0, p1, q)

frozen = np.zeros_like(mine)
t0 = time.time()
for i in range(N_CASE):
    a = (float(pos0[i, 0]), float(pos0[i, 1]))
    b = (float(p1[i, 0]), float(p1[i, 1]))
    for j, ip in enumerate([ob.polygon().inflate(geom.inflation_radius_m) for ob in spec.obstacles()]):
        frozen[i, j] = bool(segment_intersects_polygon(a, b, ip))
dt = time.time() - t0
mism = int((frozen != mine).sum())
print("对照 %d 例 × %d 岛，frozen 用时 %.1fs（%.2f ms/例）" % (N_CASE, len(infl), dt, 1000 * dt / N_CASE))
print("布尔不一致：", mism, "/", frozen.size, " (%.2f%%)" % (100 * mism / frozen.size))
tab = {}
for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
    tab["mine=%d frozen=%d" % (a, b)] = int(((mine[:, 0] == bool(a)) & (frozen[:, 0] == bool(b))).sum())
print("岛0 的 (mine,frozen) 分布：", tab)
