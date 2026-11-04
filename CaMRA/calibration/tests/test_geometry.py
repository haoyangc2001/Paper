"""CAL-021 几何与占地轮廓测试（纯 CPU，无仿真依赖）。

覆盖验收要求的四类几何情形：旋转、窄通道、擦边、连续扫掠；
另加资产运动学自洽、FK 批量一致性、轮廓包含性与"delta 只膨胀一次"。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from camra_calibration.footprint import (
    AssetGeometry, Footprint, FootprintError, POLICY_JOINT_ORDER, coverage,
    init_joint_pose, joint_ranges, outline_from_q,
)
from camra_calibration.geometry import (
    Polygon, convex_hull_2d, disc, minkowski_sum, rotating_sweep_hull, swept_collision,
)

CAL = Path(__file__).resolve().parent.parent
ASSET = CAL / "config" / "go2_asset_geometry_v1.json"


def rect(w: float, h: float) -> Polygon:
    return Polygon(np.array([[w / 2, h / 2], [-w / 2, h / 2],
                             [-w / 2, -h / 2], [w / 2, -h / 2]]))


# ---------------------------------------------------------------- 凸包与基本量
def test_convex_hull_drops_interior_points():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.5, 0.1]])
    hull = convex_hull_2d(pts)
    assert hull.shape[0] == 4
    assert abs(Polygon(hull).area - 1.0) < 1e-12


def test_hull_is_ccw_and_convex():
    rng = np.random.default_rng(0)
    poly = Polygon(rng.normal(size=(200, 2)))
    assert poly.area > 0
    v, nxt = poly.vertices, np.roll(poly.vertices, -1, axis=0)
    cross = np.cross(nxt - v, np.roll(nxt, -1, axis=0) - nxt)
    assert (cross >= -1e-12).all()


def test_area_width_radius_consistency():
    r = rect(1.0, 0.4)
    assert r.area == pytest.approx(0.4, rel=1e-12)
    assert r.min_width() == pytest.approx(0.4, rel=1e-9)
    assert r.circumscribed_radius == pytest.approx(math.hypot(0.5, 0.2), rel=1e-12)
    assert r.width_along((0, 1)) == pytest.approx(0.4, rel=1e-12)
    assert r.width_along((1, 0)) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------- 旋转
def test_rotation_preserves_area_and_radius():
    r = rect(1.0, 0.4)
    for deg in (0, 17, 45, 90, 137):
        q = r.rotate(math.radians(deg))
        assert q.area == pytest.approx(r.area, rel=1e-9)
        assert q.circumscribed_radius == pytest.approx(r.circumscribed_radius, rel=1e-9)


def test_rotation_swaps_extents_at_90deg():
    r = rect(1.0, 0.4).rotate(math.pi / 2)
    x0, x1, y0, y1 = r.bounds()
    assert (x1 - x0) == pytest.approx(0.4, rel=1e-9)
    assert (y1 - y0) == pytest.approx(1.0, rel=1e-9)


def test_rotation_changes_required_corridor_width():
    """窄通道：同一机器人在不同朝向下需要的通道宽度不同。"""
    r = rect(1.0, 0.4)
    normal = (0.0, 1.0)
    w0 = r.rotate(0.0).width_along(normal)
    w45 = r.rotate(math.radians(45)).width_along(normal)
    w90 = r.rotate(math.radians(90)).width_along(normal)
    assert w0 == pytest.approx(0.4, rel=1e-9)
    assert w90 == pytest.approx(1.0, rel=1e-9)
    assert w45 > w0 and w45 < w90


# ---------------------------------------------------------------- 窄通道
def test_narrow_corridor_admits_and_rejects():
    fp = rect(0.87, 0.57)                      # 与实测轮廓同量级
    need = fp.min_width()
    assert need == pytest.approx(0.57, rel=1e-9)
    for width, expect in ((need - 0.01, False), (need + 0.01, True), (need + 0.5, True)):
        # 通道宽度 = 两侧障碍间距；机器人可过 ⟺ 所需宽度 ≤ 通道宽度
        assert (need <= width) is expect


def test_min_width_ignores_degenerate_edges():
    """重复顶点曾经让 min_width 返回 0，必须仍给出真实宽度。"""
    poly = Polygon(np.array([[0, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 0]]))
    assert poly.min_width() > 0.5


# ---------------------------------------------------------------- 擦边
def test_grazing_boundary_containment_and_distance():
    r = rect(1.0, 0.4)
    on_edge = np.array([[0.0, 0.2]])
    assert bool(r.contains(on_edge)[0])
    assert abs(float(r.distance(on_edge)[0])) < 1e-12

    eps = 1e-3
    outside = np.array([[0.0, 0.2 + eps]])
    assert not bool(r.contains(outside)[0])
    assert float(r.distance(outside)[0]) == pytest.approx(eps, rel=1e-6)
    # 膨胀 eps 后必须刚好包含
    assert bool(r.inflate(eps).contains(outside)[0])


def test_inflate_is_conservative_for_all_directions():
    """沿每一条边的外法向外移 delta 的点，必须被 F ⊕ disc(delta) 覆盖。"""
    r = rect(0.6, 0.3)
    delta = 0.05
    grown = r.inflate(delta)
    v = r.vertices
    nxt = np.roll(v, -1, axis=0)
    for a, b in zip(v, nxt):
        edge = b - a
        normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)   # CCW 的外法向
        for t in (0.1, 0.5, 0.9):
            boundary = a + edge * t
            q = (boundary + normal * delta)[None, :]
            assert r.distance(q)[0] == pytest.approx(delta, rel=1e-6)
            assert bool(grown.contains(q)[0]), f"边界点 {boundary} 外移 delta 未被覆盖"
    # 顶点外侧（45° 方向）同样要覆盖到
    for corner in v:
        q = (corner + (corner / np.linalg.norm(corner)) * delta)[None, :]
        assert bool(grown.contains(q)[0])


def test_disc_is_circumscribed():
    for n in (16, 64):
        d = disc(0.1, n)
        radii = np.linalg.norm(d, axis=1)
        assert radii.min() >= 0.1 - 1e-12          # 外切：半径只大不小


# ---------------------------------------------------------------- 连续扫掠
def test_swept_collision_catches_thin_obstacle():
    """细长障碍：只看起终点会漏，连续扫掠必须命中。"""
    body = rect(0.1, 0.1)
    p0, p1 = (0.0, 0.0), (1.0, 0.0)
    obstacle = (0.5, 0.04)
    assert not bool(body.translate(p0).contains(np.array([obstacle]))[0])
    assert not bool(body.translate(p1).contains(np.array([obstacle]))[0])
    assert swept_collision(body, p0, p1, obstacle)


def test_swept_collision_agrees_with_dense_sampling():
    body = rect(0.12, 0.08)
    p0, p1 = (0.0, 0.0), (0.6, 0.2)
    rng = np.random.default_rng(3)
    for obstacle in rng.uniform(-0.1, 0.75, size=(40, 2)):
        hit = swept_collision(body, p0, p1, obstacle)
        dense = False
        for s in range(0, 2001):
            u = s / 2000.0
            p = np.array(p0) + (np.array(p1) - np.array(p0)) * u
            if bool(body.translate(p).contains(obstacle[None, :])[0]):
                dense = True
                break
        assert hit == dense, f"obstacle {obstacle} 扫掠与密采样不一致"


def test_rotating_sweep_hull_converges():
    """采样凸包是连续扫掠的下界：加密采样只会变大，且增量迅速变小。"""
    body = rect(0.5, 0.2)
    poses = [((0.0, 0.0), 0.0), ((0.5, 0.2), math.radians(30))]
    areas = [rotating_sweep_hull(body, poses, substeps=k).area for k in (1, 4, 16)]
    assert areas[0] <= areas[1] <= areas[2]
    assert (areas[2] - areas[1]) < (areas[1] - areas[0])
    assert areas[2] - areas[1] < 1e-3


# ---------------------------------------------------------------- 资产与轮廓
@pytest.mark.skipif(not ASSET.is_file(), reason="缺少资产几何 JSON")
def test_asset_rest_consistency():
    geom = AssetGeometry.load(ASSET)
    out = geom.rest_consistency()
    assert out["max_abs_err"] < 1e-6
    assert len(geom.links) == 19
    assert len(geom.joints) == 18
    assert len(geom.colliders) == 27


@pytest.mark.skipif(not ASSET.is_file(), reason="缺少资产几何 JSON")
def test_fk_batch_matches_single_frame():
    geom = AssetGeometry.load(ASSET)
    rng = np.random.default_rng(7)
    Q = rng.normal(scale=0.2, size=(5, 12))
    batch = geom.fk_batch(Q, POLICY_JOINT_ORDER)
    for i in range(Q.shape[0]):
        single = geom.fk({n: Q[i, j] for j, n in enumerate(POLICY_JOINT_ORDER)})
        for link, T in single.items():
            assert np.allclose(batch[link][i], T, atol=1e-12)


@pytest.mark.skipif(not ASSET.is_file(), reason="缺少资产几何 JSON")
def test_nominal_outline_contains_body():
    geom = AssetGeometry.load(ASSET)
    q0 = init_joint_pose()[None, :]
    poly = outline_from_q(geom, q0, POLICY_JOINT_ORDER)
    x0, x1, y0, y1 = poly.bounds()
    # 机身碰撞盒是 0.3762×0.0935（半宽 0.0468），必须被包含
    assert x0 <= -0.188 + 1e-6 and x1 >= 0.188 - 1e-6
    assert y1 - y0 >= 0.0935 - 1e-6
    # 腿在名义站姿下已伸到机身之外
    assert (x1 - x0) > 0.3762


@pytest.mark.skipif(not ASSET.is_file(), reason="缺少资产几何 JSON")
def test_outline_covers_all_frames_of_its_own_input():
    geom = AssetGeometry.load(ASSET)
    rng = np.random.default_rng(11)
    Q = np.stack([init_joint_pose() for _ in range(3)]) + rng.normal(scale=0.05, size=(3, 12))
    poly = outline_from_q(geom, Q, POLICY_JOINT_ORDER)
    cov = coverage(poly, geom, Q, POLICY_JOINT_ORDER)
    assert cov["n_points_outside"] == 0


@pytest.mark.skipif(not ASSET.is_file(), reason="缺少资产几何 JSON")
def test_joint_ranges_shape():
    rng = np.random.default_rng(2)
    Q = rng.normal(size=(100, 12))
    rngs = joint_ranges(Q, POLICY_JOINT_ORDER)
    assert set(rngs) == set(POLICY_JOINT_ORDER)
    for v in rngs.values():
        assert v["min_rad"] <= v["max_rad"]


# ---------------------------------------------------------------- 轮廓对象
def make_footprint(delta_note: str = "") -> Footprint:
    return Footprint(polygon=rect(0.8, 0.5), version="test-v1", frame="base_link_xy",
                     method="unit-test", source={"splits_used": ["baseline_fit"]},
                     joint_ranges={}, stats={}, notes=(delta_note,))


def test_footprint_save_load_roundtrip(tmp_path):
    fp = make_footprint()
    p = tmp_path / "fp.json"
    digest = fp.save(p)
    back = Footprint.load(p)
    assert back.sha256() == digest
    assert np.allclose(back.polygon.vertices, fp.polygon.vertices)


def test_footprint_detects_tampering(tmp_path):
    fp = make_footprint()
    p = tmp_path / "fp.json"
    fp.save(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["polygon"]["vertices"][0][0] += 0.01
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FootprintError):
        Footprint.load(p)


def test_delta_applied_exactly_once():
    """危险集 = obstacle ⊕ footprint ⊕ disc(delta)；再叠一次会在数值上明显更大。"""
    fp = make_footprint()
    delta = 0.12
    obstacle = Polygon(np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]))
    once = fp.obstacle_dilation(obstacle, delta)
    twice = fp.obstacle_dilation(obstacle, 2 * delta)
    base = fp.obstacle_dilation(obstacle, 0.0)
    # 沿 +y 的支撑：一次 = 0.1(障碍) + 0.25(轮廓) + delta
    assert once.support((0, 1)) == pytest.approx(0.1 + 0.25 + delta, abs=2e-3)
    assert base.support((0, 1)) == pytest.approx(0.1 + 0.25, abs=2e-3)
    assert twice.support((0, 1)) - once.support((0, 1)) > delta * 0.9


def test_inflated_rejects_negative_delta():
    fp = make_footprint()
    with pytest.raises(FootprintError):
        fp.inflated(-0.01)


def test_minkowski_sum_of_unit_squares():
    a = rect(1.0, 1.0)
    b = rect(0.5, 0.5)
    s = minkowski_sum(a, b)
    assert s.area == pytest.approx(1.5 * 1.5, rel=1e-9)
    assert s.bounds()[1] == pytest.approx(0.75, rel=1e-9)
