"""CAL-023 数据锁与近障几何测试（纯 CPU）。

覆盖：锁加载/散列、采集门禁（未知 split、margin 门禁、地图绑定、正式禁止覆盖）、
以及近障距离与设计器的基本性质。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from camra_calibration.geometry import Polygon
from camra_calibration.lock import DataLock, LockError, enforce_collection, sha256_file
from camra_calibration.obstacles import (
    CLEARANCE_TIERS, Obstacle, design_case, distances_to_obstacle, footprint_vertices_batch,
    min_path_clearance, nominal_path, polygon_distance,
)


# ---------------------------------------------------------------- 工具
def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload,
                    encoding="utf-8")
    return path


def make_lock(tmp: Path, splits=("calibration", "heldout", "near_obstacle")) -> Path:
    files = {}
    for split in splits:
        p = tmp / "lists" / f"{split}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"trial_id": "%s-0000", "seed": 1}\n' % split, encoding="utf-8")
        files[split] = {"path": str(p), "sha256": sha256_file(p), "count": 1,
                        "seed_block": [6000, 800], "quota": 800}
    proto = tmp / "config" / "proto.json"
    write(proto, {"version": "1.0.2"})
    maps = tmp / "config" / "maps.json"
    write(maps, {"cases": []})
    payload = {
        "schema": "camra-data-lock-v1",
        "protocol": {"path": str(proto), "sha256": sha256_file(proto)},
        "near_obstacle_maps": {"path": str(maps), "sha256": sha256_file(maps)},
        "trial_lists": {"files": files},
    }
    lock_p = tmp / "config" / "data_lock_v1.json"
    write(lock_p, payload)
    lock_p.with_suffix(".json.sha256").write_text(
        f"{sha256_file(lock_p)}  {lock_p.name}\n", encoding="utf-8")
    return lock_p


# ---------------------------------------------------------------- 加载
def test_lock_load_and_sidecar(tmp_path):
    p = make_lock(tmp_path)
    lock = DataLock.load(p)
    assert lock.sha256 == sha256_file(p)
    assert lock.list_entry("calibration") is not None
    assert lock.list_entry("nope") is None


def test_lock_rejects_missing_or_tampered(tmp_path):
    p = make_lock(tmp_path)
    p.write_text(p.read_text(encoding="utf-8").replace('"schema"', '"schemaX"'),
                 encoding="utf-8")
    with pytest.raises(LockError):
        DataLock.load(p)
    with pytest.raises(LockError):
        DataLock.load(tmp_path / "nope.json")


def test_lock_rejects_unknown_schema(tmp_path):
    p = make_lock(tmp_path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload["schema"] = "other"
    write(p, payload)
    p.with_suffix(".json.sha256").write_text(f"{sha256_file(p)}  {p.name}\n", encoding="utf-8")
    with pytest.raises(LockError):
        DataLock.load(p)


# ---------------------------------------------------------------- 门禁
def test_enforce_allows_known_split(tmp_path):
    p = make_lock(tmp_path)
    lock = DataLock.load(p)
    enforce_collection(lock, "calibration", tmp_path,
                       manifest_dir=tmp_path / "lists")


def test_enforce_rejects_unknown_split(tmp_path):
    lock = DataLock.load(make_lock(tmp_path))
    with pytest.raises(LockError):
        enforce_collection(lock, "not_a_split", tmp_path)


def test_enforce_rejects_tampered_artifact(tmp_path):
    p = make_lock(tmp_path)
    lock = DataLock.load(p)
    (tmp_path / "lists" / "calibration.jsonl").write_text('{"x":1}\n', encoding="utf-8")
    with pytest.raises(LockError):
        enforce_collection(lock, "calibration", tmp_path, manifest_dir=tmp_path / "lists")


def test_margin_gate_for_heldout(tmp_path):
    lock = DataLock.load(make_lock(tmp_path))
    margins = tmp_path / "config" / "margins_v1.json"
    with pytest.raises(LockError):
        enforce_collection(lock, "heldout", tmp_path, margins_path=margins)
    write(margins, {"alpha": 0.1, "delta": 0.2})
    enforce_collection(lock, "heldout", tmp_path, margins_path=margins)


def test_map_gate_for_near_obstacle(tmp_path):
    lock = DataLock.load(make_lock(tmp_path))
    margins = tmp_path / "config" / "margins_v1.json"
    write(margins, {"alpha": 0.1})
    with pytest.raises(LockError):
        enforce_collection(lock, "near_obstacle", tmp_path, margins_path=margins)
    other = write(tmp_path / "config" / "other_maps.json", {"cases": [{"x": 0}]})
    with pytest.raises(LockError):          # 散列不符
        enforce_collection(lock, "near_obstacle", tmp_path, margins_path=margins,
                           maps_path=other)
    enforce_collection(lock, "near_obstacle", tmp_path, margins_path=margins,
                       maps_path=Path(lock.payload["near_obstacle_maps"]["path"]))


def test_formal_splits_refuse_overwrite(tmp_path):
    lock = DataLock.load(make_lock(tmp_path))
    margins = tmp_path / "config" / "margins_v1.json"
    write(margins, {"alpha": 0.1})
    for split in ("calibration", "heldout", "near_obstacle"):
        with pytest.raises(LockError):
            enforce_collection(lock, split, tmp_path, margins_path=margins,
                               maps_path=Path(lock.payload["near_obstacle_maps"]["path"]),
                               allow_overwrite=True)


def test_manifest_hash_checked(tmp_path):
    lock = DataLock.load(make_lock(tmp_path))
    forged = write(tmp_path / "other" / "calibration.jsonl", '{"a":1}\n')
    with pytest.raises(LockError):
        enforce_collection(lock, "calibration", tmp_path,
                           manifest_dir=forged.parent)


# ---------------------------------------------------------------- 近障几何
def test_distance_matches_bruteforce():
    fp = Polygon(np.array([[0.4, 0.2], [-0.4, 0.2], [-0.4, -0.2], [0.4, -0.2]]))
    obs = Obstacle(x=1.2, y=0.0, width=0.3, depth=0.3, yaw=0.2)
    fv = np.stack([fp.vertices])
    d = float(distances_to_obstacle(fv, obs.polygon())[0])
    # 障碍绕 z 转 0.2 rad 后其最左点 x = 1.2 − 0.15(|cos|+|sin|)，减去轮廓右边界 0.4
    expect = 1.2 - 0.15 * (abs(np.cos(0.2)) + abs(np.sin(0.2))) - 0.4
    assert d == pytest.approx(expect, rel=1e-6)
    assert d == pytest.approx(polygon_distance(fp, obs.polygon()), rel=1e-9)


def test_distance_zero_when_overlapping():
    fp = Polygon(np.array([[0.4, 0.2], [-0.4, 0.2], [-0.4, -0.2], [0.4, -0.2]]))
    obs = Obstacle(x=0.0, y=0.0, width=0.3, depth=0.3, yaw=0.0)
    assert float(distances_to_obstacle(np.stack([fp.vertices]), obs.polygon())[0]) == 0.0


def test_batch_vertices_match_single():
    fp = Polygon(np.array([[0.3, 0.1], [-0.3, 0.1], [-0.3, -0.1], [0.3, -0.1]]))
    pos = np.array([[0.0, 0.0], [1.0, 2.0]])
    yaw = np.array([0.0, 0.5])
    batch = footprint_vertices_batch(fp, pos, yaw)
    single = fp.rotate(0.5).translate((1.0, 2.0)).vertices
    # Polygon 构造会重新取凸包，顶点顺序可能不同，比较排序后的坐标
    assert np.allclose(np.sort(batch[1], axis=0), np.sort(single, axis=0), atol=1e-12)


def test_design_case_meets_target_and_stays_in_arena():
    """短而直的命令序列：净距必须真正达到目标，且路径/障碍都在场地内。"""
    cmd = np.zeros((61, 3))
    cmd[:, 0] = np.linspace(0.0, 1.0, 61)
    fp = Polygon(np.array([[0.45, 0.29], [-0.49, 0.28], [-0.49, -0.31], [0.45, -0.31]]))
    for tier in (0.05, 0.2):
        design = design_case(fp, cmd, tier, np.random.default_rng(0))
        assert design["clearance_achieved_m"] >= tier - 2e-3
        assert design["path_in_arena"] and design["obstacles_in_arena"]
        assert design["n_obstacles"] >= 4
        assert design["arena"]["width_m"] >= 8.0 and design["arena"]["height_m"] >= 8.0


def test_nominal_path_is_circular_arc():
    cmd = np.zeros((11, 3))
    cmd[:, 2] = 0.5                     # 恒定角速度 → 圆弧
    cmd[:, 0] = 0.5
    pos, yaw = nominal_path(cmd, (0.0, 0.0), 0.0)
    assert pos.shape == (11, 2)
    assert abs(yaw[-1] - 0.5 * 1.0) < 1e-9        # 10 段 × 0.1 s
    r = np.linalg.norm(pos - np.array([0.0, 1.0]), axis=1)
    assert np.allclose(r, 1.0, atol=1e-6)          # 半径 1 m 的圆
