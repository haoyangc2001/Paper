"""Scene mapping checks for NAV-022.

Every physical object must come from the map spec with real values: no inherited
0.3 m box, no fixed slot count, and a boundary that exists as geometry, not only
in the cost function.

Run:

    <frozen python> CaMRA/navigation/tests/test_scene_mapping.py \
        --json CaMRA/navigation/reports/scene_mapping_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps import scene_adapter as S  # noqa: E402
from camra_navigation.maps.generator import MapSpec  # noqa: E402

DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
NUM_ENVS = 32

RESULTS: list[dict[str, Any]] = []
SEED = 1001


def check(name: str, ok: bool, detail: Any) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})


def odd_map() -> MapSpec:
    """A map with sizes and a rotation that no legacy default could produce."""
    return MapSpec(
        map_id="synthetic-odd",
        level="synthetic",
        seed=SEED,
        extent=(-7.0, 9.0, -4.0, 6.0),
        start=(-4.0, 1.0),
        start_yaw=0.37,
        goal=(6.0, -1.0),
        islands=(
            (1.0, 1.0, 1.37, 0.83, 0.0),
            (-2.0, 3.0, 0.91, 2.44, 0.6),
        ),
        design_version="synthetic",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    design = M.load_design(DESIGN)
    spec_real = M.build("L2_single_detour", SEED, design)
    spec_odd = odd_map()
    spacing = S.required_env_spacing(spec_real)

    # --- 1. island fields map exactly, no defaults ------------------------
    scene = S.build_scene(spec_odd, 0, NUM_ENVS)
    mismatches = []
    for i, (ix, iy, w, d, yaw) in enumerate(spec_odd.islands):
        b = scene.island_boxes[i]
        lx = ix - scene.map_center[0] + scene.env_origin[0]
        ly = iy - scene.map_center[1] + scene.env_origin[1]
        if (
            b.name != f"island_{i}"
            or abs(b.center_x - lx) > 1e-12
            or abs(b.center_y - ly) > 1e-12
            or b.size_x != w
            or b.size_y != d
            or b.yaw != yaw
            or b.height != S.OBSTACLE_HEIGHT_M
        ):
            mismatches.append({"i": i, "spec": [ix, iy, w, d, yaw], "box": b.as_dict()})
    check(
        "island_fields_map_exactly",
        not mismatches and len(scene.island_boxes) == len(spec_odd.islands),
        {
            "mismatches": mismatches,
            "n_islands": len(spec_odd.islands),
            "note": "尺寸、旋转、中心、高度全部取自 spec；局部坐标 = 世界坐标 - 地图中心",
        },
    )

    # --- 2. no legacy defaults --------------------------------------------
    sizes = {(1.37, 0.83), (0.91, 2.44)}
    got = {(b.size_x, b.size_y) for b in scene.island_boxes}
    constants = scene.as_dict()["constants"]
    check(
        "no_legacy_defaults",
        got == sizes
        and constants["obstacle_height_m"] == 0.5
        and constants["wall_thickness_m"] == 1.0
        and all(abs(s - 0.3) > 1e-9 for pair in got for s in pair),
        {
            "box_sizes": sorted(got),
            "constants": constants,
            "note": "0.3 m 的旧默认箱体与固定槽位不进入本适配器；高度 0.5 m 高于 Go2 身高 0.35 m",
        },
    )

    # --- 3. walls are real geometry ---------------------------------------
    ext_scene = S.extent_from_walls(scene.wall_boxes)
    ext = tuple(
        v - o + c for v, o, c in zip(ext_scene, scene.env_origin, scene.map_center)
    )
    check(
        "walls_are_physical_and_match_extent",
        len(scene.wall_boxes) == 4
        and all(abs(a - b) < 1e-12 for a, b in zip(ext, spec_odd.extent))
        and all(
            w.size_x == S.WALL_THICKNESS_M or w.size_y == S.WALL_THICKNESS_M
            for w in scene.wall_boxes
        )
        and all(w.height == S.OBSTACLE_HEIGHT_M for w in scene.wall_boxes),
        {
            "extent_scene_local": list(ext_scene),
            "extent_map_local": list(ext),
            "spec_extent": list(spec_odd.extent),
            "env_origin": list(scene.env_origin),
            "wall_sizes": [[w.size_x, w.size_y] for w in scene.wall_boxes],
            "note": "墙体内侧面与地图边界重合，厚 1 m 向外、高 0.5 m；边界是几何体而非只存在于成本函数",
        },
    )

    # --- 4. reverse round trip --------------------------------------------
    rebuilt = S.round_trip_spec(scene, "synthetic")
    original_islands = sorted(tuple(round(v, 12) for v in isl) for isl in spec_odd.islands)
    rebuilt_islands = sorted(
        tuple(round(v, 12) for v in isl) for isl in rebuilt.islands
    )
    check(
        "reverse_round_trip_recovers_islands",
        original_islands == rebuilt_islands,
        {
            "original": [list(i) for i in original_islands],
            "rebuilt": [list(i) for i in rebuilt_islands],
            "note": "从场景盒子反推岛中心/尺寸/旋转并平移回世界系，与原 spec 完全一致",
        },
    )

    # --- 5. rotation preserved --------------------------------------------
    b = scene.island_boxes[1]
    expected_q = (math.cos(0.6 / 2.0), 0.0, 0.0, math.sin(0.6 / 2.0))
    check(
        "rotation_preserved",
        abs(b.yaw - 0.6) < 1e-15
        and all(abs(a - c) < 1e-12 for a, c in zip(b.quat_wxyz, expected_q)),
        {
            "yaw_rad": b.yaw,
            "quat_wxyz": list(b.quat_wxyz),
            "expected_wxyz": list(expected_q),
            "aabb": list(b.aabb()),
            "note": "Isaac Lab 用 wxyz 四元数；yaw=0.6 rad 的箱子按原值输出",
        },
    )

    # --- 6. parallel environments cannot touch -----------------------------
    x0, x1, y0, y1 = spec_real.extent
    fx = (x1 - x0) + 2.0 * S.WALL_THICKNESS_M
    fy = (y1 - y0) + 2.0 * S.WALL_THICKNESS_M
    footprints = []
    for i in range(NUM_ENVS):
        ox, oy = S.env_origin(i, NUM_ENVS, spacing)
        footprints.append((ox - fx / 2, ox + fx / 2, oy - fy / 2, oy + fy / 2))
    touching = [
        [i, j]
        for i in range(NUM_ENVS)
        for j in range(i + 1, NUM_ENVS)
        if S.aabb_overlap(footprints[i], footprints[j])
    ]
    cols, rows = S.env_grid(NUM_ENVS)
    check(
        "parallel_envs_cannot_touch",
        not touching and spacing >= max(fx, fy) + S.ENV_SPACING_MARGIN_M
        and cols * rows >= NUM_ENVS,
        {
            "num_envs": NUM_ENVS,
            "grid": [cols, rows],
            "spacing_m": spacing,
            "footprint_xy_m": [fx, fy],
            "margin_m": S.ENV_SPACING_MARGIN_M,
            "overlapping_pairs": touching[:5],
            "note": "足迹 = 地图范围 + 两侧 1 m 墙体；间距 = 足迹长边 + 1 m 余量",
        },
    )

    # --- 7. parking is outside every environment ---------------------------
    scene32 = S.build_scene(spec_real, 0, NUM_ENVS)
    parked_boxes = []
    for px, py in scene32.parking_positions:
        parked_boxes.append((px - 1.0, px + 1.0, py - 1.0, py + 1.0))
    clashes = [
        i
        for i, pb in enumerate(parked_boxes)
        for fp in footprints
        if S.aabb_overlap(pb, fp)
    ]
    park_y = scene32.parking_positions[0][1]
    check(
        "parking_outside_all_envs",
        not clashes and park_y > max(fp[3] for fp in footprints),
        {
            "n_parked": len(scene32.parking_positions),
            "park_row_y_m": park_y,
            "highest_env_footprint_y": max(fp[3] for fp in footprints),
            "clashes": clashes[:5],
            "note": "停放的障碍体放在整片网格之外，不会与任何环境接触",
        },
    )

    # --- 8. nominal and physical boundaries are the same -------------------
    diffs = []
    for level in ("L1_empty", "L2_single_detour", "L3_multi_path"):
        sp = M.build(level, SEED, design)
        sc = S.build_scene(sp, 0, NUM_ENVS)
        got = tuple(
            v - o + c
            for v, o, c in zip(S.extent_from_walls(sc.wall_boxes), sc.env_origin, sc.map_center)
        )
        if any(abs(a - b) > 1e-12 for a, b in zip(got, sp.extent)):
            diffs.append({"level": level, "scene": list(got), "spec": list(sp.extent)})
    check(
        "nominal_physical_boundary_identical",
        not diffs,
        {
            "levels_checked": 3,
            "differences": diffs,
            "note": "场景墙体围出的范围与名义环境的 extent 逐值一致，不做任何收缩",
        },
    )

    # --- 9. spawn is inside the free area ----------------------------------
    sp = M.build("L2_single_detour", SEED, design)
    sc = S.build_scene(sp, 0, NUM_ENVS)
    spawn_map = (
        sc.robot_spawn_xy[0] - sc.env_origin[0] + sc.map_center[0],
        sc.robot_spawn_xy[1] - sc.env_origin[1] + sc.map_center[1],
    )
    x0, x1, y0, y1 = sp.extent
    inside = x0 < spawn_map[0] < x1 and y0 < spawn_map[1] < y1
    spawn_yaw_matches = abs(sc.robot_spawn_yaw - sp.start_yaw) < 1e-15
    check(
        "spawn_inside_free_area",
        inside and spawn_yaw_matches,
        {
            "spawn_xy_scene_local": list(sc.robot_spawn_xy),
            "spawn_xy_world": list(spawn_map),
            "spawn_yaw_rad": sc.robot_spawn_yaw,
            "spec_start_yaw": sp.start_yaw,
            "note": "初始位姿随 env 平移，航向按原值输出",
        },
    )

    # --- 10. deterministic build -------------------------------------------
    a = S.build_scene(spec_real, 5, NUM_ENVS).as_dict()
    b2 = S.build_scene(spec_real, 5, NUM_ENVS).as_dict()
    check(
        "scene_build_deterministic",
        json.dumps(a, sort_keys=True) == json.dumps(b2, sort_keys=True),
        {"env_index": 5, "bytes": len(json.dumps(a))},
    )

    # --- 11. grid tiling ----------------------------------------------------
    grid_ok = all(
        S.env_grid(n) == (math.ceil(math.sqrt(n)), math.ceil(n / math.ceil(math.sqrt(n))))
        for n in (1, 2, 5, 32, 64)
    )
    origins = [S.env_origin(i, NUM_ENVS, spacing) for i in range(NUM_ENVS)]
    distinct = len({(round(o[0], 9), round(o[1], 9)) for o in origins}) == NUM_ENVS
    check(
        "grid_tiling_deterministic",
        grid_ok and distinct,
        {
            "grid_32": list(S.env_grid(NUM_ENVS)),
            "distinct_origins": distinct,
            "first_origin": list(origins[0]),
            "last_origin": list(origins[-1]),
        },
    )

    verdict = all(r["ok"] for r in RESULTS)
    report = {
        "task": "NAV-022",
        "verdict": "PASS" if verdict else "FAIL",
        "n_checks": len(RESULTS),
        "n_failed": sum(1 for r in RESULTS if not r["ok"]),
        "constants": {
            "wall_thickness_m": S.WALL_THICKNESS_M,
            "obstacle_height_m": S.OBSTACLE_HEIGHT_M,
            "env_spacing_margin_m": S.ENV_SPACING_MARGIN_M,
        },
        "checks": RESULTS,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
    for r in RESULTS:
        print(("  OK   " if r["ok"] else "  FAIL ") + r["check"])
        if not r["ok"]:
            print("        " + json.dumps(r["detail"], ensure_ascii=False)[:400])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
