"""HIRA-027 第一步：从 HiRA 实际资产（URDF）抽取碰撞几何，生成足迹计算用的资产几何文件。

与 v2 的 `config/go2_asset_geometry_v1.json` 同 schema（`camra-go2-asset-geometry-v1`），
因此可以直接被 `camra_calibration.footprint.AssetGeometry.load(..., verify_hash=True)` 读取。
**不照抄 v2 的数值**：每一条碰撞体、每一个关节都是从本候选包内的 `urdf_go2.urdf` 重新解析的。

URDF → 本 schema 的两处约定：
* 关节：URDF 的 ``origin`` 就是 child 相对 parent 的零位姿，因此 ``pos0/rot0 = origin``、
  ``pos1 = 0``、``rot1 = [1,0,0,0]``（子链接与关节系重合）。
* 长方体：URDF 是三边不等，而采样器只支持立方体 + 4×4 变换，所以把单位立方体
  （size=1）用 ``matrix_link_from_shape`` 的对角缩放成 (sx,sy,sz) —— 采样点在 link 系里仍精确。

    python tests/extract_geometry_hira.py --urdf <urdf> --out config/go2_asset_geometry_hira_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent  # noqa: E402

SCHEMA = "camra-go2-asset-geometry-v1"


def rpy_to_mat(rpy) -> np.ndarray:
    """URDF 的 rpy（固定轴，等价 R = Rz·Ry·Rx）。"""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def mat_to_quat_wxyz(R: np.ndarray) -> list[float]:
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return (q / np.linalg.norm(q)).tolist()


def homog(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def parse_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            raise SystemExit("link 缺 name")
        colliders = []
        for col in link.findall("collision"):
            origin = col.find("origin")
            xyz = [0.0, 0.0, 0.0] if origin is None else [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [0.0, 0.0, 0.0] if origin is None else [float(v) for v in origin.get("rpy", "0 0 0").split()]
            geo = col.find("geometry")
            if geo is None or len(list(geo)) == 0:
                raise SystemExit(f"{name}: collision 缺 geometry")
            shape = list(geo)[0]
            tag = shape.tag
            if tag == "box":
                size = [float(v) for v in shape.get("size").split()]
                colliders.append({"kind": "cube", "size": 1.0, "scale": size,
                                  "matrix": homog(rpy_to_mat(rpy), xyz)})
            elif tag == "cylinder":
                colliders.append({"kind": "cylinder", "radius": float(shape.get("radius")),
                                  "height": float(shape.get("length")), "axis": "Z",
                                  "matrix": homog(rpy_to_mat(rpy), xyz)})
            elif tag == "sphere":
                colliders.append({"kind": "sphere", "radius": float(shape.get("radius")),
                                  "matrix": homog(rpy_to_mat(rpy), xyz)})
            else:
                raise SystemExit(f"{name}: 不支持的碰撞体 {tag}（本资产应只有 box/cylinder/sphere）")
        links[name] = {"colliders": colliders}
    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0] if origin is None else [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [0.0, 0.0, 0.0] if origin is None else [float(v) for v in origin.get("rpy", "0 0 0").split()]
        axis_el = joint.find("axis")
        axis = [1.0, 0.0, 0.0] if axis_el is None else [float(v) for v in axis_el.get("xyz", "1 0 0").split()]
        limit = joint.find("limit")
        joints.append(
            {
                "name": joint.get("name"),
                "kind": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "lower": None if limit is None else limit.get("lower"),
                "upper": None if limit is None else limit.get("upper"),
            }
        )
    return {"links": links, "joints": joints}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=str(EXP / "frozen/hira_go2_18500/urdf_go2.urdf"))
    parser.add_argument("--out", default=str(EXP / "config/go2_asset_geometry_hira_v1.json"))
    parser.add_argument("--drop-links", default="", help="逗号分隔：不参与足迹的 link（默认保留全部有碰撞的）")
    parser.add_argument("--require-collision-links", default="", help="逗号分隔：必须带碰撞体的 link")
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    parsed = parse_urdf(urdf_path)
    log: list[str] = []

    # ---- link 的零位位姿：从根累乘 origin（关节角全 0）----
    parent_of = {j["child"]: j for j in parsed["joints"]}
    children = {j["child"] for j in parsed["joints"]}
    roots = [name for name in parsed["links"] if name not in children]
    if len(roots) != 1:
        raise SystemExit(f"URDF 应有唯一根链接，得到 {roots}")
    rest = {roots[0]: np.eye(4)}
    order = [roots[0]]
    queue = [roots[0]]
    while queue:
        cur = queue.pop(0)
        for j in parsed["joints"]:
            if j["parent"] != cur:
                continue
            T = rest[cur] @ homog(rpy_to_mat(j["rpy"]), j["xyz"])
            if j["child"] in rest:
                raise SystemExit(f"链接 {j['child']} 被重复定义")
            rest[j["child"]] = T
            order.append(j["child"])
            queue.append(j["child"])
    if len(rest) != len(parsed["links"]):
        missing = set(parsed["links"]) - set(rest)
        raise SystemExit(f"有链接不在关节树上：{missing}")

    links_payload = {
        name: {"rest_xyz": [round(float(v), 12) for v in rest[name][:3, 3]]}
        for name in parsed["links"]
    }

    joints_payload = []
    for j in parsed["joints"]:
        if j["kind"] not in ("revolute", "continuous", "fixed"):
            raise SystemExit(f"{j['name']}: 未处理关节类型 {j['kind']}")
        joints_payload.append(
            {
                "name": j["name"],
                "kind": "revolute" if j["kind"] in ("revolute", "continuous") else "fixed",
                "parent_link": j["parent"],
                "child_link": j["child"],
                "pos0": [float(v) for v in j["xyz"]],
                "rot0": mat_to_quat_wxyz(rpy_to_mat(j["rpy"])),
                "pos1": [0.0, 0.0, 0.0],
                "rot1": [1.0, 0.0, 0.0, 0.0],
                "axis": "XYZ"[int(np.argmax(np.abs(j["axis"])))],
                "lower_rad": float(j["lower"]) if j["lower"] not in (None, "") else 0.0,
                "upper_rad": float(j["upper"]) if j["upper"] not in (None, "") else 0.0,
            }
        )

    drop = {s for s in args.drop_links.split(",") if s}
    colliders_payload = []
    for name, link in parsed["links"].items():
        if name in drop:
            log.append(f"按 --drop-links 排除 {name} 的 {len(link['colliders'])} 个碰撞体")
            continue
        for c in link["colliders"]:
            M = c["matrix"]
            if c["kind"] == "cube":
                sx, sy, sz = c["scale"]
                M = M @ np.diag([sx, sy, sz, 1.0])
                entry = {"link": name, "kind": "cube", "size": 1.0}
            elif c["kind"] == "cylinder":
                entry = {"link": name, "kind": "cylinder", "radius": c["radius"],
                         "height": c["height"], "axis": c["axis"]}
            else:
                entry = {"link": name, "kind": "sphere", "radius": c["radius"]}
            entry["matrix_link_from_shape"] = [float(v) for v in M.reshape(-1)]
            colliders_payload.append(entry)

    required = {s for s in args.require_collision_links.split(",") if s}
    have = {c["link"] for c in colliders_payload}
    missing_required = required - have
    if missing_required:
        raise SystemExit(f"这些 link 必须带碰撞体但没有：{missing_required}")

    payload = {
        "schema": SCHEMA,
        "source": {
            "asset": "hira go2（HiRA 仓库 legged_gym_go2/resources/robots/go2/urdf/go2.urdf）",
            "urdf_file": urdf_path.name,
            "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
            "extractor": "tests/extract_geometry_hira.py（本候选自建，未复用 v2 的 USD 提取结果）",
            "n_links": len(links_payload),
            "n_joints": len(joints_payload),
            "n_colliders": len(colliders_payload),
            "collider_kinds": sorted({c["kind"] for c in colliders_payload}),
            "notes": log,
            "conventions": {
                "joint": "pos0/rot0 = URDF origin；pos1=0、rot1=identity（子链接与关节系重合）",
                "box": "URDF 三边不等 → 单位立方体 + matrix 对角缩放",
                "frame": "机身系（base 链接原点，x 前 y 左 z 上），单位 m",
            },
        },
        "links": links_payload,
        "joints": joints_payload,
        "colliders": colliders_payload,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(f"[geom] links={len(links_payload)} joints={len(joints_payload)} colliders={len(colliders_payload)}")
    print(f"[geom] kinds={payload['source']['collider_kinds']} sha={digest[:16]}")
    print(f"[geom] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
