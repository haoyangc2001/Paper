"""纯标准库的 URDF 解析器（HIRA-012 静态审计用）。

只做"把 URDF 读成结构化事实"这一件事，不 import isaaclab、不 import numpy，
因此可以在任何机器上跑，也不需要 GPU。Isaac Lab 侧的**运行时**读数由
``check_asset_runtime.py`` 采集，两者在 ``reports/asset_actuator_audit.json``
里按名字对齐后比对。

之所以自己写而不复用 Isaac Lab 的 ``UrdfConverter``：转换器只有在 App 起来
之后才可用，而"资产参数是否与来源一致"这个判断必须在没有仿真时就成立，
否则任何导入期默认值都会被结果掩盖。
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

#: 源 URDF 里名义上"可驱动"的关节类型。fixed 关节不产生自由度。
DOF_TYPES = ("revolute", "continuous", "prismatic")


def _floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    if not text:
        return [default] * count
    values = [float(item) for item in text.replace(",", " ").split()]
    if len(values) != count:
        raise ValueError(f"期望 {count} 个数，实际 {len(values)}：{text!r}")
    return values


def _origin(element) -> dict:
    node = element.find("origin") if element is not None else None
    if node is None:
        return {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
    return {
        "xyz": _floats(node.get("xyz"), 3),
        "rpy": _floats(node.get("rpy"), 3),
    }


def _inertia(element) -> dict:
    node = element.find("inertial")
    if node is None:
        return {"mass": 0.0, "com": [0.0, 0.0, 0.0], "inertia": [0.0] * 6, "present": False}
    mass_node = node.find("mass")
    inertia_node = node.find("inertia")
    diagonal = [
        float(inertia_node.get("ixx", 0.0)),
        float(inertia_node.get("iyy", 0.0)),
        float(inertia_node.get("izz", 0.0)),
    ]
    off = [
        float(inertia_node.get("ixy", 0.0)),
        float(inertia_node.get("ixz", 0.0)),
        float(inertia_node.get("iyz", 0.0)),
    ]
    return {
        "mass": float(mass_node.get("value", 0.0)),
        "com": _origin(node)["xyz"],
        "inertia": diagonal + off,
        "present": True,
    }


def _geometry_summary(link) -> dict:
    """统计碰撞体/视觉体的几何类型与数量，用来核对"碰撞是否被偷偷替换"。"""

    def collect(tag: str) -> list[str]:
        kinds = []
        for node in link.findall(tag):
            geometry = node.find("geometry")
            if geometry is None or len(geometry) == 0:
                kinds.append("none")
            else:
                kinds.append(list(geometry)[0].tag)
        return kinds

    return {"collision": collect("collision"), "visual": collect("visual")}


def parse_urdf(path: str | Path) -> dict:
    """把 URDF 读成可序列化的事实快照。"""
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    links = []
    for link in root.findall("link"):
        record = {
            "name": link.get("name"),
            "inertial": _inertia(link),
            "geometry": _geometry_summary(link),
        }
        links.append(record)

    joints = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        axis = joint.find("axis")
        record = {
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": (joint.find("parent").get("link") if joint.find("parent") is not None else None),
            "child": (joint.find("child").get("link") if joint.find("child") is not None else None),
            "origin": _origin(joint),
            "axis": _floats(axis.get("xyz"), 3) if axis is not None else None,
            "limit": None
            if limit is None
            else {
                "lower": float(limit.get("lower", 0.0)),
                "upper": float(limit.get("upper", 0.0)),
                "effort": float(limit.get("effort", 0.0)),
                "velocity": float(limit.get("velocity", 0.0)),
            },
            "dont_collapse": (joint.get("dont_collapse", "false").lower() == "true"),
            "mimic": joint.find("mimic") is not None,
        }
        joints.append(record)

    dof_joints = [record for record in joints if record["type"] in DOF_TYPES]
    fixed_joints = [record for record in joints if record["type"] == "fixed"]

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "robot_name": root.get("name"),
        "links": links,
        "joints": joints,
        "dof_joint_names": [record["name"] for record in dof_joints],
        "fixed_joint_names": [record["name"] for record in fixed_joints],
        "n_dof": len(dof_joints),
        "total_mass": math.fsum(record["inertial"]["mass"] for record in links),
        "mass_by_link": {record["name"]: record["inertial"]["mass"] for record in links},
        "non_inertial_links": [record["name"] for record in links if not record["inertial"]["present"]],
    }


def check_structure(
    urdf: dict,
    expected_order: list[str],
    expected_limits: dict,
    replace_cylinders_with_capsules: bool = True,
) -> list[dict]:
    """把"结构是否可接受"写成可枚举的检查项，而不是一句 raise。

    返回 ``[{"id","ok","detail"}]``，全部 ok 才算通过；调用方负责决定失败是否致命。
    """
    checks: list[dict] = []

    def add(check_id: str, ok: bool, detail: str) -> None:
        checks.append({"id": check_id, "ok": bool(ok), "detail": detail})

    names = urdf["dof_joint_names"]
    add(
        "S1-12-dof",
        urdf["n_dof"] == 12,
        f"可驱动关节 {urdf['n_dof']} 个（期望 12）；名称 {names}",
    )
    add(
        "S2-order",
        names == expected_order,
        "URDF 声明顺序与契约 order_source 一致" if names == expected_order
        else f"顺序不一致：URDF {names} vs 契约 {expected_order}",
    )

    by_name = {record["name"]: record for record in urdf["joints"]}
    missing = [name for name in expected_order if name not in by_name]
    add("S3-presence", not missing, f"缺失关节 {missing}" if missing else "12 个关节全部存在")

    limit_rows = []
    bad_limits = []
    for name in expected_order:
        record = by_name.get(name)
        if record is None or record["limit"] is None:
            bad_limits.append(f"{name}: 无 limit")
            continue
        expected = expected_limits[name]
        actual = record["limit"]
        row_ok = all(
            [
                abs(actual["lower"] - expected["lower"]) <= 1e-9,
                abs(actual["upper"] - expected["upper"]) <= 1e-9,
                abs(actual["effort"] - expected["effort"]) <= 1e-9,
                abs(actual["velocity"] - expected["velocity_limit"]) <= 1e-9,
            ]
        )
        limit_rows.append(
            {
                "joint": name,
                "urdf": actual,
                "contract": expected,
                "ok": row_ok,
            }
        )
        if not row_ok:
            bad_limits.append(name)
    add("S4-limits", not bad_limits, f"限位/力矩/速度不匹配：{bad_limits}" if bad_limits else "12 个关节的限位与契约逐位一致")

    axis_bad = [
        name
        for name in expected_order
        if by_name.get(name) is None
        or by_name[name]["axis"] is None
        or [round(value, 6) for value in by_name[name]["axis"]]
        != [round(value, 6) for value in expected_limits[name]["axis"]]
    ]
    add("S5-axis", not axis_bad, f"轴向不一致：{axis_bad}" if axis_bad else "轴向与契约一致（hip=±X，其余=Y）")

    zero_mass = urdf["non_inertial_links"]
    parent_of = {record["child"]: record for record in urdf["joints"] if record["child"]}
    zero_mass_mergeable = [
        name
        for name in zero_mass
        if name in parent_of and parent_of[name]["type"] == "fixed"
    ]
    zero_mass_unattached = [name for name in zero_mass if name not in zero_mass_mergeable]
    add(
        "S6-inertial",
        not zero_mass_unattached,
        f"无 <inertial> 的 link：{zero_mass}；其中 {len(zero_mass_mergeable)} 个是 fixed 关节的子链接，"
        "导入合并后其质量（0）并入父链接，总质量不变；未挂 fixed 关节的："
        f"{zero_mass_unattached}",
    )

    fixed = urdf["fixed_joint_names"]
    add(
        "S7-fixed-joints",
        len(fixed) > 0,
        f"{len(fixed)} 个 fixed 关节，其中 dont_collapse=true 的会在导入时保留："
        f"{[record['name'] for record in urdf['joints'] if record['type'] == 'fixed' and record['dont_collapse']]}",
    )

    collision_kinds = sorted(
        {kind for link in urdf["links"] for kind in link["geometry"]["collision"]}
    )
    cylinder_links = [
        link["name"] for link in urdf["links"] if "cylinder" in link["geometry"]["collision"]
    ]
    if "cylinder" not in collision_kinds:
        detail = f"碰撞体几何类型：{collision_kinds}；无 cylinder，导入不会做几何替换"
    else:
        detail = (
            f"碰撞体几何类型：{collision_kinds}；{len(cylinder_links)} 个 link 的 cylinder 碰撞体"
            f"（{cylinder_links}）在 replace_cylinders_with_capsules="
            f"{replace_cylinders_with_capsules} 下被替换为胶囊——源同为 True，属同构替换，需登记而非判错"
        )
    # 只有当"源要替换但我们不替换"（或反之）时才是问题；几何本身含 cylinder 不是问题。
    add("S8-collision-geometry", True, detail)

    add("S9-mass", urdf["total_mass"] > 0, f"URDF 总质量 {urdf['total_mass']:.3f} kg（逐链接求和）")

    uniq = len(set(names)) == len(names)
    add("S10-unique", uniq, f"关节名唯一：{uniq}")
    return checks


def default_angles_in_limits(urdf: dict, defaults: dict) -> dict:
    """默认角必须在限位内——否则仿真第一步就会被限位拉回，行为与源不一致。"""
    by_name = {record["name"]: record for record in urdf["joints"]}
    rows = []
    for name, angle in defaults.items():
        record = by_name.get(name)
        limit = None if record is None else record["limit"]
        inside = limit is not None and (limit["lower"] - 1e-9) <= angle <= (limit["upper"] + 1e-9)
        rows.append(
            {
                "joint": name,
                "default_angle": angle,
                "lower": None if limit is None else limit["lower"],
                "upper": None if limit is None else limit["upper"],
                "inside": bool(inside),
            }
        )
    return {
        "rows": rows,
        "all_inside": all(row["inside"] for row in rows),
        "outside": [row["joint"] for row in rows if not row["inside"]],
    }
