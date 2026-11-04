"""从冻结 Go2 USD 资产提取几何与运动学（CAL-021 第一步）。

只读取、不修改任何 USD；输出一份可复算的 JSON：链接、关节链、碰撞体（形状参数 +
链接系变换）。**不启动 Isaac 应用**，直接用 omni.usd.libs 里的 USD 核心库
（见 env_setup/usdpy.sh 的 PYTHONPATH/LD_LIBRARY_PATH 设置）。

用法：
    bash usdpy.sh scripts/extract_asset_geometry.py \
        --usd <CAMRA_ASSET_ROOT>/Go2/usd/go2.usd --out config/go2_asset_geometry_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quat_wxyz(q) -> list[float]:
    """Gf.Quatf 的实部在前，输出 [w, x, y, z]。"""
    return [float(q.GetReal()), float(q.GetImaginary()[0]),
            float(q.GetImaginary()[1]), float(q.GetImaginary()[2])]


def mat_row_major(m: Gf.Matrix4d) -> list[float]:
    flat: list[float] = []
    for r in range(4):
        flat.extend(float(x) for x in m.GetRow(r))
    return flat


def shape_params(prim) -> dict:
    t = prim.GetTypeName()
    if t == "Cube":
        c = UsdGeom.Cube(prim)
        return {"kind": "cube", "size": float(c.GetSizeAttr().Get())}
    if t == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return {"kind": "cylinder", "radius": float(c.GetRadiusAttr().Get()),
                "height": float(c.GetHeightAttr().Get()),
                "axis": str(c.GetAxisAttr().Get())}
    if t in ("Capsule",):
        c = UsdGeom.Capsule(prim)
        return {"kind": "capsule", "radius": float(c.GetRadiusAttr().Get()),
                "height": float(c.GetHeightAttr().Get()),
                "axis": str(c.GetAxisAttr().Get())}
    if t == "Sphere":
        s = UsdGeom.Sphere(prim)
        return {"kind": "sphere", "radius": float(s.GetRadiusAttr().Get())}
    raise ValueError(f"未支持的碰撞体类型 {t} @ {prim.GetPath()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    usd_path = Path(args.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()

    # ---- 关节链 ----
    joints: list[dict] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        b0 = [str(x) for x in prim.GetRelationship("physics:body0").GetTargets()]
        b1 = [str(x) for x in prim.GetRelationship("physics:body1").GetTargets()]
        if not b0 or not b1:
            continue
        kind = "fixed" if prim.GetTypeName() == "PhysicsFixedJoint" else "revolute"
        node = {
            "name": prim.GetName(),
            "kind": kind,
            "parent_link": b0[0].rsplit("/", 1)[-1],
            "child_link": b1[0].rsplit("/", 1)[-1],
            "pos0": [float(v) for v in UsdPhysics.Joint(prim).GetLocalPos0Attr().Get()],
            "rot0": quat_wxyz(UsdPhysics.Joint(prim).GetLocalRot0Attr().Get()),
            "pos1": [float(v) for v in UsdPhysics.Joint(prim).GetLocalPos1Attr().Get()],
            "rot1": quat_wxyz(UsdPhysics.Joint(prim).GetLocalRot1Attr().Get()),
        }
        if kind == "revolute":
            rj = UsdPhysics.RevoluteJoint(prim)
            node["axis"] = str(rj.GetAxisAttr().Get())
            node["lower_rad"] = float(Gf.DegreesToRadians(rj.GetLowerLimitAttr().Get()))
            node["upper_rad"] = float(Gf.DegreesToRadians(rj.GetUpperLimitAttr().Get()))
        joints.append(node)
    joints.sort(key=lambda j: j["name"])

    # ---- 链接静止位姿（用于自洽校验：FK(q=0) 必须复现）----
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    links: dict[str, dict] = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            links[prim.GetName()] = {"rest_xyz": [float(t[0]), float(t[1]), float(t[2])]}
    for j in joints:
        for k in ("parent_link", "child_link"):
            if j[k] not in links:
                raise SystemExit(f"关节 {j['name']} 引用了未知链接 {j[k]}")

    # ---- 碰撞体：来自 physics 层的 /colliders/<link>/... ----
    phys_path = usd_path.parent / "configuration" / "go2_description_physics.usd"
    phys = Usd.Stage.Open(str(phys_path))
    phys_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    colliders: list[dict] = []
    skipped: list[str] = []
    for link_scope in phys.GetPrimAtPath("/colliders").GetChildren():
        link = link_scope.GetName()
        if link not in links:
            # /colliders 下还有碰撞分组等非链接节点（如 robotCollisionGroup），跳过并记录
            skipped.append(str(link_scope.GetPath()))
            continue
        x_link = phys_cache.GetLocalToWorldTransform(link_scope)
        x_link_inv = x_link.GetInverse()
        for prim in Usd.PrimRange(link_scope):
            if prim.GetTypeName() not in ("Cube", "Cylinder", "Sphere", "Capsule"):
                continue
            m = x_link_inv * phys_cache.GetLocalToWorldTransform(prim)
            params = shape_params(prim)
            params.update({"link": link, "prim_path": str(prim.GetPath()),
                           "matrix_link_from_shape": mat_row_major(m)})
            colliders.append(params)
    colliders.sort(key=lambda c: c["prim_path"])

    payload = {
        "schema": "camra-go2-asset-geometry-v1",
        "source": {
            "usd": str(usd_path),
            "usd_sha256": sha256_file(usd_path),
            "physics_usd": str(phys_path),
            "physics_usd_sha256": sha256_file(phys_path),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "default_prim": str(root.GetPath()),
        },
        "links": links,
        "joints": joints,
        "colliders": colliders,
        "skipped_non_link_scopes": skipped,
        "notes": [
            "只读提取：未修改任何 USD 内容，也未改动碰撞几何。",
            "碰撞体来自 configuration/go2_description_physics.usd 的 /colliders/<link>；"
            "matrix_link_from_shape 是相对该链接坐标系的 4x4（行主序）。",
            "go2.usd 里的 payload 默认不加载，故碰撞体直接从 physics 层读取。",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    digest = sha256_file(out)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")

    n_by_kind: dict[str, int] = {}
    for c in colliders:
        n_by_kind[c["kind"]] = n_by_kind.get(c["kind"], 0) + 1
    print(f"links {len(links)}  joints {len(joints)}  colliders {len(colliders)} {n_by_kind}")
    print(f"wrote {out}  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
