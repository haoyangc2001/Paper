"""Go2 物理占地轮廓的构造与覆盖检查（CAL-021）。

数据来源只有两处，二者都不含正式校准/检验结果：

1. **冻结资产**：``config/go2_asset_geometry_v1.json``（由 ``extract_asset_geometry.py``
   从 ``<CAMRA_ASSET_ROOT>/Go2/usd`` 只读提取：19 个链接、18 个关节、27 个碰撞体）；
2. **独立 clean 基线 + pilot** 的 50 Hz 关节记录（``state_50hz.npz:joint_pos``），
   用来给出"腿部运动包络"——即各关节实际出现过的角度范围。

坐标约定：全部在**机身坐标系**（x 前、y 左、z 上，原点在 base 链接原点）；
轮廓是 xy 平面上的凸多边形，单位米。本模块不启动仿真、不修改任何资产。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .geometry import Polygon, convex_hull_2d, minkowski_sum

# 冻结清单 contract/action/joint_names 给出的策略关节顺序；build 时会用初始姿态反查校验
POLICY_JOINT_ORDER = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
# 采集环境 init_state（unitree.py UNITREE_GO2_CFG）展开到具体关节名：用于校验关节顺序与名义站姿
INIT_JOINT_POS = {
    "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
}
LEG_PREFIXES = ("FL", "FR", "RL", "RR")
BODY_LINKS = ("base", "Head_upper", "Head_lower")


class FootprintError(RuntimeError):
    """资产几何、关节记录或轮廓参数不合法。"""


def init_joint_pose(joint_names: Sequence[str] = POLICY_JOINT_ORDER) -> np.ndarray:
    """采集环境 init_state 的关节角向量（用于名义站姿与关节顺序自检）。"""
    try:
        return np.array([INIT_JOINT_POS[n] for n in joint_names], dtype=np.float64)
    except KeyError as exc:
        raise FootprintError(f"关节 {exc.args[0]!r} 没有对应的 init_state 值") from exc


# ---------------------------------------------------------------- 资产几何
def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """[w,x,y,z] → 3x3 旋转矩阵。"""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise FootprintError("四元数模长为 0")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_mat(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise FootprintError(f"未知转轴 {axis!r}")


def _batch_axis_rot(axis: str, ang: np.ndarray) -> np.ndarray:
    """批量轴向旋转矩阵 (N,3,3)。"""
    ang = np.asarray(ang, dtype=np.float64)
    n = ang.shape[0]
    c, s = np.cos(ang), np.sin(ang)
    out = np.zeros((n, 3, 3))
    if axis == "X":
        out[:, 0, 0] = 1.0
        out[:, 1, 1], out[:, 1, 2], out[:, 2, 1], out[:, 2, 2] = c, -s, s, c
    elif axis == "Y":
        out[:, 1, 1] = 1.0
        out[:, 0, 0], out[:, 0, 2], out[:, 2, 0], out[:, 2, 2] = c, s, -s, c
    elif axis == "Z":
        out[:, 2, 2] = 1.0
        out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = c, -s, s, c
    else:
        raise FootprintError(f"未知转轴 {axis!r}")
    return out


def _homog(rot: np.ndarray, trans: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(trans, dtype=np.float64)
    return m


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    frame_parent: np.ndarray   # 4x4：parent 系下的关节零位姿（Trans(pos0)·Rot(rot0)）
    frame_child_inv: np.ndarray  # 4x4：(Trans(pos1)·Rot(rot1))⁻¹，子链接系下的关节系之逆
    axis: str | None
    lower: float
    upper: float


@dataclass(frozen=True)
class Collider:
    link: str
    kind: str
    matrix: np.ndarray         # 4x4：link 系下 shape 位姿
    local_points: np.ndarray   # (m,3)：shape 局部采样点

    def points_in_link(self) -> np.ndarray:
        p = self.local_points @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        return p


def _primitive_points(kind: str, params: dict, ring: int = 24) -> np.ndarray:
    """碰撞体的局部采样点；对凸体取其表面足够密的点即可（外面再取凸包）。"""
    ang = np.linspace(0.0, 2.0 * math.pi, ring, endpoint=False)
    if kind == "cube":
        s = float(params.get("size", 1.0)) / 2.0
        return np.array([[sx, sy, sz] for sx in (-s, s) for sy in (-s, s) for sz in (-s, s)])
    if kind == "sphere":
        r = float(params["radius"])
        pts = [np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang)], 1),
               np.stack([r * np.cos(ang), np.zeros_like(ang), r * np.sin(ang)], 1),
               np.stack([np.zeros_like(ang), r * np.cos(ang), r * np.sin(ang)], 1),
               np.array([[0, 0, r], [0, 0, -r]])]
        return np.vstack(pts)
    if kind in ("cylinder", "capsule"):
        r, h = float(params["radius"]), float(params["height"])
        axis = str(params.get("axis", "Z"))
        rings = []
        for z in (-h / 2.0, 0.0, h / 2.0):
            ring_pts = np.stack([r * np.cos(ang), r * np.sin(ang), np.full_like(ang, z)], 1)
            if axis == "X":
                ring_pts = ring_pts[:, [2, 1, 0]]
            elif axis == "Y":
                ring_pts = ring_pts[:, [0, 2, 1]]
            rings.append(ring_pts)
        return np.vstack(rings)
    raise FootprintError(f"未知碰撞体类型 {kind!r}")


@dataclass(frozen=True)
class AssetGeometry:
    """冻结资产的几何与运动学（可直接从 JSON 复算）。"""

    source: dict
    links: dict
    joints: tuple[Joint, ...]
    colliders: tuple[Collider, ...]
    path: Path | None = None

    # ---- 加载 ----
    @classmethod
    def load(cls, path: Path | str, verify_hash: bool = True) -> "AssetGeometry":
        p = Path(path)
        if not p.is_file():
            raise FootprintError(f"资产几何文件不存在: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != "camra-go2-asset-geometry-v1":
            raise FootprintError(f"未知 schema: {payload.get('schema')!r}")
        if verify_hash:
            side = p.with_suffix(p.suffix + ".sha256")
            if not side.is_file():
                raise FootprintError(f"缺少资产几何散列侧车: {side}")
            want = side.read_text(encoding="utf-8").split()[0]
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            if want != got:
                raise FootprintError(f"资产几何散列不匹配：{want} vs {got}")
        joints = []
        for j in payload["joints"]:
            # PhysX/USD 约定：关节系在父链接里是 Trans(pos0)·Rot(rot0)，
            # 在子链接里是 Trans(pos1)·Rot(rot1)，故子链接位姿需要乘后者之逆。
            frame_parent = _homog(quat_to_mat(j["rot0"]), j["pos0"])
            frame_child_inv = np.linalg.inv(_homog(quat_to_mat(j["rot1"]), j["pos1"]))
            joints.append(Joint(name=j["name"], kind=j["kind"], parent=j["parent_link"],
                                child=j["child_link"], frame_parent=frame_parent,
                                frame_child_inv=frame_child_inv,
                                axis=j.get("axis"), lower=float(j.get("lower_rad", 0.0)),
                                upper=float(j.get("upper_rad", 0.0))))
        colliders = []
        for c in payload["colliders"]:
            matrix = np.asarray(c["matrix_link_from_shape"], dtype=np.float64).reshape(4, 4)
            colliders.append(Collider(link=c["link"], kind=c["kind"], matrix=matrix,
                                      local_points=_primitive_points(c["kind"], c)))
        return cls(source=payload["source"], links=payload["links"],
                   joints=tuple(joints), colliders=tuple(colliders), path=p)

    # ---- 查询 ----
    def link_names(self) -> list[str]:
        return list(self.links)

    def joints_of_leg(self, leg: str) -> list[str]:
        return [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"]

    def chain_to(self) -> list[Joint]:
        """按父→子拓扑排序的关节序列。"""
        by_parent: dict[str, list[Joint]] = {}
        for j in self.joints:
            by_parent.setdefault(j.parent, []).append(j)
        order: list[Joint] = []
        queue = ["base"]
        seen = {"base"}
        while queue:
            cur = queue.pop(0)
            for j in by_parent.get(cur, []):
                order.append(j)
                if j.child in seen:
                    raise FootprintError(f"关节链出现重复链接 {j.child}")
                seen.add(j.child)
                queue.append(j.child)
        if len(order) != len(self.joints):
            raise FootprintError(f"关节链未覆盖全部关节：{len(order)}/{len(self.joints)}")
        return order

    # ---- 正运动学 ----
    def _joint_rot_single(self, j: Joint, q: dict[str, float]) -> np.ndarray:
        if j.kind == "revolute":
            return axis_angle_mat(j.axis or "Z", float(q.get(j.name, 0.0)))
        return np.eye(3)

    def fk(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        """单帧 FK：返回 link → 4x4（base 系）。

        T_child = T_parent · Frame_parent · Rot(axis,q) · Frame_child⁻¹
        """
        poses: dict[str, np.ndarray] = {"base": np.eye(4)}
        for j in self.chain_to():
            rot = self._joint_rot_single(j, q)
            step = j.frame_parent @ _homog(rot, (0, 0, 0)) @ j.frame_child_inv
            poses[j.child] = poses[j.parent] @ step
        return poses

    def fk_batch(self, Q: np.ndarray, joint_names: Sequence[str]) -> dict[str, np.ndarray]:
        """批量 FK：Q 形状 (N,J)。返回 link → (N,4,4)。"""
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != len(joint_names):
            raise FootprintError(f"关节角矩阵形状应为 (N,{len(joint_names)})，得到 {Q.shape}")
        n = Q.shape[0]
        idx = {name: i for i, name in enumerate(joint_names)}
        poses: dict[str, np.ndarray] = {"base": np.tile(np.eye(4), (n, 1, 1))}
        for j in self.chain_to():
            parent = poses[j.parent]
            rp, rci = j.frame_parent[:3, :3], j.frame_child_inv[:3, :3]
            tp, tci = j.frame_parent[:3, 3], j.frame_child_inv[:3, 3]
            if j.kind == "revolute":
                if j.name not in idx:
                    raise FootprintError(f"关节记录缺少 {j.name}")
                rot = _batch_axis_rot(j.axis or "Z", Q[:, idx[j.name]])
            else:
                rot = np.tile(np.eye(3), (n, 1, 1))
            # 旋转：Rp · R(q) · Rci ；平移：tp + Rp · (R(q) · tci)
            rq_tci = np.einsum("nij,j->ni", rot, tci)
            step = np.tile(np.eye(4), (n, 1, 1))
            step[:, :3, :3] = rp @ rot @ rci
            step[:, :3, 3] = tp + rq_tci @ rp.T
            poses[j.child] = parent @ step
        return poses

    # ---- 自洽校验 ----
    def rest_consistency(self) -> dict:
        """FK(全部关节=0) 必须复现资产里作者写的链接静止平移。"""
        poses = self.fk({})
        worst = 0.0
        rows = []
        for name, info in self.links.items():
            got = poses[name][:3, 3]
            want = np.asarray(info["rest_xyz"], dtype=np.float64)
            err = float(np.abs(got - want).max())
            worst = max(worst, err)
            rows.append({"link": name, "fk_xyz": got.tolist(), "asset_xyz": want.tolist(),
                         "max_abs_err": err})
        return {"max_abs_err": worst, "rows": rows}


# ---------------------------------------------------------------- 轮廓构造
@dataclass(frozen=True)
class OutlineResult:
    polygon: Polygon
    points_xy: np.ndarray      # 参与判定的采样点（已去重/裁剪）
    n_configs: int


def _points_by_link(geom: AssetGeometry) -> dict[str, np.ndarray]:
    """把每个链接下的碰撞体采样点合并成 (m,3)，只算一次。"""
    out: dict[str, list[np.ndarray]] = {}
    for col in geom.colliders:
        out.setdefault(col.link, []).append(col.points_in_link())
    return {k: np.vstack(v) for k, v in out.items()}


def outline_from_q(geom: AssetGeometry, Q: np.ndarray, joint_names: Sequence[str],
                   links: Sequence[str] | None = None, chunk: int = 512) -> Polygon:
    """对一批关节角（N,J）求占地轮廓：全部帧的投影点一次取凸包。

    分批是为了控内存；每批的点取凸包后与已累积的凸包顶点合并再取凸包，
    结果与"把所有点堆在一起取凸包"完全一致（凸包可结合）。
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    by_link = _points_by_link(geom)
    use = list(links) if links is not None else list(by_link)
    acc = np.zeros((0, 2))
    for start in range(0, Q.shape[0], chunk):
        sub = Q[start:start + chunk]
        poses = geom.fk_batch(sub, joint_names)
        blocks = []
        for link in use:
            if link not in by_link:
                continue
            T = poses[link]                       # (n,4,4)
            P = by_link[link]                     # (m,3)
            world = np.einsum("nij,mj->nmi", T[:, :3, :3], P) + T[:, None, :3, 3]
            blocks.append(world[:, :, :2].reshape(-1, 2))
        pts = np.vstack(blocks + [acc])
        acc = convex_hull_2d(pts)
    return Polygon(acc)


def coverage(polygon: Polygon, geom: AssetGeometry, Q: np.ndarray,
             joint_names: Sequence[str], chunk: int = 256,
             links: Sequence[str] | None = None) -> dict:
    """独立覆盖检查：逐帧投影的**全部点**必须落在轮廓内。

    返回最大越界距离（正数即越界）、越界点数与最差帧索引，便于定位问题。
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    by_link = _points_by_link(geom)
    use = list(links) if links is not None else list(by_link)
    worst = -float("inf")
    n_out = 0
    worst_frame = -1
    for start in range(0, Q.shape[0], chunk):
        sub = Q[start:start + chunk]
        poses = geom.fk_batch(sub, joint_names)
        blocks = []
        for link in use:
            if link not in by_link:
                continue
            T = poses[link]
            world = np.einsum("nij,mj->nmi", T[:, :3, :3], by_link[link]) + T[:, None, :3, 3]
            blocks.append(world[:, :, :2])
        stacked = np.concatenate(blocks, axis=1)      # (n, m_total, 2)
        d = polygon.distance(stacked.reshape(-1, 2))
        if d.size:
            if d.max() > worst:
                worst = float(d.max())
                worst_frame = start + int(np.argmax(d.reshape(stacked.shape[0], -1).max(axis=1)))
            n_out += int((d > 1e-9).sum())
    return {"max_outside_m": worst, "n_points_outside": n_out,
            "worst_frame": worst_frame, "n_configs": int(Q.shape[0])}


def joint_ranges(Q: np.ndarray, joint_names: Sequence[str]) -> dict:
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    return {name: {"min_rad": float(Q[:, i].min()), "max_rad": float(Q[:, i].max()),
                   "min_deg": float(np.degrees(Q[:, i].min())),
                   "max_deg": float(np.degrees(Q[:, i].max()))}
            for i, name in enumerate(joint_names)}


@dataclass(frozen=True)
class Footprint:
    """冻结后的占地轮廓（含方法与溯源，可独立复算）。"""

    polygon: Polygon
    version: str
    frame: str
    method: str
    source: dict
    joint_ranges: dict
    stats: dict
    notes: tuple[str, ...] = ()

    # ---- 序列化 ----
    def content_dict(self) -> dict:
        return {
            "version": self.version,
            "frame": self.frame,
            "unit": "m",
            "method": self.method,
            "polygon": self.polygon.as_dict(),
            "source": self.source,
            "joint_ranges_rad": self.joint_ranges,
            "stats": self.stats,
        }

    def to_dict(self) -> dict:
        out = self.content_dict()
        out["notes"] = list(self.notes)
        return out

    def sha256(self) -> str:
        blob = json.dumps(self.content_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: Path | str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.to_dict(), sha256=self.sha256())
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
        p.with_suffix(p.suffix + ".sha256").write_text(f"{self.sha256()}  {p.name}\n",
                                                      encoding="utf-8")
        return self.sha256()

    @classmethod
    def load(cls, path: Path | str, verify_hash: bool = True) -> "Footprint":
        p = Path(path)
        if not p.is_file():
            raise FootprintError(f"轮廓文件不存在: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        fp = cls(polygon=Polygon.from_dict(payload["polygon"]),
                 version=str(payload["version"]), frame=str(payload["frame"]),
                 method=str(payload["method"]), source=dict(payload["source"]),
                 joint_ranges=dict(payload["joint_ranges_rad"]),
                 stats=dict(payload["stats"]), notes=tuple(payload.get("notes", ())))
        if verify_hash:
            want = payload.get("sha256")
            if want is not None and want != fp.sha256():
                raise FootprintError(f"轮廓散列不匹配：{want} vs {fp.sha256()}")
            side = p.with_suffix(p.suffix + ".sha256")
            if side.is_file() and side.read_text(encoding="utf-8").split()[0] != fp.sha256():
                raise FootprintError("轮廓侧车散列不匹配")
        return fp

    # ---- 使用侧 ----
    def inflated(self, delta: float) -> Polygon:
        """轮廓 ⊕ disc(delta)。**只做一次**膨胀，禁止对结果再叠一次 delta。"""
        if delta < 0:
            raise FootprintError("delta 必须 ≥ 0")
        return self.polygon.inflate(delta)

    def obstacle_dilation(self, obstacle: Polygon, delta: float = 0.0) -> Polygon:
        """危险集扩张：obstacle ⊕ footprint ⊕ disc(delta)。

        轮廓与 delta 各只进入一次——这是协议 ``near_obstacle.robot_footprint``
        "delta 膨胀只应用一次"的具体实现；重复调用本函数会重复叠加，禁止。
        """
        grown = self.inflated(delta)
        return minkowski_sum(obstacle, grown)

    def contains(self, points) -> np.ndarray:
        return self.polygon.contains(points)
