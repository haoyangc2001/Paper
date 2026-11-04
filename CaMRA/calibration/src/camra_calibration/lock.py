"""正式数据锁（CAL-023）：绑定所有依赖、配额、seed、统计口径与资源预算。

锁的作用是"机检 + 采集前置门禁"：采集正式 split 时必须给出锁文件，锁里的散列与现场
文件不符、split 不在锁里、或检验/近障阶段的 margin 尚未冻结，采集器都会拒绝启动。

本模块不启动仿真、不读 torch，便于在 CPU 上单测。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FORMAL_SPLITS = ("calibration", "heldout", "near_obstacle")
MARGIN_GATED_SPLITS = ("heldout", "near_obstacle")     # 需 margin 冻结后才可采
MAP_GATED_SPLITS = ("near_obstacle",)                   # 需冻结地图与命令清单


class LockError(RuntimeError):
    """锁缺失、散列不符或采集前置条件不满足。"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DataLock:
    payload: dict
    path: Path
    sha256: str

    # ---- 加载 ----
    @classmethod
    def load(cls, path: Path | str, verify_hash: bool = True) -> "DataLock":
        p = Path(path)
        if not p.is_file():
            raise LockError(f"锁文件不存在: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("schema") != "camra-data-lock-v1":
            raise LockError(f"未知 schema: {payload.get('schema')!r}")
        digest = sha256_file(p)
        if verify_hash:
            side = p.with_suffix(p.suffix + ".sha256")
            if not side.is_file():
                raise LockError(f"缺少锁散列侧车: {side}")
            if side.read_text(encoding="utf-8").split()[0] != digest:
                raise LockError("锁文件散列与其侧车不符（文件被改动？）")
        return cls(payload=payload, path=p, sha256=digest)

    # ---- 查询 ----
    @property
    def protocol(self) -> dict:
        return self.payload["protocol"]

    def list_entry(self, split: str) -> dict | None:
        return self.payload["trial_lists"]["files"].get(split)

    def quota(self, split: str) -> int | None:
        entry = self.list_entry(split)
        return None if entry is None else int(entry["count"])

    def seed_block(self, split: str) -> tuple[int, int] | None:
        entry = self.list_entry(split)
        if entry is None or "seed_block" not in entry:
            return None
        return int(entry["seed_block"][0]), int(entry["seed_block"][1])

    # ---- 校验 ----
    def verify_artifacts(self, root: Path) -> list[str]:
        """逐项核对锁里记录的散列与现场文件；返回不一致项（空表示一致）。"""
        bad: list[str] = []
        for key in ("protocol", "power_model", "footprint", "step_cost_schema",
                    "near_obstacle_maps"):
            entry = self.payload.get(key)
            if not entry:
                continue
            p = _resolve(root, entry["path"])
            if not p.is_file():
                bad.append(f"{key}: 文件不存在 {p}")
            elif sha256_file(p) != entry["sha256"]:
                bad.append(f"{key}: 散列不符 {p}")
        asset = self.payload.get("asset", {})
        for key in ("geometry", "usd", "physics_usd"):
            entry = asset.get(key)
            if not entry:
                continue
            p = _resolve(root, entry["path"])
            if not p.is_file():
                bad.append(f"asset.{key}: 文件不存在 {p}")
            elif sha256_file(p) != entry["sha256"]:
                bad.append(f"asset.{key}: 散列不符 {p}")
        for split, entry in self.payload["trial_lists"]["files"].items():
            p = _resolve(root, entry["path"])
            if not p.is_file():
                bad.append(f"trial_lists.{split}: 文件不存在 {p}")
            elif sha256_file(p) != entry["sha256"]:
                bad.append(f"trial_lists.{split}: 散列不符 {p}")
        for rel, digest in self.payload.get("code", {}).get("files", {}).items():
            p = _resolve(root, rel)
            if not p.is_file():
                bad.append(f"code: 文件不存在 {p}")
            elif sha256_file(p) != digest:
                bad.append(f"code: 散列不符 {p}")
        return bad


def _resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def enforce_collection(lock: DataLock, split: str, root: Path, *,
                       manifest_dir: Path | None = None,
                       maps_path: Path | None = None,
                       margins_path: Path | None = None,
                       allow_overwrite: bool = False) -> None:
    """采集前置门禁；任何一条不满足就抛 :class:`LockError`。

    * split 必须在锁里（防止拼错的 split 或未登记的域）；
    * 现场文件的散列必须与锁一致（防止换模型/换轮廓后偷偷采集）；
    * 检验与近障要求 margin 已冻结（``margins_path`` 存在）；
    * 近障还要求冻结地图，且其散列与锁一致；
    * 正式 split 一律拒绝 ``allow_overwrite``（重试只走"删/改名 DONE + 新 attempt"）。
    """
    if split not in lock.payload["trial_lists"]["files"]:
        raise LockError(f"split {split!r} 不在数据锁里（可用："
                        f"{sorted(lock.payload['trial_lists']['files'])}）")
    bad = lock.verify_artifacts(root)
    if bad:
        raise LockError("锁与现场不一致：\n  " + "\n  ".join(bad[:8]))
    if manifest_dir is not None:
        entry = lock.list_entry(split)
        p = Path(manifest_dir) / f"{split}.jsonl"
        if not p.is_file():
            raise LockError(f"清单文件不存在: {p}")
        if sha256_file(p) != entry["sha256"]:
            raise LockError(f"清单 {p} 与锁记录的散列不符")
    if split in FORMAL_SPLITS and allow_overwrite:
        raise LockError(f"{split} 属正式数据，禁止 --allow-overwrite；"
                        "重试请改名/删除 DONE 后以新 attempt 采集")
    if split in MARGIN_GATED_SPLITS:
        mp = Path(margins_path) if margins_path else Path("config/margins_v1.json")
        if not mp.is_file():
            raise LockError(f"{split} 需要先冻结 margin（缺 {mp}）："
                            "CAL-025 完成前不得采集检验/近障数据")
    if split in MAP_GATED_SPLITS:
        if maps_path is None or not Path(maps_path).is_file():
            raise LockError("near_obstacle 采集必须给出 --maps（冻结地图与命令清单）")
        recorded = lock.payload["near_obstacle_maps"]["sha256"]
        if sha256_file(Path(maps_path)) != recorded:
            raise LockError("--maps 的散列与锁记录不一致")
