"""冻结采样协议的加载与散列校验（对应 config/calibration_protocol_v<ver>.json）。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VERSION = "1.0.0"


class ProtocolError(RuntimeError):
    """协议缺失、JSON 非法或散列不匹配。"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_path(version: str = DEFAULT_VERSION) -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config" / f"calibration_protocol_v{version}.json"


@dataclass(frozen=True)
class Protocol:
    path: Path
    payload: dict
    sha256: str

    # ---- 基本属性 ----
    @property
    def version(self) -> str:
        return str(self.payload.get("version", ""))

    @property
    def frozen(self) -> bool:
        return self.payload.get("state") == "frozen"

    @property
    def unit(self) -> str:
        return str(self.payload["energy"]["unit"])

    @property
    def j_min(self) -> float:
        return float(self.payload["energy"]["near_zero"]["j_min_J"])

    @property
    def measured_steps(self) -> int:
        return int(self.payload["clocks"]["measured_steps"])

    @property
    def cells(self) -> int:
        return int(self.payload["split"]["cells"])

    def family_ids(self) -> list[str]:
        return [f["id"] for f in self.payload["command_program"]["families"]]

    def profile_ids(self) -> list[str]:
        return [p["id"] for p in self.payload["profiles"]]

    def quota(self, split: str) -> int:
        """给定 split 名返回该域总条数；baseline_dev 等子集取对应子集配额。"""
        s = self.payload["split"]
        if split in ("baseline_fit", "baseline_dev", "baseline_validate"):
            return int(s["baseline"]["subsets"][split.replace("baseline_", "")])
        return int(s[split]["total"])

    def seed_block(self, split: str) -> tuple[int, int]:
        """返回该 split 的 (start, count)。"""
        blocks = self.payload["seeds"]["blocks"]
        if split not in blocks:
            raise ProtocolError(f"split {split!r} 没有对应的种子块（可用：{sorted(blocks)}）")
        spec = blocks[split]
        return int(spec["start"]), int(spec["count"])

    # ---- 加载 ----
    @classmethod
    def load(cls, path: Path | None = None, verify_hash: bool = True,
             version: str = DEFAULT_VERSION) -> "Protocol":
        p = Path(path) if path is not None else default_path(version)
        if not p.is_file():
            raise ProtocolError(f"协议文件不存在: {p}")
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"协议 JSON 非法: {p}: {exc}") from exc
        digest = sha256_file(p)
        if verify_hash:
            hf = p.with_suffix(p.suffix + ".sha256")
            if not hf.is_file():
                raise ProtocolError(f"缺少冻结散列文件: {hf}")
            recorded = hf.read_text(encoding="utf-8").split()[0]
            if recorded != digest:
                raise ProtocolError(
                    f"协议散列不匹配（文件被改动？）\n  recorded {recorded}\n  actual   {digest}")
        if payload.get("state") != "frozen":
            raise ProtocolError(f"协议未处于 frozen 状态: state={payload.get('state')!r}")
        return cls(path=p, payload=payload, sha256=digest)
