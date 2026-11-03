"""trial 清单的加载与 schema 校验（CAL-010）。

清单为 JSONL：每行一条 trial 记录。校验覆盖
  * 必备字段与类型
  * trial_id 格式与唯一性
  * split / family / profile 合法性，seed 落在协议对应种子块内
  * 四类产物路径（200 Hz 物理测量、50 Hz 状态、10 Hz 配对、能量 summary）的相对性、
    后缀、目录逃逸与跨 trial 复用
  * 能量 summary 的单位、非负性、近零标记与协议 j_min 的一致性
  * 协议版本与散列与当前冻结协议一致
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .protocol import Protocol

SPLITS: tuple[str, ...] = (
    "pilot",
    "baseline_fit",
    "baseline_dev",
    "baseline_validate",
    "calibration",
    "heldout",
    "near_obstacle",
)

#: 产物种类 -> 允许的后缀
KINDS: dict[str, tuple[str, ...]] = {
    "physics_200hz": (".npz",),
    "state_50hz": (".npz",),
    "pairing_10hz": (".npz",),
    "energy_summary": (".json",),
}

OUTCOMES = ("completed", "terminated", "truncated")

REQUIRED: tuple[str, ...] = (
    "trial_id", "split", "family", "profile", "seed", "attempt",
    "controller", "protocol_version", "protocol_sha256", "bundle_tree_sha256",
    "outcome", "steps_executed", "profile_applied", "initial_state", "paths", "energy",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(RuntimeError):
    """清单无法读取或整体结构错误（单条记录的问题见 ManifestReport.errors）。"""


@dataclass
class ManifestReport:
    total: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_split: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, trial_id: str, msg: str) -> None:
        self.errors.append(f"{trial_id}: {msg}")

    def add_warning(self, trial_id: str, msg: str) -> None:
        self.warnings.append(f"{trial_id}: {msg}")

    def summary(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(self.per_split.items()))
        head = f"{self.total} trials" + (f" ({counts})" if counts else "")
        return f"{head}, {len(self.errors)} errors, {len(self.warnings)} warnings"


def load_manifest(path: Path) -> list[dict]:
    """读取 JSONL 清单；空行忽略，非对象或 JSON 非法直接报错。"""
    p = Path(path)
    if not p.is_file():
        raise ManifestError(f"清单不存在: {p}")
    records: list[dict] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{p}:{lineno} JSON 非法: {exc}") from exc
        if not isinstance(obj, dict):
            raise ManifestError(f"{p}:{lineno} 期望 JSON 对象，得到 {type(obj).__name__}")
        records.append(obj)
    return records


def _check_path(rel, kind: str, root: Path | None, report: ManifestReport, tid: str,
                require_files: bool) -> str | None:
    if not isinstance(rel, str) or not rel.strip():
        report.add_error(tid, f"paths.{kind} 缺失或非字符串")
        return None
    rp = Path(rel)
    if rp.is_absolute():
        report.add_error(tid, f"paths.{kind} 必须是相对路径，不能是绝对路径: {rel}")
        return None
    if ".." in rp.parts:
        report.add_error(tid, f"paths.{kind} 含目录逃逸 '..': {rel}")
        return None
    if not rel.endswith(KINDS[kind]):
        report.add_error(tid, f"paths.{kind} 后缀应为 {KINDS[kind]}，得到 {rel}")
        return None
    if root is None:
        # 没有根目录时，用规范化后的相对路径作为去重键，仍能发现跨 trial 复用
        return str(PurePosixPath(*rp.parts))
    resolved = (root / rp).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        report.add_error(tid, f"paths.{kind} 解析后逃出根目录: {resolved}")
        return None
    if require_files and not resolved.is_file():
        report.add_error(tid, f"paths.{kind} 指向的文件不存在: {resolved}")
    return str(resolved)


def validate_manifest(source, protocol: Protocol | None = None, root: Path | None = None,
                      require_files: bool = False) -> ManifestReport:
    """校验清单。source 可以是路径或已读取的记录列表。"""
    if isinstance(source, (str, Path)):
        records = load_manifest(Path(source))
    else:
        records = list(source)
    if protocol is None:
        protocol = Protocol.load()
    report = ManifestReport()
    report.total = len(records)
    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, str] = {}
    fam = set(protocol.family_ids())
    prof = set(protocol.profile_ids())

    for idx, rec in enumerate(records):
        tid = str(rec.get("trial_id") or f"<record {idx}>")

        # 1. 必备字段
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            report.add_error(tid, f"缺少必备字段 {missing}")
            continue

        # 2. trial_id 格式与一致性
        split = rec.get("split")
        m = re.fullmatch(r"([a-z_]+)-([a-z_]+)-([a-z_]+)-(\d{4})", str(rec["trial_id"]))
        if not m:
            report.add_error(tid, "trial_id 格式应为 <split>-<family>-<profile>-<NNNN>")
        else:
            t_split, t_fam, t_prof, _ = m.groups()
            if t_split != split:
                report.add_error(tid, f"trial_id 中的 split={t_split} 与字段 split={split} 不一致")
            if t_fam != rec.get("family"):
                report.add_error(tid, f"trial_id 中的 family={t_fam} 与字段不一致")
            if t_prof != rec.get("profile"):
                report.add_error(tid, f"trial_id 中的 profile={t_prof} 与字段不一致")

        # 3. 重复
        if tid in seen_ids:
            report.add_error(tid, "trial_id 重复出现")
        else:
            seen_ids[tid] = idx
        report.per_split[split] = report.per_split.get(split, 0) + 1

        # 4. 枚举与协议一致性
        if split not in SPLITS:
            report.add_error(tid, f"split 非法: {split!r}（允许 {SPLITS}）")
        if rec.get("family") not in fam:
            report.add_error(tid, f"family 不在协议命令族中: {rec.get('family')!r}")
        if rec.get("profile") not in prof:
            report.add_error(tid, f"profile 不在协议 profile 中: {rec.get('profile')!r}")

        seed = rec.get("seed")
        if not isinstance(seed, int):
            report.add_error(tid, f"seed 必须是整数，得到 {type(seed).__name__}")
        elif split in SPLITS:
            start, count = protocol.seed_block(split)
            if not (start <= seed < start + count):
                report.add_error(tid, f"seed={seed} 不在 {split} 的种子块 [{start}, {start + count}) 内")

        if not isinstance(rec.get("attempt"), int) or rec["attempt"] < 1:
            report.add_error(tid, f"attempt 必须是 >=1 的整数，得到 {rec.get('attempt')!r}")

        # 5. 结果与步数
        if rec.get("outcome") not in OUTCOMES:
            report.add_error(tid, f"outcome 非法: {rec.get('outcome')!r}（允许 {OUTCOMES}）")
        steps = rec.get("steps_executed")
        limit = protocol.measured_steps
        if not isinstance(steps, int) or not (1 <= steps <= limit):
            report.add_error(tid, f"steps_executed 必须在 1..{limit}，得到 {steps!r}")
        elif rec.get("outcome") == "completed" and steps != limit:
            report.add_error(tid, f"outcome=completed 时 steps_executed 必须等于 {limit}，得到 {steps}")

        # 6. 版本与散列绑定
        if rec.get("controller") != "camra_go2_velocity_it2999_v2":
            report.add_error(tid, f"controller 不符: {rec.get('controller')!r}")
        if rec.get("protocol_version") != protocol.version:
            report.add_error(tid, f"protocol_version 与当前协议 {protocol.version} 不一致")
        if rec.get("protocol_sha256") != protocol.sha256:
            report.add_error(tid, "protocol_sha256 与当前冻结协议不一致（清单可能由旧协议生成）")
        if not HEX64.match(str(rec.get("bundle_tree_sha256", ""))):
            report.add_error(tid, "bundle_tree_sha256 必须是 64 位十六进制")

        # 7. 记录的实际配置与初态
        pa = rec.get("profile_applied")
        if not isinstance(pa, dict):
            report.add_error(tid, "profile_applied 必须是对象")
        else:
            for key in ("friction", "added_mass_kg", "push_events", "obs_noise"):
                if key not in pa:
                    report.add_error(tid, f"profile_applied 缺少 {key}")
        ist = rec.get("initial_state")
        if not isinstance(ist, dict):
            report.add_error(tid, "initial_state 必须是对象")
        else:
            for key in ("px", "py", "yaw"):
                if key not in ist:
                    report.add_error(tid, f"initial_state 缺少 {key}")

        # 8. 四类产物路径
        paths = rec.get("paths")
        if not isinstance(paths, dict):
            report.add_error(tid, "paths 必须是对象")
        else:
            missing_kinds = [k for k in KINDS if k not in paths]
            if missing_kinds:
                report.add_error(tid, f"paths 缺少 {missing_kinds}")
            for kind in KINDS:
                if kind not in paths:
                    continue
                key = _check_path(paths[kind], kind, root, report, tid, require_files)
                if key is not None:
                    if key in seen_paths and seen_paths[key] != tid:
                        report.add_error(tid, f"paths.{kind} 与其他 trial 复用同一文件: {paths[kind]}（已属 {seen_paths[key]}）")
                    else:
                        seen_paths[key] = tid

        # 9. 能量 summary
        en = rec.get("energy")
        if not isinstance(en, dict):
            report.add_error(tid, "energy 必须是对象")
        else:
            if en.get("unit") != protocol.unit:
                report.add_error(tid, f"energy.unit 必须是 {protocol.unit!r}，得到 {en.get('unit')!r}")
            for key in ("j_exec", "j_nom"):
                val = en.get(key)
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    report.add_error(tid, f"energy.{key} 必须是数值，得到 {val!r}")
                elif val < 0:
                    report.add_error(tid, f"energy.{key} 不能为负: {val}")
            if en.get("j_min") != protocol.j_min:
                report.add_error(tid, f"energy.j_min 必须是 {protocol.j_min}，得到 {en.get('j_min')!r}")
            nc = en.get("non_computable")
            if not isinstance(nc, bool):
                report.add_error(tid, "energy.non_computable 必须是布尔值")
            elif isinstance(en.get("j_nom"), (int, float)):
                should = en["j_nom"] < protocol.j_min
                if nc != should:
                    report.add_error(
                        tid, f"energy.non_computable={nc} 与 j_nom={en['j_nom']} 对 j_min={protocol.j_min} 的判定不符")

    return report
