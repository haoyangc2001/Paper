"""采集落盘与批次规划（CAL-015/017 抽出，便于纯 CPU 测试）。

设计约束（协议 split / 采集器 §17）：

* **批次大小不影响清单与派生统计**：``plan_batches`` 只是把同一有序列表切片，
  不重排、不重采样；任何 n_env 下 ``flatten(plan_batches(trials, n))`` 都等于 trials。
* **断点续采幂等**：``resume_filter`` 只跳过已 ``DONE`` 的 trial；重复调用结果相同。
* **绝不覆盖旧分片**：``prepare`` 遇到已存在但没有 ``DONE`` 的目录，改用
  ``<trial_id>__a<n>``，原目录保持不动。
* 写文件先写 ``.tmp`` 再 ``os.replace``，``DONE`` 最后写；records 追加后 fsync。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

DONE_MARKER = "DONE"


def plan_batches(trials: Sequence[dict], n_env: int) -> list[list[dict]]:
    """把有序 trial 列表按 n_env 切片；不重排、不丢弃。"""
    if n_env <= 0:
        raise ValueError(f"n_env 必须 > 0，得到 {n_env}")
    return [list(trials[i:i + n_env]) for i in range(0, len(trials), n_env)]


def flatten_batches(batches: Iterable[Sequence[dict]]) -> list[dict]:
    return [t for b in batches for t in b]


def resume_filter(trials: Sequence[dict], store: "TrialStore",
                  allow_overwrite: bool = False) -> tuple[list[dict], int]:
    """返回 (待采列表, 跳过数)；已 DONE 的跳过，顺序不变。"""
    if allow_overwrite:
        return list(trials), 0
    todo = [t for t in trials if not store.is_done(t["trial_id"])]
    return todo, len(trials) - len(todo)


class TrialStore:
    """``<root>/<split>/<trial_id>/`` 布局 + 原子写 + 续采。"""

    def __init__(self, root: Path | str, split: str,
                 writer: Callable[[Path, dict], None] | None = None):
        self.base = Path(root) / split
        self.base.mkdir(parents=True, exist_ok=True)
        self.records = self.base / "records.jsonl"
        self._writer = writer or self._default_npz_writer

    # ---- 查询 ----
    def trial_dir(self, trial_id: str) -> Path:
        """已存在目录（含 attempt 后缀）中优先返回 DONE 的那个。"""
        primary = self.base / trial_id
        if primary.is_dir():
            return primary
        cands = sorted(self.base.glob(f"{trial_id}__a*"))
        return cands[-1] if cands else primary

    def is_done(self, trial_id: str) -> bool:
        d = self.base / trial_id
        if (d / DONE_MARKER).is_file():
            return True
        return any((p / DONE_MARKER).is_file() for p in self.base.glob(f"{trial_id}__a*"))

    def done_ids(self) -> set[str]:
        out: set[str] = set()
        for p in self.base.iterdir():
            if p.is_dir() and (p / DONE_MARKER).is_file():
                out.add(p.name.split("__a")[0])
        return out

    # ---- 写入 ----
    def prepare(self, trial_id: str) -> tuple[Path, int]:
        d = self.base / trial_id
        if not d.exists():
            d.mkdir(parents=True)
            return d, 1
        n = 2
        while (self.base / f"{trial_id}__a{n}").exists():
            n += 1
        d2 = self.base / f"{trial_id}__a{n}"
        d2.mkdir(parents=True)
        return d2, n

    def write_json(self, d: Path, name: str, payload: dict) -> None:
        tmp = d / f".{name}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, d / name)

    def write_npz(self, d: Path, name: str, payload: dict) -> None:
        tmp = d / f".{name}.tmp"
        self._writer(tmp, payload)
        os.replace(tmp, d / name)

    @staticmethod
    def _default_npz_writer(path: Path, payload: dict) -> None:
        import torch  # 延迟导入：纯逻辑测试不需要 torch
        torch.save(payload, path)

    def mark_done(self, d: Path) -> None:
        (d / DONE_MARKER).write_text("ok\n", encoding="utf-8")

    def append_record(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.records.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def read_records(self) -> list[dict]:
        if not self.records.is_file():
            return []
        return [json.loads(l) for l in self.records.read_text(encoding="utf-8").splitlines() if l.strip()]
