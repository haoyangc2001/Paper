#!/usr/bin/env python
"""可恢复的短训练循环（NAV-032）。

这不是"训练出效果"的脚本，是**记录与恢复机制的载体**：一轮短训练，图上定点，
初始位姿抖动提供回合多样性，每回合一个 chunk，回合结束即时更新并落检查点。

为什么在名义环境上跑：本步要证的是"样本/模型/优化器/RNG 在中断-恢复后连续、
且不重复写入"，这需要**逐位可复现**。Isaac 那边的物理步进跨进程不具备这个性质，
放到后面 NAV-035 的整链训练里再验。名义环境是冻结的（NAV-019 已独立验过），
10 Hz 导航时钟、成本按 applied 指令计，正好够用。

三个必须成立的性质，对应本脚本的三处结构：

  1. **不把复位后的新状态当旧回合的 bootstrap**：``_step`` 里先取 ``next_obs``
     （复位前），再判 ``rec.done``，最后才 ``reset``。复位只发生在
     ``_begin_episode``，且那时新回合的样本尚未产生。
  2. **不重复写入**：chunk 序号 = manifest 已有条数，``global_step`` 必须接续；
     崩溃后重放只允许"内容完全相同"的幂等提交（见 ``RolloutStore.write_chunk``）。
  3. **中断-恢复等价于连续跑**：状态文件存模型/三优化器/两个 RNG/半截回合/环境快照；
     若状态文件落后于已提交的 chunk（崩在写盘与存点之间），启动时**重放**这些
     chunk 的确定性更新，而不是重跑回合。

Run::

    <frozen python> CaMRA/navigation/scripts/train_nominal.py \\
        --run-id nav032_short --episodes 4 --json /tmp/nav032.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(NAV_ROOT))
CALIB = os.path.join(REPO, "CaMRA", "calibration")
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))
sys.path.insert(0, os.path.join(CALIB, "src"))

from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402
from camra_navigation.algorithm import storage as S  # noqa: E402
from camra_navigation.costs import JCostModel, Z_REF_J  # noqa: E402
from camra_navigation.envs.nominal import NominalNavEnv  # noqa: E402
from camra_navigation.maps import generator as M  # noqa: E402
from camra_navigation.maps.geometry import SafetyGeometry  # noqa: E402
from camra_navigation.runs import default_work_root  # noqa: E402

MAP_DESIGN = os.path.join(NAV_ROOT, "config", "map_design_v1.json")
GEOMETRY_CONTRACT = os.path.join(NAV_ROOT, "config", "geometry_contract_v1.json")
NAV_CONTRACT = os.path.join(NAV_ROOT, "config", "navigation_contract_v1.json")
VALUE_CONTRACT = os.path.join(NAV_ROOT, "config", "value_objective_contract_v1.json")
FOOTPRINT = os.path.join(CALIB, "config", "footprint_v1.json")
MARGINS = os.path.join(CALIB, "config", "margins_v1.json")
POWER = os.path.join(CALIB, "config", "power_model_v1.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_env(level: str, seed: int, z0_J: float, jitter_m: float) -> NominalNavEnv:
    design = M.load_design(MAP_DESIGN)
    spec = M.build(level, seed, design)
    geom = SafetyGeometry.from_frozen(FOOTPRINT, MARGINS, GEOMETRY_CONTRACT)
    model = JCostModel.from_frozen(POWER, MARGINS, variant="full")
    return NominalNavEnv(spec, geom, model, z0_J=z0_J,
                         initial_pose_jitter_m=jitter_m)


def batch_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """把落盘的数组拼成更新器要的 batch（唯一映射点，离线复算也走这里）。"""
    def t(key: str) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arrays[key]), dtype=torch.float32)

    return {
        "obs": t("obs"),
        "action": t("action_raw"),
        "log_prob_old": t("log_prob"),
        "adv_total": t("obj_adv_total"),
        "value_cost_old": t("value_cost"),
        "target_cost": t("obj_target_cost"),
        "value_reach_old": t("value_reach"),
        "target_reach": t("obj_target_reach"),
    }


class Trainer:
    """一轮短训练的全部状态都在这个对象里（``state_payload`` 逐字段存点）。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # 网络初始化要靠 torch RNG（正交初始化），不固定种子的话每次进程启动
        # 都会得到不同初值——那样"中断-恢复 vs 连续"根本不可比。
        torch.manual_seed(int(args.seed_base))
        self.run_root = os.path.abspath(args.run_root)
        self.run_dir = os.path.join(self.run_root, args.run_id)
        self.ckpt_path = os.path.join(self.run_dir, "state.pt")
        self.log_path = os.path.join(self.run_dir, "train_log.jsonl")
        os.makedirs(self.run_dir, exist_ok=True)

        nav_contract = json.load(open(NAV_CONTRACT, encoding="utf-8"))
        self.obs_dim = int(nav_contract["observation_spec"]["total_dim"])

        self.env = build_env(args.level, args.map_seed, args.z0_j, args.jitter_m)
        self.net = NW.build(nav_contract)
        self.ppo_cfg = P.PPOConfig.from_contract()
        self.obj_cfg = S.ObjectiveConfig.from_contracts(VALUE_CONTRACT, r_goal_m=0.5)
        if abs(self.obj_cfg.Z_REF_J - Z_REF_J) > 1e-9:
            raise SystemExit(
                f"契约 Z_REF_J={self.obj_cfg.Z_REF_J} 与 costs.Z_REF_J={Z_REF_J} 不一致")
        if not (0.0 < args.z0_j <= self.obj_cfg.Z_REF_J):
            raise SystemExit(
                f"z0_J={args.z0_j} 必须落在 (0, Z_REF_J={self.obj_cfg.Z_REF_J}]，"
                "否则 z_ratio 在初始时刻就饱和，W 定义在截断量上不再可加")
        self.updater = P.MinimizationUpdater(
            self.net, self.ppo_cfg, lr_policy=args.lr_policy, lr_critic=args.lr_critic)

        self.store = S.RolloutStore(self.run_root, args.run_id, obs_dim=self.obs_dim)
        self.ep_seed = np.random.default_rng(args.seed_base)
        self.global_step = 0
        self.episodes_done = 0
        #: 已经"抽过种子"的回合数。恢复时靠它决定重放要不要再消耗一次种子流：
        #: 存点若发生在回合中途，该回合的种子已经抽过，重放就不能再抽一次。
        self.episodes_started = 0
        self.seg: S.Segment | None = None
        self.obs: np.ndarray | None = None
        self.ep_sampler = torch.Generator().manual_seed(0)
        self.last_update: dict[str, float] = {}
        self.log_events: list[dict[str, Any]] = []
        self.resume_info: dict[str, Any] = {"resumed": False}

    # ------------------------------------------------------------ 配置指纹
    def config_fingerprint(self) -> dict[str, Any]:
        """决定"这轮跑的是什么"的全部量；恢复时逐字段比对，不同就拒绝。"""
        return {
            "level": self.args.level,
            "map_seed": int(self.args.map_seed),
            "z0_J": float(self.args.z0_j),
            "jitter_m": float(self.args.jitter_m),
            "seed_base": int(self.args.seed_base),
            "total_episodes": int(self.args.episodes),
            "lr_policy": float(self.args.lr_policy),
            "lr_critic": float(self.args.lr_critic),
            "ppo": self.ppo_cfg.as_dict(),
            "objective": self.obj_cfg.as_dict(),
            "network": {
                "obs_dim": self.net.cfg.obs_dim,
                "action_dim": self.net.cfg.action_dim,
                "hidden": list(self.net.cfg.hidden),
                "action_scale": list(self.net.cfg.action_scale),
            },
            "contracts": {
                "navigation_contract_v1.json": sha256_file(NAV_CONTRACT),
                "value_objective_contract_v1.json": sha256_file(VALUE_CONTRACT),
                "geometry_contract_v1.json": sha256_file(GEOMETRY_CONTRACT),
                "margins_v1.json": sha256_file(MARGINS),
                "power_model_v1.json": sha256_file(POWER),
            },
        }

    def code_fingerprint(self) -> dict[str, str]:
        return {
            "storage.py": sha256_file(S.__file__),
            "train_nominal.py": sha256_file(os.path.abspath(__file__)),
            "ppo.py": sha256_file(P.__file__),
            "networks.py": sha256_file(NW.__file__),
        }

    # ------------------------------------------------------------ 回合
    def _begin_episode(self) -> None:
        seed = int(self.ep_seed.integers(0, 2 ** 31 - 1))
        self.episodes_started += 1
        obs = self.env.reset(seed=seed)
        self.obs = np.asarray(obs, dtype=np.float32)
        # 采样噪声用**本回合独立**的生成器，只由回合种子决定。
        # 不共用全局 RNG：否则"写盘后崩溃"的恢复会少抽掉那一回合的噪声，
        # 后续回合的采样全部错位（这正是第一次跑出来的失败）。
        self.ep_sampler = torch.Generator().manual_seed(seed)
        self.seg = S.Segment(episode_id=self.episodes_done, map_id=self.env.spec.map_id,
                             map_index=0, seed=seed, obs_dim=self.obs_dim)

    def _act(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """在**归一化动作**上的对角高斯采样，噪声来自回合内生成器。"""
        dist = self.net.policy(obs_t)
        mean = dist.base_dist.mean
        std = dist.base_dist.stddev
        noise = torch.randn(mean.shape, generator=self.ep_sampler, dtype=mean.dtype)
        action = mean + std * noise
        return action, dist.log_prob(action)

    def _step(self) -> None:
        assert self.seg is not None and self.obs is not None
        obs_t = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_raw, logp = self._act(obs_t)
            vc, vh = self.net.values(obs_t)
        action = a_raw[0].numpy().astype(np.float32)

        rec = self.env.step(action.tolist())
        # 关键顺序：先取复位前的后继观测，再判 done；复位只在 _begin_episode 里发生
        next_obs = np.asarray(self.env.observation(), dtype=np.float32)
        done = bool(rec.done)
        kind = S.TERMINAL_KIND.get(str(rec.label)) if done else None

        boot_c = boot_h = 0.0
        if done and kind == "truncation":
            with torch.no_grad():
                bc, bh = self.net.values(torch.as_tensor(next_obs).unsqueeze(0))
            boot_c, boot_h = float(bc[0]), float(bh[0])

        self.seg.append(
            global_step=self.global_step,
            episode_id=self.episodes_done,
            t=int(rec.index),
            obs=self.obs,
            next_obs=next_obs,
            action_raw=action,
            action_applied=np.asarray(rec.command.applied, dtype=np.float32),
            log_prob=float(logp[0]),
            value_cost=float(vc[0]),
            value_reach=float(vh[0]),
            cost_j=float(rec.cost.cost_model_J),
            cost_norm=float(rec.cost.cost_model_J) / self.obj_cfg.Z_REF_J,
            z_j=float(rec.z_J),
            z_ratio=float(rec.z_ratio),
            safety_y=float(rec.safety_y),
            goal_distance_m=float(rec.goal_distance_m),
            terminated=bool(kind in ("failure", "success")),
            truncated=bool(kind == "truncation"),
            event_code=S.EVENT_CODES["none" if rec.label is None else str(rec.label)],
            map_index=0,
            bootstrap_value_cost=boot_c,
            bootstrap_value_reach=boot_h,
            action_clipped=bool(rec.command.clipped),
            action_invalid=bool(rec.command.invalid),
        )
        self.global_step += 1
        self.obs = next_obs
        if done:
            self._finish_episode(str(rec.label), str(kind))

    def _finish_episode(self, label: str, kind: str) -> None:
        assert self.seg is not None
        arrays = self.seg.arrays()
        arrays.update(S.compute_objectives(arrays, self.obj_cfg, terminal_kind=kind))
        meta = {
            "chunk_index": self.store.expected_chunk_index(),
            "episode_id": self.seg.episode_id,
            "map_id": self.seg.map_id,
            "map_index": self.seg.map_index,
            "reset_seed": self.seg.seed,
            "event": label,
            "terminal_kind": kind,
            "terminated": kind in ("failure", "success"),
            "truncated": kind == "truncation",
            "episode_cost_j": float(np.asarray(arrays["cost_j"]).sum()),
            "episode_cost_norm": float(np.asarray(arrays["cost_norm"]).sum()),
            "z0_J": float(self.args.z0_j),
            "n_samples": len(self.seg),
        }
        entry = self.store.write_chunk(arrays, meta)

        if self.args.crash_after_chunk is not None and entry["chunk_index"] == self.args.crash_after_chunk:
            # 故障注入：chunk 已提交、更新与存点都还没做，模拟掉电
            raise SystemExit(90)

        stats = self.updater.update(batch_from_arrays(arrays))
        self.last_update = stats
        self._log({
            "kind": "episode",
            "episode_id": self.seg.episode_id,
            "chunk_index": entry["chunk_index"],
            "event": label,
            "terminal_kind": kind,
            "n_samples": entry["n_samples"],
            "global_step_start": entry["global_step_start"],
            "global_step_end": entry["global_step_end"],
            "episode_cost_j": meta["episode_cost_j"],
            "content_sha256": entry["content_sha256"],
            "update": stats,
            "replayed": False,
        })
        self.episodes_done += 1
        self.seg = None
        self.obs = None

    # ------------------------------------------------------------ 更新与日志
    def _apply_update(self, arrays: dict[str, np.ndarray]) -> dict[str, float]:
        stats = self.updater.update(batch_from_arrays(arrays))
        self.last_update = stats
        return stats

    def _log(self, record: dict[str, Any]) -> None:
        S.assert_no_secrets(record)
        line = S.redact(json.dumps(record, ensure_ascii=False, sort_keys=True))
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.log_events.append(record)

    # ------------------------------------------------------------ 存点
    def state_payload(self, reason: str) -> dict[str, Any]:
        payload = {
            "schema": S.SCHEMA_CHECKPOINT,
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "reason": reason,
            "run_id": self.args.run_id,
            "config": self.config_fingerprint(),
            "code": self.code_fingerprint(),
            "model": self.net.state_dict(),
            "optimizers": {role: opt.state_dict() for role, opt in self.updater.optimizers.items()},
            "rng": {
                "torch": S.rng_state(),
                "episode_seed_stream": S.generator_state(self.ep_seed),
            },
            "counters": {
                "global_step": int(self.global_step),
                "episodes_done": int(self.episodes_done),
                "episodes_started": int(self.episodes_started),
                "n_chunks": int(self.store.n_chunks),
            },
            "normalization": S.normalization_block(
                adv_mean=self.last_update.get("adv_mean"),
                adv_std=self.last_update.get("adv_std"),
            ),
            "buffer": {
                "n_chunks": int(self.store.n_chunks),
                "n_samples": int(self.store.n_samples),
                "last_global_step": int(self.store.last_global_step),
                "manifest_sha256": sha256_file(self.store.manifest_path)
                if os.path.exists(self.store.manifest_path) else "",
            },
            "last_update": self.last_update,
            "episode": {
                "in_progress": self.seg is not None,
                "segment": self.seg.to_state() if self.seg is not None else None,
                "env": S.snapshot_env(self.env) if self.seg is not None else None,
                "obs": None if self.obs is None else np.asarray(self.obs, dtype=np.float32),
                "sampler_state": self.ep_sampler.get_state() if self.seg is not None else None,
            },
        }
        S.assert_no_secrets(payload["config"])
        return payload

    def save_checkpoint(self, reason: str) -> str:
        digest = S.atomic_save(self.state_payload(reason), self.ckpt_path)
        self._log({"kind": "checkpoint", "reason": reason, "sha256": digest,
                   "global_step": self.global_step, "n_chunks": self.store.n_chunks})
        return digest

    # ------------------------------------------------------------ 恢复
    def resume(self) -> None:
        ck = S.atomic_load(self.ckpt_path)
        if ck.get("schema") != S.SCHEMA_CHECKPOINT:
            raise SystemExit(f"状态文件 schema 不符：{ck.get('schema')}")
        if ck["config"] != self.config_fingerprint():
            diff = {k: (ck["config"].get(k), self.config_fingerprint().get(k))
                    for k in set(ck["config"]) | set(self.config_fingerprint())
                    if ck["config"].get(k) != self.config_fingerprint().get(k)}
            raise SystemExit(f"配置与状态文件不一致，拒绝恢复（换 run-id 重开）：{diff}")
        code_now = self.code_fingerprint()
        code_changed = {k: (ck["code"].get(k), v) for k, v in code_now.items()
                        if ck["code"].get(k) != v}

        stored_chunks = int(ck["buffer"]["n_chunks"])
        if self.store.n_chunks < stored_chunks:
            raise SystemExit(
                f"rollout 目录只有 {self.store.n_chunks} 个 chunk，少于状态文件记录的 "
                f"{stored_chunks} 个：数据丢失，拒绝恢复")
        if self.store.n_samples < int(ck["buffer"]["n_samples"]):
            raise SystemExit("rollout 样本数少于状态文件记录，拒绝恢复")

        self.net.load_state_dict(ck["model"])
        for role, state in ck["optimizers"].items():
            self.updater.optimizers[role].load_state_dict(state)
        S.restore_rng(ck["rng"]["torch"])
        S.restore_generator(self.ep_seed, ck["rng"]["episode_seed_stream"])
        self.global_step = int(ck["counters"]["global_step"])
        self.episodes_done = int(ck["counters"]["episodes_done"])
        self.episodes_started = int(ck["counters"].get("episodes_started", self.episodes_done))
        self.last_update = dict(ck.get("last_update") or {})

        ep = ck["episode"]
        if ep["in_progress"]:
            self.seg = S.Segment.from_state(ep["segment"], obs_dim=self.obs_dim)
            S.restore_env(self.env, ep["env"])
            self.obs = np.asarray(ep["obs"], dtype=np.float32)
            self.ep_sampler = torch.Generator()
            self.ep_sampler.set_state(ep["sampler_state"])

        # 半截回合可能在存点之后就已经提交了（崩在 chunk 写盘与存点之间）：
        # 这种情况下磁盘上的 chunk 才是事实，半截快照必须丢弃，改走下面的重放。
        if self.seg is not None and self.store.n_chunks > 0:
            last = self.store.entries[-1]
            if int(last["episode_id"]) == self.seg.episode_id:
                self.seg = None
                self.obs = None

        self.resume_info = {
            "resumed": True,
            "checkpoint_reason": ck.get("reason"),
            "code_changed": code_changed,
            "stored_chunks": stored_chunks,
            "chunks_on_disk": int(self.store.n_chunks),
            "replayed_chunks": [],
        }
        self._log({"kind": "resume", "code_changed": bool(code_changed),
                   "stored_chunks": stored_chunks, "chunks_on_disk": self.store.n_chunks,
                   "in_progress_episode": bool(ep["in_progress"])})

        # 崩在"写完 chunk"与"存点"之间：chunk 已提交、更新没做。更新是确定性的，
        # 直接重放已提交的 chunk，比重跑回合更安全（重跑会因为 RNG 前进而写出不同内容）。
        while self.store.n_chunks > self.episodes_done:
            idx = self.episodes_done
            arrays = self.store.load_chunk(idx)
            entry = self.store.entries[idx]
            stats = self._apply_update(arrays)
            self.episodes_done += 1
            self.global_step = int(entry["global_step_end"]) + 1
            eid = int(entry["episode_id"])
            if eid >= self.episodes_started:
                # 该回合的种子还没抽过（存点发生在它开始之前），补抽一次
                self.ep_seed.integers(0, 2 ** 31 - 1)
                self.episodes_started += 1
            self.resume_info["replayed_chunks"].append(idx)
            self._log({
                "kind": "episode",
                "episode_id": int(entry["episode_id"]),
                "chunk_index": idx,
                "event": entry["event"],
                "terminal_kind": entry["terminal_kind"],
                "n_samples": entry["n_samples"],
                "global_step_start": entry["global_step_start"],
                "global_step_end": entry["global_step_end"],
                "episode_cost_j": entry["episode_cost_j"],
                "content_sha256": entry["content_sha256"],
                "update": stats,
                "replayed": True,
            })

    # ------------------------------------------------------------ 主循环
    def run(self) -> str:
        if self.args.resume:
            self.resume()
        else:
            if os.path.exists(self.ckpt_path):
                raise SystemExit(f"run 目录已有状态文件，若确要续跑请加 --resume：{self.ckpt_path}")
            self.save_checkpoint("start")

        while self.episodes_done < self.args.episodes:
            if self.seg is None:
                self._begin_episode()
            self._step()
            if self.args.stop_after_samples and self.global_step >= self.args.stop_after_samples:
                self.save_checkpoint("stop_after_samples")
                return "stopped_after_samples"
        self.save_checkpoint("finished")
        return "finished"

    def report(self, outcome: str) -> dict[str, Any]:
        audit = S.audit_store(self.store)
        return {
            "task": "NAV-032",
            "outcome": outcome,
            "run_id": self.args.run_id,
            "run_dir": self.run_dir,
            "config": self.config_fingerprint(),
            "code": self.code_fingerprint(),
            "counters": {
                "global_step": self.global_step,
                "episodes_done": self.episodes_done,
                "n_chunks": self.store.n_chunks,
                "n_samples": self.store.n_samples,
            },
            "normalization": S.normalization_block(
                adv_mean=self.last_update.get("adv_mean"),
                adv_std=self.last_update.get("adv_std")),
            "last_update": self.last_update,
            "resume_info": self.resume_info,
            "audit": audit,
            "verdict": "PASS" if not audit["problems"] else "FAIL",
        }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="可恢复的短训练循环（NAV-032）")
    ap.add_argument("--run-root", default=str(default_work_root()))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--level", default="L1_empty")
    ap.add_argument("--map-seed", type=int, default=11)
    ap.add_argument("--z0-j", type=float, default=180.0)
    ap.add_argument("--jitter-m", type=float, default=0.05)
    ap.add_argument("--seed-base", type=int, default=9100)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--lr-policy", type=float, default=3e-4)
    ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-samples", type=int, default=0,
                    help="达到该全局样本数就存点并退出（0 表示不中断）")
    ap.add_argument("--crash-after-chunk", type=int, default=None,
                    help="故障注入：写完该 chunk 后立即退出，不更新不存点")
    ap.add_argument("--json", default="")
    return ap


def main() -> int:
    args = build_argparser().parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)

    trainer = Trainer(args)
    outcome = trainer.run()
    rep = trainer.report(outcome)
    S.assert_no_secrets(rep)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    head = {k: rep[k] for k in ("task", "outcome", "verdict", "counters", "run_dir")}
    print(json.dumps(head, ensure_ascii=False))
    if rep["audit"]["problems"]:
        for p in rep["audit"]["problems"]:
            print("PROBLEM:", p)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
