"""NAV-064 策略层 X 光（只读）。

问题：第 11 轮里策略输出层「前进」那一行的权重范数 0.057→0.204（3.5 倍），
策略从「几乎不响应观测」变成「强响应观测」，同期没盖章段的优势幅度掉 81%、
离终点 9.96→12.70 m。要查清是什么在推这层权重、推的方向对不对。

做法（不训练、不改任何现有文件）：
  1) 取每个检查点之后紧接的窗口样本（由 θ_ck 自己采样得到），在 θ_ck 上重算
     策略损失，把梯度拆成「代理项（优势）」与「熵项」，量范数、夹角与落点。
     熵项只作用于 log_std，理论上对均值参数梯度恒为 0 —— 作为实现自检。
  2) 按总价值取哪一支（V_h 还是 V_c−z）把样本分组，量两支各推多少、是否对抗。
  3) 量信号质量：优势与「这一步是否真的更靠近终点」的相关性。
  4) 量动作统计（前进速度、转向幅度）随训练怎么变。

输出：reports/training_round_r11_policy_xray.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

NAV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(NAV, "src"))
sys.path.insert(0, os.path.join(NAV, "..", "calibration", "src"))

from camra_navigation.algorithm import networks as NW  # noqa: E402
from camra_navigation.algorithm import ppo as P  # noqa: E402

WORK = "/pub/data/caohy/work/camra_navigation"
RUNS = ["nav064_r11_uncalibrated_s0", "nav064_r11_full_s0"]
VAR = {"nav064_r11_uncalibrated_s0": "uncalibrated",
       "nav064_r11_full_s0": "full"}
CKPTS = [50, 150, 300, 450, 550]
PREREG = os.path.join(NAV, "config", "training_loop_prereg_r11.json")

FIELDS = ("global_step", "obs", "action_raw", "log_prob", "obj_adv_total",
          "obj_adv_reach", "obj_adv_cost", "value_cost", "value_reach",
          "z_norm", "goal_distance_m", "goal_distance_m_next", "safety_y",
          "map_index", "obj_h", "obj_g_train", "episode_id", "t",
          "cost_norm", "z_norm_next")


def build_net() -> NW.BudgetConditionedNetworks:
    """按 r11 预登记的开关重建网络（与训练/评估同一口径）。"""
    with open(PREREG, encoding="utf-8") as fh:
        pre = json.load(fh)
    fix = pre.get("action_init_fix", {})
    on = bool(fix.get("enabled", False))
    steer = pre.get("steering_prior", {})
    gain = float(steer.get("gain", steer.get("steer_gain", 0.0)) or 0.0)
    return NW.build(
        action_bounded=bool(pre.get("action_bounded", False)),
        initial_forward_speed=float(fix.get("initial_forward_speed", 0.0)) if on else 0.0,
        log_std_min=float(fix["log_std_min"]) if on and "log_std_min" in fix else None,
        log_std_init=float(fix["log_std_init"]) if on and "log_std_init" in fix else None,
        steering_gain=gain)


def policy_params(net) -> tuple[list, dict]:
    """显式给出参数顺序与切片，不依赖 Module.parameters() 的注册顺序推断。"""
    body = list(net.policy.body.parameters())
    head_w = net.policy.head.weight
    head_b = net.policy.head.bias
    log_std = net.policy.log_std
    params = body + [head_w, head_b, log_std]
    sizes = [p.numel() for p in params]
    idx, start = {}, 0
    for p, s in zip(params, sizes):
        idx[id(p)] = (start, start + s)
        start += s
    return params, {"head_w": idx[id(head_w)], "head_b": idx[id(head_b)],
                    "log_std": idx[id(log_std)], "total": start}


def load_windows(run_id: str, windows: list[tuple[int, int]], cum: np.ndarray,
                 per_window_chunks: int = 160) -> dict:
    """单遍扫描 + 等步长抽样 + 磁盘缓存。

    全量 2 万多个分块、每个 npz 都要解压，单跑一遍要 8 分钟；这里按等步长抽到
    每个窗口约 160 个分块（约 3.6 万步），足够稳定估计梯度夹角与相关系数。
    抽样在每个环境内部都是等间隔的，不偏向某一段训练。
    """
    cache = f"/tmp/nav064_xray_cache_{run_id}.npz"
    if os.path.exists(cache):
        with np.load(cache) as z:
            return {w: {f: np.asarray(z[f"{i}|{f}"]) for f in FIELDS}
                    for i, w in enumerate(windows)}
    root = os.path.join(WORK, run_id)
    env_dirs = sorted(glob.glob(os.path.join(root, "rollout_env*")))
    n_env = max(1, len(env_dirs))
    cum_env = cum / float(n_env)          # global_step 是**每环境**计数
    files = sorted(glob.glob(os.path.join(root, "rollout_env*", "rollout", "chunk_*.npz")))
    stride = max(1, len(files) // (per_window_chunks * len(windows)))
    files = files[::stride]
    acc = [{f: [] for f in FIELDS} for _ in windows]
    for p in files:
        with np.load(p) as z:
            gs = np.asarray(z["global_step"])
            idx = np.searchsorted(cum_env, gs, side="right") - 1
            for wi, (lo, hi) in enumerate(windows):
                keep = (idx >= lo) & (idx < hi)
                if not keep.any():
                    continue
                for f in FIELDS:
                    acc[wi][f].append(np.asarray(z[f])[keep])
    out, flat = [], {}
    for wi, w in enumerate(windows):
        d = {}
        for f in FIELDS:
            d[f] = (np.concatenate(acc[wi][f]) if acc[wi][f]
                    else np.zeros((0,) + ((41,) if f == "obs" else
                                          (3,) if f == "action_raw" else ())))
            flat[f"{wi}|{f}"] = d[f]
        out.append(d)
    np.savez(cache, **flat)
    print(f"  [{run_id}] n_env={n_env} files={len(files)} stride={stride} "
          f"per_window={[int(d['obs'].shape[0]) for d in out]}", flush=True)
    return {w: d for w, d in zip(windows, out)}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


class Probe:
    """在固定样本上，按需要重算损失并取梯度（每次重算图，避免二次反传）。"""

    def __init__(self, net, obs, act, lp_old, adv, cfg):
        self.net = net
        self.obs = obs
        self.act = act
        self.lp_old = lp_old
        self.adv_n, _ = P.normalize_advantage(adv)
        self.cfg = cfg
        self.params, self.slices = policy_params(net)

    def terms(self, subset: torch.Tensor | None = None):
        """返回 (surrogate 损失张量, 熵项损失张量, ratio)；图是新的，可分别反传。"""
        o = self.obs if subset is None else self.obs[subset]
        a = self.act if subset is None else self.act[subset]
        lp0 = self.lp_old if subset is None else self.lp_old[subset]
        adv_n = self.adv_n if subset is None else self.adv_n[subset]
        dist = self.net.policy(o)
        lp = dist.log_prob(a)
        ratio = torch.exp(lp - lp0)
        lo, hi = 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps
        surrogate = torch.maximum(ratio * adv_n, torch.clamp(ratio, lo, hi) * adv_n)
        return surrogate.mean(), -self.cfg.ent_coef * dist.entropy().mean(), ratio

    def grad(self, which: str, subset: torch.Tensor | None = None) -> np.ndarray:
        loss_s, loss_e, _ = self.terms(subset)
        loss = loss_s if which == "surrogate" else loss_e
        for p in self.params:
            p.grad = None
        loss.backward()
        g = np.concatenate([p.grad.detach().reshape(-1).numpy()
                            if p.grad is not None else np.zeros(p.numel())
                            for p in self.params])
        for p in self.params:
            p.grad = None
        return g



def action_pressure(net, obs, act, lp_old, adv, cfg) -> dict:
    """把损失对「动作均值」求导：更新到底把每个动作维度往哪边推。

    绕开参数层直接对 head 输出求导，单位是「每单位动作均值对应的损失变化」。
    最小化语义下，正号 = 该维度应当调小（压低），负号 = 调大（抬高）。
    """
    adv_n, _ = P.normalize_advantage(adv)
    with torch.no_grad():
        body_out = net.policy.body(obs)
        m0 = net.policy.head(body_out)
        if net.policy.steering_gain != 0.0:
            m0 = m0.clone()
            m0[:, 2] = m0[:, 2] + net.policy.steering_gain * obs[:, net.policy._goal_dir_y_idx]
        std = net.policy.log_std.clamp(net.policy.log_std_min, 1.0).exp().expand_as(m0)
    m = m0.clone().detach().requires_grad_(True)
    dist = NW.SquashedNormal(m, std) if net.policy.action_bounded else \
        torch.distributions.Independent(torch.distributions.Normal(m, std), 1)
    ratio = torch.exp(dist.log_prob(act) - lp_old)
    lo, hi = 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
    surrogate = torch.maximum(ratio * adv_n, torch.clamp(ratio, lo, hi) * adv_n)
    surrogate.mean().backward()
    g = m.grad.detach()
    out = {}
    for i, nm in enumerate(("forward", "lateral", "yaw")):
        gi = g[:, i]
        out[nm] = {"mean_g": round(float(gi.mean()), 6),
                   "frac_push_down": round(float((gi > 0).float().mean()), 4)}
    return out


def analyze(run_id: str, ck: int, d: dict, cfg: P.PPOConfig) -> dict:
    net = build_net()
    st = torch.load(os.path.join(WORK, run_id, f"state_u{ck}.pt"),
                    map_location="cpu", weights_only=False)
    net.load_state_dict(st["model"])
    net.eval()

    n = int(d["obs"].shape[0])
    if n < 200:
        return {"checkpoint": ck, "samples": n, "note": "样本不足"}

    obs = torch.as_tensor(d["obs"], dtype=torch.float32)
    act = torch.as_tensor(d["action_raw"], dtype=torch.float32)
    lp_old = torch.as_tensor(d["log_prob"], dtype=torch.float32)
    adv = torch.as_tensor(d["obj_adv_total"], dtype=torch.float32)
    adv_reach = torch.as_tensor(d["obj_adv_reach"], dtype=torch.float32)
    adv_cost = torch.as_tensor(d["obj_adv_cost"], dtype=torch.float32)
    vc = torch.as_tensor(d["value_cost"], dtype=torch.float32)
    vh = torch.as_tensor(d["value_reach"], dtype=torch.float32)
    z = torch.as_tensor(d["z_norm"], dtype=torch.float32)
    y = torch.as_tensor(d["safety_y"], dtype=torch.float32)
    gd = d["goal_distance_m"]
    gdn = d["goal_distance_m_next"]
    vx = d["action_raw"][:, 0]
    wz = d["action_raw"][:, 2]

    pr = Probe(net, obs, act, lp_old, adv, cfg)
    sl = pr.slices
    loss_s, loss_e, ratio = pr.terms()
    g_s = pr.grad("surrogate")
    g_e = pr.grad("entropy")
    a_ls, b_ls = sl["log_std"]
    a_hw, b_hw = sl["head_w"]
    a_hb, b_hb = sl["head_b"]

    g_s_mean = np.concatenate([g_s[:a_hw], g_s[b_hw:a_hb]])  # body + head.weight
    g_s_mean2 = np.concatenate([g_s[:a_hw], g_s[b_hw:b_hb]])
    g_e_offmean = float(np.linalg.norm(np.concatenate(
        [g_e[:a_ls], g_e[b_ls:]])))

    w = net.policy.head.weight.detach().numpy()
    g_w = g_s[a_hw:b_hw].reshape(w.shape)
    rows = {}
    for i, nm in enumerate(("forward", "lateral", "yaw")):
        rows[nm] = {"w_norm": round(float(np.linalg.norm(w[i])), 6),
                    "g_norm": round(float(np.linalg.norm(g_w[i])), 6),
                    "g_radial": round(cos(g_w[i].reshape(-1), w[i].reshape(-1)), 4)}

    budget_branch = (vc - z) > vh
    branches = {}
    g_r = g_b = None
    if int(budget_branch.sum()) > 100 and int((~budget_branch).sum()) > 100:
        g_r = pr.grad("surrogate", ~budget_branch)
        g_b = pr.grad("surrogate", budget_branch)
        sq = float(np.linalg.norm(g_r)) ** 2 + float(np.linalg.norm(g_b)) ** 2
        branches = {
            "n_reach": int((~budget_branch).sum()),
            "n_budget": int(budget_branch.sum()),
            "share_reach": round(float(np.linalg.norm(g_r)) ** 2 / max(sq, 1e-12), 4),
            "share_budget": round(float(np.linalg.norm(g_b)) ** 2 / max(sq, 1e-12), 4),
            "cos_reach_vs_budget": round(cos(g_r, g_b), 4),
        }

    step_d = gdn - gd
    live = (y <= 0.5).numpy()

    # 回合内相关：去掉「回合进度」这个混杂（预算随时间单调消耗，会让同一回合后段
    # 的优势系统性偏高）。做法是按 episode 去均值后再求相关，等价于固定回合效应。
    # 拼真实回合 id：episode_id 变化或 global_step 不连续（跨分块/跨环境）都算新回合
    eid = d["episode_id"]
    gstep = d["global_step"]
    brk = np.r_[True, (np.diff(eid) != 0) | (np.diff(gstep) != 1)]
    ep = np.cumsum(brk)
    adv_np = adv.numpy()
    reach_np = adv_reach.numpy()

    z_np = d["z_norm"]
    ep_min_z = np.array([z_np[ep == e].min() if (ep == e).any() else 0.0
                         for e in np.unique(ep)])
    ep_ids = np.unique(ep)
    clean_ep = ep_ids[ep_min_z >= 0.0]
    over_ep = ep_ids[ep_min_z < 0.0]
    clean_share = float(np.isin(ep, clean_ep).mean())
    clean_mask = np.isin(ep, clean_ep)
    over_mask = np.isin(ep, over_ep)

    def within_ep_corr(x, yy):
        xs, ys = [], []
        for e in np.unique(ep):
            m = ep == e
            if int(m.sum()) < 5:
                continue
            xs.append(x[m] - x[m].mean())
            ys.append(yy[m] - yy[m].mean())
        if not xs:
            return None
        a2 = np.concatenate(xs)
        b2 = np.concatenate(ys)
        return corr(a2, b2)

    def corr(a, b):
        if a.size < 10 or a.std() < 1e-12 or b.std() < 1e-12:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    def _wcorr(mask):
        xs, ys = [], []
        for e in np.unique(ep[mask]):
            m = (ep == e) & mask
            if int(m.sum()) < 5:
                continue
            xs.append(adv_np[m] - adv_np[m].mean())
            ys.append(-step_d[m] - (-step_d[m]).mean())
        if not xs:
            return None
        return corr(np.concatenate(xs), np.concatenate(ys))

    clean_corr = _wcorr(clean_mask)
    over_corr = _wcorr(over_mask)
    clean_mean = (round(float(adv_np[clean_mask].mean()), 5)
                  if clean_mask.any() else None)

    return {
        "checkpoint": ck, "samples": n,
        "adv": {"mean": round(float(adv.mean()), 5),
                "std": round(float(adv.std()), 5),
                "std_live": round(float(adv[live].std()), 5) if live.any() else None,
                "std_reach": round(float(adv_reach.std()), 5),
                "std_cost": round(float(adv_cost.std()), 5)},
        "loss": {"surrogate": round(float(loss_s), 6),
                 "entropy_term": round(float(loss_e), 6),
                 "clip_frac": round(float(((ratio < 1 - cfg.clip_eps) |
                                           (ratio > 1 + cfg.clip_eps)).float().mean()), 4),
                 "ratio_mean": round(float(ratio.mean()), 5)},
        "grad": {"g_surrogate_norm": round(float(np.linalg.norm(g_s)), 6),
                 "g_entropy_norm": round(float(np.linalg.norm(g_e)), 6),
                 "g_entropy_off_logstd": round(g_e_offmean, 10),
                 "cos_surrogate_entropy": round(cos(g_s, g_e), 4),
                 "g_s_on_mean_params": round(float(np.linalg.norm(g_s_mean2)), 6),
                 "g_s_on_logstd": round(float(np.linalg.norm(g_s[a_ls:b_ls])), 6),
                 "entropy_share": round(float(np.linalg.norm(g_e)) /
                                        max(float(np.linalg.norm(g_s)) +
                                            float(np.linalg.norm(g_e)), 1e-12), 4)},
        "branches": branches,
        "mean_head": {"rows": rows,
                      "g_bias": [round(float(v), 6) for v in g_s[a_hb:b_hb]],
                      "g_bias_ent": [round(float(v), 10) for v in g_e[a_hb:b_hb]],
                      "head_w_fro": round(float(np.linalg.norm(w)), 6),
                      "radial_head": round(cos(g_w.reshape(-1), w.reshape(-1)), 4)},
        "pressure_total": action_pressure(net, obs, act, lp_old, adv, cfg),
        "pressure_reach_only": action_pressure(net, obs, act, lp_old, adv_reach, cfg),
        "log_std": [round(float(v), 4) for v in net.policy.log_std.detach().numpy()],
        "action": {"vx_mean": round(float(vx.mean()), 4),
                   "vx_std": round(float(vx.std()), 4),
                   "wz_mean": round(float(wz.mean()), 4),
                   "wz_absmean": round(float(np.abs(wz).mean()), 4),
                   "wz_std": round(float(wz.std()), 4)},
        "signal": {"corr_adv_vs_ddist": corr(adv.numpy(), step_d),
                   "corr_adv_reach_vs_minus_ddist": corr(adv_reach.numpy(), -step_d),
                   "corr_adv_cost_vs_minus_ddist": corr(adv_cost.numpy(), -step_d),
                   "adv_mean_closing": (round(float(adv[torch.as_tensor(step_d < 0)].mean()), 5)
                                        if bool((step_d < 0).any()) else None),
                   "adv_mean_receding": (round(float(adv[torch.as_tensor(step_d > 0)].mean()), 5)
                                         if bool((step_d > 0).any()) else None),
                   "adv_reach_mean_closing": (round(float(adv_reach[torch.as_tensor(step_d < 0)].mean()), 5)
                                              if bool((step_d < 0).any()) else None),
                   "adv_reach_mean_receding": (round(float(adv_reach[torch.as_tensor(step_d > 0)].mean()), 5)
                                               if bool((step_d > 0).any()) else None),
                   "h_mean": round(float(torch.as_tensor(d["obj_h"]).mean()), 4),
                   "g_train_mean": round(float(torch.as_tensor(d["obj_g_train"]).mean()), 4),
                   "within_ep_corr_adv_spend": within_ep_corr(adv_np, -(d["z_norm_next"] - d["z_norm"])),
                   "corr_adv_vs_cost_norm": corr(adv_np, d["cost_norm"]),
                   "corr_spend_vs_closing": corr(-(d["z_norm_next"] - d["z_norm"]), -step_d),
                   "spend_share_of_adv_var": round(float(
                       np.var(d["cost_norm"]) / max(np.var(adv_np), 1e-12)), 4),
                   "z_neg_frac": round(float((d["z_norm"] < 0).mean()), 4),
                   "corr_adv_vs_h": corr(adv_np, d["obj_h"]),
                   "corr_adv_vs_dist": corr(adv_np, gd),
                   "corr_reach_vs_h": corr(reach_np, d["obj_h"]),
                   "within_ep_corr_adv_h": within_ep_corr(adv_np, d["obj_h"]),
                   "within_ep_corr_reach_h": within_ep_corr(reach_np, d["obj_h"]),
                   "within_ep_corr_adv_t": within_ep_corr(adv_np, d["t"].astype(np.float64)),
                   "clean_ep_share": round(float(clean_share), 4),
                   "clean_within_ep_corr_adv": clean_corr,
                   "clean_adv_mean": clean_mean,
                   "overspent_within_ep_corr_adv": over_corr,
                   "within_ep_corr_adv_minus_ddist": within_ep_corr(adv_np, -step_d),
                   "within_ep_corr_reach_minus_ddist": within_ep_corr(reach_np, -step_d),
                   "within_ep_n_episodes": int(np.unique(ep).size),
                   "corr_adv_vs_minus_ddist": corr(adv.numpy(), -step_d),
                   "corr_adv_vs_minus_ddist_live": (corr(adv[live].numpy(), -step_d[live])
                                                    if live.any() else None),
                   "corr_adv_vs_vx": corr(adv.numpy(), vx),
                   "mean_ddist": round(float(step_d.mean()), 4),
                   "mean_ddist_live": (round(float(step_d[live].mean()), 4)
                                       if live.any() else None)},
        "y_frac": round(float((y > 0.5).float().mean()), 4),
    }


def main() -> int:
    cfg = P.PPOConfig.from_contract(os.path.join(NAV, "config",
                                                 "value_objective_contract_v1.json"))
    out = {"note": "r11 策略层 X 光：梯度分项落点 / 支路对抗 / 信号质量 / 动作统计",
           "prereg": "training_loop_prereg_r11.json", "runs": {}}
    for run_id in RUNS:
        log = os.path.join(WORK, run_id, "train_log.jsonl")
        ups = [json.loads(l) for l in open(log, encoding="utf-8")
               if json.loads(l).get("kind") == "update"]
        cum = np.cumsum([u["n_samples"] for u in ups])
        windows = [(ck, ck + 50) for ck in CKPTS]
        data = load_windows(run_id, windows, cum)
        rows = []
        for ck in CKPTS:
            r = analyze(run_id, ck, data[(ck, ck + 50)], cfg)
            rows.append(r)
            if "grad" in r:
                g, mh, ac = r["grad"], r["mean_head"], r["action"]
                print(f"{VAR[run_id]:12s} u{ck:<4d} n={r['samples']:<6d} "
                      f"|gS|={g['g_surrogate_norm']:.4f} |gE|={g['g_entropy_norm']:.4f} "
                      f"cos={g['cos_surrogate_entropy']:+.3f} "
                      f"hF={mh['head_w_fro']:.4f} gW={mh['rows']['forward']['g_norm']:.4f} "
                      f"rad={mh['rows']['forward']['g_radial']:+.3f} "
                      f"gbias={mh['g_bias']} vx={ac['vx_mean']:.3f} |wz|={ac['wz_absmean']:.3f} "
                      f"c_dd={r['signal']['corr_adv_vs_minus_ddist']} y={r['y_frac']:.2f} "
                      f"a~h={r['signal']['corr_adv_vs_h']}"
                      f"/{r['signal']['within_ep_corr_adv_h']} "
                      f"r~h={r['signal']['corr_reach_vs_h']}"
                      f"/{r['signal']['within_ep_corr_reach_h']} "
                      f"a~t={r['signal']['within_ep_corr_adv_t']} "
                      f"clean={r['signal']['clean_ep_share']}"
                      f" wCclean={r['signal']['clean_within_ep_corr_adv']}"
                      f" wCover={r['signal']['overspent_within_ep_corr_adv']} "
                      f"wS={r['signal']['within_ep_corr_adv_spend']} "
                      f"cCn={r['signal']['corr_adv_vs_cost_norm']} "
                      f"sC={r['signal']['corr_spend_vs_closing']} "
                      f"zN={r['signal']['z_neg_frac']} "
                      f"wC={r['signal']['within_ep_corr_adv_minus_ddist']}"
                      f"/{r['signal']['within_ep_corr_reach_minus_ddist']} "
                      f"P_fwd={r['pressure_total']['forward']['mean_g']:+.4f}"
                      f"/{r['pressure_reach_only']['forward']['mean_g']:+.4f} "
                      f"P_yaw={r['pressure_total']['yaw']['mean_g']:+.4f}"
                      f"/{r['pressure_reach_only']['yaw']['mean_g']:+.4f}")
        out["runs"][run_id] = {"variant": VAR[run_id], "rows": rows}

    p = os.path.join(NAV, "reports", "training_round_r11_policy_xray.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
