"""契约测试：预算条件策略与两个评价网络（NAV-029）。

验证四件事：
  1) 维度与尺度严格来自契约（obs-v2 41 维、cmd-v1 3 维、仿射映射 (1.0, 0.4, 1.0)）；
  2) 前向 / 梯度 / 序列化可用且数值自洽；
  3) 三个子网参数互不共享，优化器只能拿到上层参数；
  4) 冻结 v2 底层的权重散列不变（建网不碰底层）。

Run:

    <frozen python> CaMRA/navigation/tests/test_network_contract.py \
        --json CaMRA/navigation/reports/network_contract_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from typing import Any

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NAV_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NAV_ROOT, "src"))

from camra_navigation.algorithm.networks import (  # noqa: E402
    UNIT_TABLE,
    BudgetConditionedNetworks,
    NetworkConfig,
    ObservationLayout,
    PolicyNetwork,
    ValueNetwork,
    build,
    load_contracts,
    normalized_to_command,
)

RESULTS: list[dict[str, Any]] = []


def check(name: str, ok: bool, detail: Any) -> bool:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nav, val = load_contracts()
    cfg = NetworkConfig()
    layout = ObservationLayout.from_contract(nav)

    # ---- 1) 契约一致性 ---------------------------------------------------
    rep = cfg.validate_against(nav)
    check("config_matches_contract", rep["ok"], rep)

    dims_ok = sum(layout.dims) == nav["observation_spec"]["total_dim"] == cfg.obs_dim
    slices = {name: [layout.slice_of(name).start, layout.slice_of(name).stop]
              for name in layout.names}
    contiguous = all(
        slices[layout.names[i]][1] == slices[layout.names[i + 1]][0]
        for i in range(len(layout.names) - 1)
    ) and slices[layout.names[0]][0] == 0
    check("obs_layout_slices", dims_ok and contiguous,
          {"total_dim": layout.total_dim, "contiguous": contiguous,
           "slices": slices,
           "budget_fields": {k: layout.index_of(k)
                             for k in ("z_ratio", "safety_y", "time_remaining_n")}})

    budget = layout.budget_slices()
    maps = layout.map_slices()
    dims_ok2 = ([s.stop - s.start for s in budget.values()] == [1, 1, 1]
                and [s.stop - s.start for s in maps.values()] == [16, 4, 8])
    check("observation_fields_configurable",
          dims_ok2,
          {"budget": {k: [v.start, v.stop] for k, v in budget.items()},
           "map": {k: [v.start, v.stop] for k, v in maps.items()},
           "note": "y/z/time/map 都可按名字取切片，下游做消融或缩放不必硬编码下标"})

    # ---- 2) 前向 ---------------------------------------------------------
    torch.manual_seed(args.seed)
    net: BudgetConditionedNetworks = build(nav, cfg)
    net.eval()
    B = 8
    obs = torch.randn(B, cfg.obs_dim)
    dist = net.policy(obs)
    mean = dist.mean
    std = net.policy.log_std.exp()
    shapes_ok = (tuple(mean.shape) == (B, cfg.action_dim)
                 and tuple(dist.log_prob(mean).shape) == (B,)
                 and tuple(std.shape) == (cfg.action_dim,))
    action, log_prob, _ = net.policy.act(obs)
    check("policy_output_shapes", shapes_ok and tuple(action.shape) == (B, cfg.action_dim),
          {"mean": list(mean.shape), "log_prob": list(log_prob.shape),
           "log_std": list(std.shape), "action": list(action.shape)})

    vc, vh = net.values(obs)
    check("critic_output_shapes",
          tuple(vc.shape) == (B,) and tuple(vh.shape) == (B,),
          {"V_c": list(vc.shape), "V_h": list(vh.shape),
           "init_vc_absmax": float(vc.abs().max()), "init_vh_absmax": float(vh.abs().max())})

    # ---- 3) 动作映射（仿射、非 tanh、低速段精确可达） ---------------------
    a = torch.tensor([[1.0, 1.0, 1.0], [2.0, -2.0, 0.5], [-1.0, 0.0, -0.25]])
    u = normalized_to_command(a, cfg.action_scale)
    expect = torch.tensor([[1.0, 0.4, 1.0], [1.0, -0.4, 0.5], [-1.0, 0.0, -0.25]])
    tanh_ref = torch.tanh(torch.tensor([1.0])) * 1.0
    check("action_mapping_affine",
          torch.allclose(u, expect, atol=1e-7) and abs(float(tanh_ref) - 0.7615942) < 1e-6,
          {"mapping": "u = clip(a,-1,1) * (1.0,0.4,1.0)", "result": u.tolist(),
           "expected": expect.tolist(),
           "why_not_tanh": f"tanh(1.0)={float(tanh_ref):.7f} ≠ 1.0，会压低指令值"})

    low = normalized_to_command(torch.tensor([[0.25, 0.0, 0.0]]), cfg.action_scale)
    check("low_speed_band_exactly_reachable",
          abs(float(low[0, 0]) - 0.25) < 1e-12,
          {"a_vx": 0.25, "u_vx": float(low[0, 0]),
           "note": "NAV-059 要探 0.2-0.31 m/s，仿射映射保证精确"})

    # ---- 4) 分布的采样统计 ----------------------------------------------
    torch.manual_seed(args.seed + 1)
    with torch.no_grad():
        big = net.policy(torch.zeros(20000, cfg.obs_dim))
        samples = big.sample()
    emp_mean = samples.mean(dim=0)
    emp_std = samples.std(dim=0, unbiased=False)
    check("sampling_statistics",
          torch.allclose(emp_mean, big.mean, atol=0.02)
          and torch.allclose(emp_std, net.policy.log_std.exp().expand_as(emp_std), atol=0.02),
          {"n": 20000,
           "mean_err_max": float((emp_mean - big.mean).abs().max()),
           "std_err_max": float((emp_std - net.policy.log_std.exp()).abs().max())})

    # ---- 5) 梯度 ---------------------------------------------------------
    torch.manual_seed(args.seed + 2)
    net2: BudgetConditionedNetworks = build(nav, cfg)
    obs2 = torch.randn(16, cfg.obs_dim)
    act2, lp2, _ = net2.policy.act(obs2)
    loss_pi = -lp2.mean()
    loss_pi.backward()
    pi_grads = [(n, p.grad) for n, p in net2.policy.named_parameters()]
    pi_ok = all(g is not None and torch.isfinite(g).all() for _, g in pi_grads)
    log_std_grad = float(net2.policy.log_std.grad.abs().sum())
    check("policy_gradient_flows", pi_ok and log_std_grad > 0,
          {"n_param_tensors": len(pi_grads), "all_finite": pi_ok,
           "log_std_grad_abs_sum": log_std_grad,
           "critic_untouched": all(p.grad is None for p in net2.critic_cost.parameters())})

    net3: BudgetConditionedNetworks = build(nav, cfg)
    vc3, vh3 = net3.values(torch.randn(16, cfg.obs_dim))
    (vc3.sum() + vh3.sum()).backward()
    c_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in list(net3.critic_cost.parameters()) + list(net3.critic_reach.parameters()))
    check("critic_gradients_flow",
          c_ok and all(p.grad is None for p in net3.policy.parameters()),
          {"both_critics_finite": c_ok,
           "policy_untouched": all(p.grad is None for p in net3.policy.parameters())})

    # ---- 6) 初始化（正交核） -------------------------------------------
    w_hidden = net.policy.body[0].weight.detach()
    gram = w_hidden.t() @ w_hidden
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    hidden_orth_err = float((gram - eye * cfg.hidden_gain ** 2).abs().max())
    head_row_norm = float(net.policy.head.weight.detach().norm(dim=1).mean())
    vhead_row_norm = float(net.critic_cost.head.weight.detach().norm(dim=1).mean())
    check("orthogonal_init",
          hidden_orth_err < 1e-4
          and abs(head_row_norm - cfg.policy_head_gain) < 1e-6
          and abs(vhead_row_norm - cfg.value_head_gain) < 1e-6,
          {"hidden_gram_err": hidden_orth_err,
           "hidden_gain": cfg.hidden_gain,
           "policy_head_row_norm": head_row_norm,
           "value_head_row_norm": vhead_row_norm})

    # ---- 7) 序列化 -------------------------------------------------------
    obs_s = torch.randn(4, cfg.obs_dim)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        path = fh.name
    torch.save(net.state_dict(), path)
    reloaded: BudgetConditionedNetworks = build(nav, cfg)
    try:
        sd = torch.load(path, weights_only=True)
    except TypeError:                     # 旧版 torch 没有 weights_only
        sd = torch.load(path)
    reloaded.load_state_dict(sd)
    reloaded.eval()
    net.eval()
    with torch.no_grad():
        d0 = net.policy(obs_s).mean
        d1 = reloaded.policy(obs_s).mean
        v0 = net.values(obs_s)
        v1 = reloaded.values(obs_s)
    rt_err = float((d0 - d1).abs().max())
    rt_err_v = max(float((v0[0] - v1[0]).abs().max()), float((v0[1] - v1[1]).abs().max()))
    os.unlink(path)
    check("serialization_roundtrip", rt_err == 0.0 and rt_err_v == 0.0,
          {"policy_mean_max_abs_diff": rt_err, "critic_max_abs_diff": rt_err_v,
           "file": "state_dict round-trip via torch.save/load"})

    # ---- 8) 参数隔离与优化器 ---------------------------------------------
    groups = net.param_groups()
    ids = {role: {id(p) for p in ps} for role, ps in groups.items()}
    disjoint = (not (ids["policy"] & ids["critic_cost"])
                and not (ids["policy"] & ids["critic_reach"])
                and not (ids["critic_cost"] & ids["critic_reach"]))
    frozen = torch.nn.Linear(4, 4)
    for p in frozen.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(
        [{"params": groups["policy"], "lr": 3e-4},
         {"params": groups["critic_cost"], "lr": 1e-3},
         {"params": groups["critic_reach"], "lr": 1e-3}])
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    all_net_ids = {id(p) for p in net.parameters()}
    check("optimizer_only_upper_params",
          disjoint and opt_ids == all_net_ids
          and not (opt_ids & {id(p) for p in frozen.parameters()}),
          {"roles_disjoint": disjoint,
           "n_params_by_role": net.n_parameters(),
           "optimizer_param_count": len(opt_ids),
           "network_param_count": len(all_net_ids),
           "frozen_module_excluded": not (opt_ids & {id(p) for p in frozen.parameters()})})

    # ---- 9) 冻结底层散列不变 --------------------------------------------
    # 字段语义（实测澄清）：契约 controller.checkpoint_sha256 指**训练 checkpoint**
    # model/model_2999.pt；实际被执行的是 exported/policy.pt（其散列登记在
    # frozen_manifest.json 的 files 表里）。两者不同物，必须各自核对。
    ctrl = nav["controller"]
    bundle = ctrl["bundle"]
    policy_path = os.path.join(bundle, ctrl["policy_entry"].split(" ")[0])
    manifest = json.load(open(os.path.join(bundle, "frozen_manifest.json"), encoding="utf-8"))
    files = manifest["integrity"]["files"]
    rel_policy = os.path.relpath(policy_path, bundle)
    declared_policy = files[rel_policy]["sha256"]

    before = sha256_file(policy_path)
    _ = build(nav, cfg)                      # 建网不应触碰底层文件
    after = sha256_file(policy_path)
    check("frozen_v2_policy_hash_stable",
          before == after == declared_policy,
          {"policy_pt": policy_path, "sha256_before": before, "sha256_after": after,
           "declared_in_frozen_manifest": declared_policy})

    # training.checkpoint 是训练机上的绝对路径；包内副本按 basename 在 integrity 表里定位
    ckpt_name = manifest["training"]["checkpoint_name"]
    cand = [rel for rel in files if os.path.basename(rel) == ckpt_name]
    if not cand:
        raise SystemExit(f"包内找不到 checkpoint {ckpt_name}")
    ckpt_rel = cand[0]
    ckpt_sha = sha256_file(os.path.join(bundle, ckpt_rel))
    check("checkpoint_field_semantics",
          ckpt_sha == ctrl["checkpoint_sha256"],
          {"contract_field": "controller.checkpoint_sha256",
           "resolves_to": ckpt_rel, "sha256": ckpt_sha,
           "contract_value": ctrl["checkpoint_sha256"],
           "exported_policy_is_a_different_file": declared_policy,
           "note": "契约字段点名的是训练 checkpoint；执行用的是 exported/policy.pt，"
                   "两者散列不同属正常，不能混用"})

    verified, mismatched, skipped = 0, [], []
    for rel, meta in files.items():
        path = os.path.join(bundle, rel)
        if not os.path.exists(path):
            mismatched.append({"file": rel, "reason": "missing"})
            continue
        if os.path.getsize(path) > 50 * 1024 * 1024:
            skipped.append(rel)
            continue
        if sha256_file(path) == meta["sha256"]:
            verified += 1
        else:
            mismatched.append({"file": rel, "reason": "hash_mismatch"})
    check("frozen_bundle_manifest_matches_disk",
          not mismatched,
          {"n_entries": len(files), "n_verified": verified,
           "n_mismatched": len(mismatched), "mismatched": mismatched[:5],
           "skipped_gt_50mb": skipped})

    # ---- 10) 初始化可复现 ------------------------------------------------
    torch.manual_seed(1234)
    a1 = build(nav, cfg)
    torch.manual_seed(1234)
    a2 = build(nav, cfg)
    same = all(torch.equal(p1, p2) for p1, p2 in zip(a1.parameters(), a2.parameters()))
    diff_seed = False
    torch.manual_seed(4321)
    a3 = build(nav, cfg)
    diff_seed = any(not torch.equal(p1, p3) for p1, p3 in zip(a1.parameters(), a3.parameters()))
    check("deterministic_init", same and diff_seed,
          {"same_seed_identical": same, "different_seed_differs": diff_seed})

    # ---- 11) 单位说明与依赖 ---------------------------------------------
    units_ok = all(k in UNIT_TABLE for k in ("action_normalized", "command", "V_c", "V_h")) \
        and all("unit" in UNIT_TABLE[k] for k in UNIT_TABLE)
    check("units_declared", units_ok, {"unit_table": UNIT_TABLE})

    import importlib
    mods_before = set(sys.modules)
    importlib.reload(sys.modules["camra_navigation.algorithm.networks"])
    jax_loaded = any(m == "jax" or m.startswith("jax.") for m in sys.modules)
    check("no_jax_dependency", not jax_loaded,
          {"note": "参考实现是 JAX/Flax；本仓库上层框架 NAV-008 已冻结为 PyTorch",
           "jax_in_sys_modules": jax_loaded,
           "new_modules": len(set(sys.modules) - mods_before)})

    # ---- 12) 契约中的尺度只用不造 ---------------------------------------
    z_ref_contract = float(val["quantities"]["Z_REF_J"]["value"])
    z_ref_obs = float(nav["observation_spec"]["constants"]["z_ref_J"])
    check("z_ref_single_source", abs(z_ref_contract - z_ref_obs) < 1e-12,
          {"value_contract": z_ref_contract, "obs_constants": z_ref_obs,
           "note": "网络不重算 Z_REF，只引用契约；训练与观测共用同一常数"})

    failed = [r for r in RESULTS if not r["ok"]]
    summary = {
        "task": "NAV-029",
        "n_checks": len(RESULTS),
        "n_failed": len(failed),
        "verdict": "PASS" if not failed else "FAIL",
        "network_config": {
            "obs_dim": cfg.obs_dim, "action_dim": cfg.action_dim,
            "hidden": list(cfg.hidden), "activation": cfg.activation,
            "log_std_init": cfg.log_std_init,
            "hidden_gain": cfg.hidden_gain,
            "policy_head_gain": cfg.policy_head_gain,
            "value_head_gain": cfg.value_head_gain,
            "action_scale": list(cfg.action_scale),
        },
        "reference": {
            "path": "RC-PPO/model/actorcritic.py",
            "policy": "Dense256(sqrt2)->tanh->Dense256(sqrt2)->tanh->Dense(A,0.01)+log_std",
            "value": "Dense256(sqrt2)->tanh->Dense256(sqrt2)->tanh->Dense(1,1.0)",
            "port": "PyTorch（框架由 NAV-008 冻结；参考实现为 JAX/Flax，只读对照）",
        },
        "checks": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    head = {k: v for k, v in summary.items() if k != "checks"}
    print(json.dumps(head, ensure_ascii=False))
    for item in failed:
        print("FAILED:", item["check"], json.dumps(item["detail"], ensure_ascii=False)[:400])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
