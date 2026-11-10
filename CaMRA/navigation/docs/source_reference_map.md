# RC-PPO 原算法追踪（NAV-003）

只读追踪 `RC-PPO/`（冻结参考副本，未修改）。基线：主仓库 `8eddd8f`。
所有行号对应当前副本文件。

## 1. 调用图（实际执行路径）

```
EC-EFPPO.train(env, env_params, config, rng)
└─ _train（jax.jit）
   ├─ env.reset  × NUM_ENVS                      → (obs, env_state)
   ├─ jax.lax.scan(_env_step, ..., NUM_STEPS)    → traj_batch
   │   └─ _env_step  (EFPPO_utils.py:5)
   │      ├─ policy.apply → pi ；energy.apply → value ；h.apply → value_h
   │      ├─ action = pi.sample()；log_prob = pi.log_prob(action)
   │      ├─ env.step（vmap）→ obs, env_state, reward, done, info
   │      └─ Transition_reach(done, action, value, value_h, reward,
   │                           last_env_state.energy, log_prob, obs, info,
   │                           last_env_state.reach)
   ├─ last_val = energy(last_obs)；last_val_h = h(last_obs)
   ├─ reach_append  = cat(reach_t,  env_state.reach)          # (T+1, N)
   ├─ V_reach_append= cat(value_h,  last_val_h)               # (T+1, N)
   ├─ energy_append = cat(energy_t, env_state.energy)         # (T+1, N)
   ├─ V_append      = cat(value_t,  last_val)                 # (T+1, N)
   ├─ V_total_append= max(V_reach_append, V_append - energy_append)
   ├─ g_append      = max(reach_append,  -energy_append)
   ├─ done = calculate_indexs3(GAMMA_ENERGY, reward, energy, reach_append,
   │                           last_val, last_val_h)[1][:-1]   # (T, N)
   ├─ advantages_h, targets_h     = calculate_gae_reach4(ent_gamma[1], GAE_LAMBDA, reach_append, V_reach_append, done)
   ├─ advantages_V, targets_V     = calculate_gae2(GAMMA_ENERGY, GAE_LAMBDA, traj_batch, done, last_val)
   ├─ advantages_total, _         = calculate_gae_reach4(GAMMA_REACH_INIT, GAE_LAMBDA, g_append, V_total_append, done)
   └─ jax.lax.scan(update_epoch, ..., UPDATE_EPOCHS)
       └─ _ecefppo_update (EFPPO_utils.py:35)
          ├─ _loss_fn_policy(params, traj_batch, advantages_total)   ← 用 total advantage
          ├─ _loss_fn_energy(params, traj_batch, targets_V)
          └─ _loss_fn_reach (params, traj_batch, targets_h)
```

## 2. 张量形状（单环境视角）

| 量 | 形状 | 说明 |
|---|---|---|
| obs | `(O,)` | WindField：`14`；= `cat(state, [avoid, energy])`，**energy 在最后一维** |
| action | `(3,)` | 连续；经 `tanh` 后使用 |
| value / value_h | `(1,)` | 两个独立 critic，各自单标量输出 |
| reach_append / g_append | `(T+1, N)` | 以 env 数 N 并行 |
| done | `(T, N)` | **推导量**，非环境返回 |

## 3. 价值单位与符号（最易错处）

1. **`reward` 实际是"成本"**。`wind_field.step_env` 返回 `jnp.sum(control_real_1**2)/2`，是**正的成本**，不是奖励；`energy` 同步扣减该值。变量名叫 reward 属历史命名。
2. **策略损失取 `max` 而非 `min`**（`EFPPO_utils.py`）：
   `loss_actor = mean(max(ratio*gae, clip(ratio)*gae))`，随后 `jax.value_and_grad` + `apply_gradients` 做**梯度下降**。
   这与标准 PPO"取 min、做梯度上升"等价，但**符号不能照抄**：改成最小化后若仍写 `min`，裁剪方向会反。
3. **GAE 的 discount 是成本折扣**。`calculate_gae2` 用 `GAMMA_ENERGY`；`calculate_gae_reach4` 用 `ent_gamma[1]` / `GAMMA_REACH_INIT`。`<1` 的成本折扣 **不等于**"未折扣累计机械功"。CaMRA 的目标是未折扣 J，必须逐项核对（计划 `NAV-056`）。
4. **reachability 的 Bellman 是下确界**：`calculate_gae_reach4` 内
   `disc_to_h = (1-gamma)*hs + gamma*(next_Vhs + done*inf)`，`Vhs = min(hs, disc_to_h)`。
   `done` 处用 `inf` 截断，保证不再延伸。
5. **双评价网络共享停止时刻**：`V_total = max(V_h, V_cost − z)` 由同一 `done` 与同一 `g=max(reach,-z)` 计算。不能让两个 critic 各自选停止时刻。

## 4. `done` 不是环境终止

`calculate_indexs3` 在 `(T+1)` 长度上倒序扫描，构造 `V_total` 并取 **`argmin`**，把该位置标为 `done=1`：

```
index_1 = argmin(V_total_1)      # 选"价值最小"的终端时刻
done[index_1, env] = 1.0
```

因此存在**三种不同的"结束"**，实现时必须分开命名：

| 名称 | 来源 | 含义 |
|---|---|---|
| `env_done` | `env.step` 返回 | 时间截断 `time >= max_steps`（`wind_field.is_terminal`） |
| `avoid` | 环境状态 | 进入危险集后**不可逆**（`avoid` 一旦为 -1 保持，状态冻结） |
| `done`（停止分段） | `calculate_indexs3` | 训练用的**最优停止时刻**，对应 MCRA 里 `min over T` |

把 `done` 当成 `env_done` 使用会同时破坏 GAE 截断与停止语义。

## 5. 动作语义差异（CaMRA 必须转换）

原实现（`wind_field.step_env`）：

```
control      = tanh(action)                 # (-1, 1)
goal         = control * 1.5 + state[0:3]   # 位置目标偏移（米）
control_real = u_ref(state, goal)           # 由位置目标反解出的真实控制
cost         = sum(control_real**2)/2
```

即：**网络输出是"位置目标偏移量"，经 tanh 有界**。

而 CaMRA 的上层输出是**机身系速度** `[vx, vy, wz]`（计划 `runtime_contract`）。二者不是同一空间：

- 不能把速度直接乘 1.5 当位置目标；
- 需要在契约里明确"速度 × 0.1 s = 位移"的积分关系，并保证限幅后的**同一实际命令**同时用于运动、成本、底层观测与日志。

## 6. 预算归一化：`[-1,1]` 不是焦耳

`root_finding.Bisection`：

```
init_state = self.init_state(-1., 1.)        # 搜索区间固定 [-1, 1]
obsv = obsv.at[:, -1].set(x)                 # 把 x 写进观测最后一维（预算位）
y = max(V_energy(obs) - (x*variance + mean),
        V_h(obs) + threshold)                # x*variance+mean = 反归一化
```

配合 `env_list.get_env` 的 `transform_observation(obs-mean)/variance`（WindField 预算位 `mean=0, variance=400`）可知：

- `x ∈ [-1,1]` 是**归一化预算**，对应物理预算 `x*400`（原任务单位，非焦耳）；
- CaMRA 必须定义 `z_J`、`z_ref_J` 与归一化映射，并做逆变换测试；
- 二分区间不能直接当 J 区间使用。

## 7. 迁移检查表（NAV-029–031 用）

| # | 检查点 | 原实现 | CaMRA 要求 | 风险 |
|---|---|---|---|---|
| 1 | 目标函数 | `max(V_h, V_cost − z)` | 同 | 符号写反 |
| 2 | 策略损失 | `max(ratio·A, clip(ratio)·A)` + 梯度下降 | 等价最小化形式 | 照抄成 min+上升则方向反 |
| 3 | 优势归一化 | `(gae - mean)/(std+1e-8)` | 同 | 忘记则尺度漂移 |
| 4 | 熵项符号 | `total = actor_loss − ent*entropy` | 同（最小化下减熵） | 写成加熵 |
| 5 | 动作变换 | `tanh(a)` → 位置偏移 ×1.5 | 速度指令 + 限幅 | 空间不同，需重构 |
| 6 | log_prob | 对原始 `action` 采样后计入 | 同 | 在变换后空间取 log_prob 会导致比不一致 |
| 7 | `done` 语义 | 推导的最优停止 | 同，且与 `env_done` 分开命名 | 混用 |
| 8 | 预算单位 | 归一化 `[-1,1]`（×400） | 焦耳 + 固定 `z_ref` | 直接把 J 塞进 [-1,1] |
| 9 | 成本折扣 | `GAMMA_ENERGY < 1` | 未折扣累计 J | 把折扣值当总成本 |
| 10 | 双 critic 停止 | 共享 `done` 与 `g` | 同 | 各自 argmin |
| 11 | 二分搜索 | 无解时仍取中点 | 需端点/单调性/无解检查 | 输出伪可行预算 |

## 8. 原副本已知缺陷（只记录，不改写）

- 绝对导入：`sys.path.append("/home/mepear_gc")`，模块路径为 `EFPPO.src.rl.*`，与当前目录布局不一致，直接运行会 ImportError。
- 依赖 `plot_utils`（`calculate_minimal_reach`、`calculate_consumption` 等），该文件在只读副本中**不存在**（已 find 确认），故 EC-EFPPO 无法直接运行。
- `reward` 命名与"成本"语义相反（见 §3.1）。
- `Bisection.run()` 在 `sign==0`（无括根）时仍返回区间中点，不报告不可行。

以上均属参考副本的既有状态，不修改 `RC-PPO/`。

---

# HiRA 分层工程结构（NAV-005）

只读追踪 `HiRA/`（锁定子模块 `6a05af5`）。**仅借鉴工程组织**，不加载其控制器，不复用 Isaac Gym 运行环境，不采用其单价值 ReachAvoidPPO 作为 CaMRA 算法。

## 1. 真实入口链

```
scripts/train_reach_avoid.py
├─ HierarchicalGO2Env(cfg, low_level_model_path=train_cfg.runner.low_level_model_path, args)
│   ├─ _create_base_env()          → task_registry.make_env("go2")            # 底层 GO2Robot
│   ├─ _load_low_level_policy()    → OnPolicyRunner(...).load(path) → get_inference_policy()
│   └─ HighLevelNavigationEnv(base_env, HighLevelNavigationConfig())
└─ ReachAvoidPPO(...)              # rsl_rl/algorithms/reach_avoid_ppo.py
   └─ 循环：actions → env.step(actions) → (obs, g_vals, h_vals, dones, infos)
```

## 2. 分层 `step` 的实现模式（可直接借鉴）

`HierarchicalGO2Env.step(high_level_actions)`（`hierarchical_go2_env.py`）做四件事：

```python
# 1) 写入新指令
self.high_level_env.update_velocity_commands(high_level_actions)
desired = self.base_env.commands[:, :3].clone()

# 2) 重复调用低层，每次都用"新鲜"观测重新推理
for _ in range(self.low_level_action_repeat):
    self.base_env.commands[:, :3] = desired        # 每个子步重新写回，防被内部覆盖
    self.base_env.compute_observations()           # 重新观测（含新指令）
    obs = self.base_env.get_observations()
    low_actions = self.low_level_policy(obs)       # 重新推理
    base_obs, _, _, step_dones, base_infos, avoid, reach = self.base_env.step(low_actions)
    aggregated_dones |= step_dones.bool()          # 子步事件用 OR 累计

# 3) 重算高层观测  4) g/h
self.high_level_env._compute_high_level_observations()
g_values, h_values = self.high_level_env.compute_g_h_values(avoid, reach)
```

三个值得照搬的设计：

1. **指令每个子步都重写**，因为底层 `step` 可能自行改写 `commands`；不重写会出现"高层下了新指令、底层还在跑旧指令"。
2. **每个子步都重算底层观测并重新推理**，而不是复用同一个关节动作。这正是计划 `global_constraints[4]` 要求的"不能把一次关节输出保持 0.1 秒"。
3. **子步事件用 `|=` 累计**，避免只看最后一个子步导致中间摔倒/越界被吞掉。

## 3. 观测与动作维度

| 项 | 组成 |
|---|---|
| 高层观测 | `num_high_level_obs = base_obs_dim + target_feature_dim + lidar_feature_dim` |
| 高层动作 | `3`：`[vx, vy, vyaw]` |
| 速度映射 | `clip(a, -1, 1) * action_scale`，再分别乘 `0.6 / 0.2 / 0.8` 后写入 `base_env.commands[:, :3]` |

**注意**：HiRA 的实际速度域约为 `±0.6 / ±0.2 / ±0.8`，与 v2 的 `±1.0 / ±0.4 / ±1.0` **不同**。不能沿用它的缩放系数，CaMRA 必须按锁定协议的命令域重写 `update_velocity_commands` 等价函数。

## 4. 需要改写的部分（Isaac Gym → Isaac Lab）

| HiRA 做法 | CaMRA 需要 |
|---|---|
| `task_registry.make_env("go2")`（legged_gym / Isaac Gym） | Isaac Lab 的 `DirectRLEnv` / `ManagerBasedRLEnv` + manager 配置 |
| `OnPolicyRunner` 加载 `.pt` 推理 | v2 冻结包的 JIT/ONNX 或原始 `model_2999.pt`，按 `contract.observation` 构造 45 维 |
| `base_env.compute_observations()` | Isaac Lab 的 observation manager / 显式张量拼接 |
| `base_env.commands` 直写 | 外部命令桥（10 Hz，零阶保持），需要 `external_command.py` 等价实现 |
| `base_env.step(low_actions)` 内部自动 reset | Isaac Lab 的 `step` 语义不同，需显式处理 truncate/terminate 与 reset 隔离 |
| 单价值 `ReachAvoidPPO` | CaMRA 的双评价 + 成本预算（见 `source_reference_map.md` 前文 RC-PPO 部分） |

## 5. 已发现的缺口（不可照搬）

`HierarchicalGO2Env.step` 的循环里**没有自动复位隔离**：

- 若某个 env 在子步中 `step_dones=True`，legged_gym 会在 `base_env.step` 内部**立刻复位该 env**；
- 循环继续跑剩余子步时，用的是**已复位的新回合**状态；
- `aggregated_dones` 虽然把 `True` 记录下来，但后续子步的 `avoid/reach` 已经属于新回合；
- 结果是"旧回合的事件 + 新回合的指标"被拼在一起。

CaMRA 实现必须在以下位置做隔离（对应计划 `global_constraints[7]` 与 `NAV-026`）：

1. 子步循环内检测到 `done` 的 env，**立即冻结**其后续子步的指标写入，只保留本回合已发生的事件；
2. 高层 `step` 返回前，对已 done 的 env 使用**回合结束时刻**的快照计算 `g/h`，而不是循环结束时的状态；
3. 记日志时把 `env_done`、`avoid`、停止分段三者分开命名与落盘。

## 6. 可复用与不可复用清单

| 可借鉴 | 不可复用 |
|---|---|
| 分层 `step` 的四步结构 | Isaac Gym / legged_gym 运行环境 |
| 每子步重写指令 + 重算观测 + 重新推理 | HiRA 的低层权重（必须用 v2 的 `model_2999.pt`） |
| 子步事件 OR 累计的写法 | HiRA 的速度缩放（0.6/0.2/0.8 ≠ v2 的 1.0/0.4/1.0） |
| `g/h` 的计算与日志分离 | 单价值 `ReachAvoidPPO`（CaMRA 用双 critic + 成本预算） |
| 训练脚本的参数/检查点记录方式 | 雷达/边界特征（首轮用仿真真值，不引入感知） |
