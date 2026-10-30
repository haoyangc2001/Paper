# RC-PPO：最小成本到达避障参考实现

本目录保存项目前期采用的最小成本到达避障（minimum-cost reach-avoid）算法实现，使用 JAX 在摆锤、Hopper、HalfCheetah、风场和 F-16 等系统上定义训练任务，供后续四足机器人导航实现学习和对照。

目录沿用 `RC-PPO` 名称，代码中的训练入口和内部命名为 `EC-EFPPO` / `EFPPO`。本地副本没有独立的上游 Git 元数据、原始说明和依赖锁定文件，不能仅凭目录名确认上游版本或与论文的逐项一致性。

## 保留约定

**本目录作为冻结的算法参考副本，后续不得直接调整其中的代码。** 训练算法、网络、环境、参数脚本和资源文件均保留原样；发现问题时先记录，修复、重构及四足机器人适配放在 [CaMRA/navigation](../CaMRA/navigation/README.md) 等实现模块中进行。

底层速度跟踪器放在 [CaMRA/locomotion](../CaMRA/locomotion/README.md)，执行误差校准与检验放在 [CaMRA/calibration](../CaMRA/calibration/README.md)。本目录只承担原实现的阅读、追溯和对照用途。

2026-10-31仅将原 `RC-PPO/src/` 内部目录上移一级，并新增本说明。迁移前后全部 28 个原文件的 SHA-256 一致，未修正原有导入或运行路径。整理前源文件树可通过 Git 对象 `19ea6dcfad4189b07a1fda4d8bd3d549122f5344` 追溯。

## 代码结构与阅读顺序

```text
RC-PPO/
├── env/
│   ├── env_list.py          # 环境选择、观测缩放
│   ├── wrappers.py          # 观测、动作及日志包装工具
│   ├── reach_avoid/         # 各系统的动力学、到达和避障定义
│   ├── wind_field_2.npz     # 风场数据
│   └── nv_humanoid.xml      # 随副本保留的 XML 资源
├── model/
│   └── actorcritic.py      # 策略与评价网络定义
├── rl/
│   ├── EC-EFPPO.py         # 主训练入口
│   ├── EFPPO_utils.py      # 环境采样、PPO 更新
│   ├── gae.py              # 优势、目标值及候选终止位置计算
│   ├── root_finding.py     # 基于评价网络的预算二分搜索
│   ├── arguments.py        # 命令行参数
│   └── utils.py            # 优化器、学习率和辅助函数
├── script/                 # 六份原始实验参数脚本
└── README.md
```

建议先看 `env/env_list.py` 和一个具体环境，再读 `model/actorcritic.py`、`rl/EC-EFPPO.py`、`rl/gae.py`、`rl/EFPPO_utils.py`，最后读 `rl/root_finding.py`。`model/actorcritic.py` 还保留了合并式 actor-critic 和 SAC 网络定义；主入口实际实例化的是独立策略网络及两个独立评价网络。

## 核心实现

环境将物理状态与剩余预算 `energy` 组合为观测；部分环境还加入记录是否曾进入危险区域的 `avoid` 标志。观测经过 `env_list.py` 中的固定缩放后送入网络。

| 组成 | 源码行为 |
| --- | --- |
| 策略网络 `Policy_Network` | 两层 256 单元的隐藏层，输出高斯分布并采样连续动作；环境通常再用 `tanh` 限幅。另有离散策略实现。 |
| 成本评价网络 `train_state_energy` | 独立的两层 256 单元网络，估计累计成本相关的值；网络输出与剩余预算作差。 |
| 到达评价网络 `train_state_h` | 同结构的独立网络，学习环境定义的 `reach` 值；若环境将危险状态写入 `reach`，其目标也包含该影响。 |
| 联合评价 | 主训练代码使用 `max(V_h, V_energy - energy)`，并构造即时项 `max(reach, -energy)`。 |
| 更新流程 | `jax.vmap` 并行采样环境，`jax.lax.scan` 收集轨迹；计算目标和优势后，分别更新策略及两个评价网络。 |
| 预算搜索 | `Bisection` 修改观测最后一维，在归一化区间 `[-1, 1]` 内按两个评价网络的组合值搜索零点；当前训练入口未直接调用该工具。 |

这里的 `energy` 是代码中的剩余成本预算，不应自动理解为真实电池能量。环境接口虽然沿用 `reward` 字段名，但已检查的任务返回的是非负的逐步消耗，例如动作平方和，并从预算中扣除；它不是常规“奖励越大越好”的任务奖励。

策略更新也保留了相应的最小化符号：`_ecefppo_update` 对两个裁剪候选取 `maximum` 并最小化，不能直接替换成普通 PPO 的奖励最大化写法。`gae.py` 中存在多种历史变体，阅读当前入口时应重点跟踪 `calculate_indexs3`、`calculate_gae2` 和 `calculate_gae_reach4`；其中重新构造的 `done` 用于学习目标的分段，不能直接等同于环境返回的终止标志。

## 环境与实验脚本

| 环境选择名 | 主要任务 | 对应原脚本 |
| --- | --- | --- |
| `PendulumConstraint` | 摆锤到达任务，包含动作阈值与消耗定义 | `script/run_pendulum_constraint.sh` |
| `HopperAvoidCeiling` | Hopper 到达目标位置并避开顶部约束 | `script/run_hopper_avoid_ceiling.sh` |
| `HalfCheetahAvoid` | HalfCheetah 到达目标位置并避开指定足部危险区域 | `script/run_half_cheetah_avoid.sh` |
| `WindField` | 风场中的到达避障，使用本地 NPZ 场景数据 | `script/run_wind_field.sh` |
| `F16Avoid` | F-16 到达指定位置并避开环境定义的危险状态 | `script/run_f16_avoid.sh` |

另有 `script/run_hopper_reach.sh`，但其 `HopperReach` 环境名没有在当前 `get_env` 中注册。参数解析器默认的 `GridConstraint` 同样不在当前环境选择分支中。环境文件存在表示本地保存了相应实现，不代表这些任务已在本机运行验证。

## 当前副本的运行状态

当前副本经过源码阅读与目录完整性检查，未执行训练或验证实验结果。它保留了原运行环境的假设，不能当作已经配置完成的可运行发行版。

| 检查项 | 已确认的情况 |
| --- | --- |
| 导入与机器路径 | 主入口引用 `EFPPO.src.*`，追加 `/home/mepear_gc`，检查点写入 `/home/mepear_gc/EFPPO/model/`；这些路径依赖原机器布局。 |
| 缺少的本地模块 | `rl/plot_utils.py`、`env/reach_avoid/hopper_deterministic.py`、`env/reach_avoid/half_cheetah_deterministic.py` 均被引用，但不在本地副本中。 |
| 环境导入范围 | `env_list.py` 在模块顶层导入 Hopper、HalfCheetah 和风场等环境，相关缺失依赖可能影响其他任务的启动。 |
| 原始相对路径 | 六份脚本保留 `./src/rl/EC-EFPPO.py`，风场保留 `./src/env/wind_field_2.npz`；目录上移后这些字符串也保持原样。 |
| 依赖配置 | 未随副本提供 `requirements.txt`、`pyproject.toml` 或环境锁文件，无法由本地材料确认兼容版本。 |
| 随机种子 | 参数中定义了 `--SEED`，但当前主入口实际使用 `jax.random.PRNGKey(20)`；阅读复现配置时应以实际调用为准。 |

从源码导入可见，主要依赖包括 JAX、Flax、Optax、Distrax、Chex、Gymnax、Brax、NumPy、Einops、Weights & Biases、Matplotlib、Colour 和 Python Control；F-16 环境另依赖外部 `jax_f16`。这是静态导入清单，不是经过安装验证的依赖说明。

后续若需复现，应在独立工作副本中补齐缺失模块、恢复兼容环境并处理路径，再验证结果；本目录继续保留原始参考状态。
