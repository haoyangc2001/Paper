# CaMRA

面向 Unitree Go2 四足机器人的分层最小成本到达避障系统。底层学习速度跟踪及稳定运动，冻结后采集剩余执行误差，校准成本裕度 α 和几何裕度 δ；上层结合校准结果，根据观测、目标和预算生成导航速度指令。

## 当前范围：仿真复现

本阶段在仿真中实现并评估论文方法，先沿用现有 Isaac Lab 环境。真机通信、关节映射、sim2real 适配和实机试验暂不开展，也不作为仿真复现的验收条件。

底层优先复用已有的冻结策略或适配后的现成策略，不默认从头训练。当前已有 `camra_go2_velocity_it2999_v2`，先核对执行服务器上的包体、配套资产及重载入口，并完成已识别的补充验收；只有现有策略确实不满足任务需要时，再评估是否补训或更换。训练代码保留供后续使用。

在校准中，名义轨迹来自导航使用的确定性运动模型；执行轨迹来自仿真中的 Go2、固定底层策略和物理执行过程，并在明确的摩擦、外力等扰动条件下采集。关闭随机扰动不等于没有跟踪误差。校准、独立检验和导航评估须固定同一控制器及接口；更换控制器或执行配置后重新校准。

## 模块划分

```text
CaMRA/
├── README.md
├── locomotion/       # 底层速度跟踪训练、评估、导出及部署
├── calibration/      # 执行误差采集、裕度校准与独立检验
└── navigation/       # 上层导航环境、网络、训练与评估
```

| 模块 | 职责 | 主要输入与产物 | 当前状态 |
| --- | --- | --- | --- |
| [locomotion](locomotion/README.md) | 学习速度跟踪，承担平衡、步态协调和动作平滑 | 目标速度与本体状态 → 关节控制；产出冻结的策略及执行配置 | **训练、评估、导出与冻结已完成，确认性复测未通过验收**：旧 final_test 200–209 为场景级口径，nominal 与 perturbed 各 10/10 场景通过（实测覆盖 15 个协议变体中的 10 个）；TASK-036 补充复测 9000 回合、逐变体判定，29/30 变体达标，nominal `lateral#v1`（`vy=−0.3`）`rmse_vy` 0.1035 超阈值 0.1000。冻结交付件 `camra_go2_velocity_it2999_v2` 校验 PASS，但**不得宣称已具备校准交付条件**。根因（低速死区源于训练采样中的停车样本）与已选定的修复路径（提高低速跟踪奖励权重，见交接笔记 §39–47）见 locomotion/README |
| [calibration](calibration/README.md) | 测量剩余执行误差，校准并独立检验两类裕度 | 冻结控制器、配对的名义与执行轨迹 → α、δ 及对应配置记录 | 目录和用途说明已建立，**代码待实现**；所需输入（冻结包契约 + 19,200 条观测样本 + 闭环重载脚本）已齐备，但底层逐变体验收未通过（TASK-036），进入校准前需先裁决处置方式 |
| [navigation](navigation/README.md) | 学习满足安全与预算要求的到达决策 | 环境观测、目标、预算及校准结果 → 三维速度指令 | 目录和用途说明已建立，**代码待实现** |

## 模块配合与开发顺序

1. **复用并核验底层。** 已完成一版训练与冻结，优先使用现有控制器，核验配套配置及可重载性并补齐验收。
   交付件是执行服务器上的 `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2`
   （仓库外 28 MB，只读；仓库内只放 17 KB 清单副本 `locomotion/docs/frozen/`）。普通克隆不会获取模型包，运行前需取得包体及匹配环境。
2. **仿真执行校准（待做）。** 在 `calibration/` 中运行冻结控制器，采集并配对名义轨迹和仿真执行轨迹，计算成本裕度 α
   和几何裕度 δ，再用独立数据检验。结果必须与底层模型版本及采样条件一同保存。
3. **仿真导航（待做）。** 在 `navigation/` 中实现导航环境、策略网络和两个评价网络，接入裕度及冻结控制器，
   再完成训练、方法对比和消融评估。

运行时，上层将机身坐标系下的目标速度 `[vx, vy, yaw_rate]` 发送给冻结的底层控制器，底层结合本体状态输出关节控制。上层负责目标、地图及预算；校准模块用于离线确定执行裕度。速度范围、控制频率及动作映射以冻结包的
`params/deploy.yaml` 与 `params/export_manifest.json` 为唯一权威来源，**不要从文档转述取值**。

## 宇树官方现成方案核查（2026-11-03）

本次核查以下官方公开仓库的主线说明、Go2 配置及预训练目录，未找到同时配齐 **Go2 仿真环境、Go2 预训练行走控制器和可直接调用的导航速度接口** 的完整开箱方案。

| 官方项目 | 已提供的部分 | 与当前需要的差距 |
| --- | --- | --- |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/blob/main/readme.md) | Go2 物理仿真、机器人模型、底层控制及状态通信 | 官方明确说明当前仅支持低层开发；不包含真机内置的 Sport 速度控制服务，不能仅靠 `SportClient.Move(vx, vy, wz)` 让仿真 Go2 行走 |
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | Go2 速度跟踪训练环境及策略加载、控制框架 | [Go2 配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/robots/go2/config/config.yaml) 的 `policy_dir` 指向训练日志；该 Go2 配置目录未附现成策略，仍需取得匹配模型 |
| [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | Go2 训练任务，以及部分机器人的 MuJoCo 推理示例和预训练模型 | 当前 [预训练目录](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/pre_train) 与 [MuJoCo 配置目录](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_mujoco/configs) 仅列 G1、H1、H1-2，不能将这些模型当作 Go2 模型使用 |
| [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) | 带部分权重及移动操作入口的 Isaac Lab 仿真场景 | 当前说明列出的对象为 G1、H1-2，不是 Go2 的配套方案 |

因此当前保留本项目已有的 **Isaac Lab + 冻结控制器 + 三维外部速度命令接口** 路线，不因本次调研切换仿真器。现有 v2 的补充复测未通过；按当前安排，等待底层训练完成并通过验收后，再继续校准与上层导航。闭环策略重载入口为 `locomotion/scripts/rollout_exported_policy.py`；命令桥接实现为 `locomotion/source/unitree_rl_lab/unitree_rl_lab/utils/external_command.py`。这套组合仍需在执行设备上取得模型包、配置依赖并核验，不能将文档核查当作本次已完成的运行测试。

## 安装与运行入口

底层框架的环境要求、安装和 Go2 训练/评估/导出/冻结命令见 [locomotion/README.md](locomotion/README.md)；
逐条真跑过的完整命令（含恢复训练、校准入口、日志路径与已知坑）见
[locomotion/docs/implementation_handoff.md §30](locomotion/docs/implementation_handoff.md)。从工作区根目录进入底层模块后执行：

```bash
cd CaMRA/locomotion
python -m pytest tests/test_units.py -q                      # 纯 CPU，34 项
python scripts/verify_frozen_bundle.py \
  --bundle /pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2 --samples 512
```

上游 VS Code 配置也以 `locomotion/` 为工作目录；使用这些配置时，将该模块目录作为编辑器工作区打开。上层和校准模块的运行入口将在实现后补充。

## 来源与工作区关系

- 底层框架来自宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，具体版本及本地调整见 [上游来源记录](locomotion/UPSTREAM.md)。原始许可证和版权声明保留；Python 包名及启动脚本沿用上游命名，任务 ID 改为本项目的 `CaMRA-Go2-Velocity`。
- [RC-PPO](../RC-PPO/README.md) 是工作区中独立保留的冻结算法参考副本，后续修复及机器人适配在 `navigation/` 中进行。
- [HiRA](../HiRA/README.md) 是 Reach-Avoid 分层导航参考项目，以根仓库 Git 子模块固定版本，保留独立历史，本阶段只读。可参考上下层衔接、已有 Go2 策略和轨迹评估工具；缺少源码时按根 README 初始化子模块。
- [manuscripts](../manuscripts/README.md) 保存论文、绘图及参考资料。

本目录由工作区根目录的 Git 仓库统一管理，不包含嵌套 Git 仓库或子模块。
