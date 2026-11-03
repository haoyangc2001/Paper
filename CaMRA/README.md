# CaMRA

面向 Unitree Go2 四足机器人的分层最小成本到达避障系统。底层学习速度跟踪及稳定运动，冻结后采集剩余执行误差，校准成本裕度 α 和几何裕度 δ；上层结合校准结果，根据观测、目标和预算生成导航速度指令。

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
| [locomotion](locomotion/README.md) | 学习速度跟踪，承担平衡、步态协调和动作平滑 | 目标速度与本体状态 → 关节控制；产出冻结的策略及执行配置 | **已完成训练、评估、导出与冻结**：final_test 200–209 单次执行，nominal 与 perturbed 各 10/10 场景通过；冻结交付件 `camra_go2_velocity_it2999_v2` 校验 PASS |
| [calibration](calibration/README.md) | 测量剩余执行误差，校准并独立检验两类裕度 | 冻结控制器、配对的名义与执行轨迹 → α、δ 及对应配置记录 | 目录和用途说明已建立，**代码待实现**；所需输入（冻结包契约 + 19,200 条观测样本 + 闭环重载脚本）已齐备 |
| [navigation](navigation/README.md) | 学习满足安全与预算要求的到达决策 | 环境观测、目标、预算及校准结果 → 三维速度指令 | 目录和用途说明已建立，**代码待实现** |

## 模块配合与开发顺序

1. **训练底层（已完成）。** 在 `locomotion/` 中训练、评估速度跟踪策略，验证停车、低速转向与连续变速后
   冻结策略及配套观测、动作和控制配置。交付件是 `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2`
   （仓库外 28 MB，只读；仓库内只放 17 KB 清单副本 `locomotion/docs/frozen/`）。
2. **执行校准（待做）。** 在 `calibration/` 中运行冻结控制器，采集并配对名义轨迹和实际执行轨迹，计算成本裕度 α
   和几何裕度 δ，再用独立数据检验。结果必须与底层模型版本及采样条件一同保存。
3. **训练上层（待做）。** 在 `navigation/` 中实现导航环境、策略网络和两个评价网络，接入裕度及冻结控制器，
   再完成训练和评估。

运行时，上层将机身坐标系下的目标速度 `[vx, vy, yaw_rate]` 发送给冻结的底层控制器，底层结合本体状态输出关节控制。上层负责目标、地图及预算；校准模块用于离线确定执行裕度。速度范围、控制频率及动作映射以冻结包的
`params/deploy.yaml` 与 `params/export_manifest.json` 为唯一权威来源，**不要从文档转述取值**。

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
