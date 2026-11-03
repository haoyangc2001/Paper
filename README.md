# 四足机器人导航研发工作区

本仓库集中管理 CaMRA 的实现、参考项目与论文资料。当前目标是在 Isaac Lab 中复现 Go2 分层最小成本到达避障：复用冻结的底层控制器，完成执行误差校准与独立检验，再实现上层导航。

## 当前进展与下一步

当前实施任务是 [仿真能量与位置误差校准](CaMRA/calibration/README.md)，执行入口为 [32项校准实施计划](doc/plan/20261103_camra_calibration_implementation.json)。计划尚未执行，从 CAL-001 开始：先让新上下文的接手模型学习论文和项目代码，再在新设备上配置环境、取得冻结模型并验证运行，然后实现校准模块。

本轮固定30秒测量窗口，建立独立名义功率模型并测量仿真执行能量；使用800条校准、400条独立检验和200条近障检查，交付功率模型、成本裕度 α、几何裕度 δ、检验报告和上层消费接口。能量指标及基线采样预算在 CAL-008 核验确认。当前任务状态以计划中的 `tasks` 为准，底层训练计划保留为历史证据。

2026-11-03 已决定接受冻结控制器 `camra_go2_velocity_it2999_v2` 的已知低速局限，继续推进仿真校准，不再把底层继续调参或重新训练作为默认前置条件。旧复测仍记为未通过；接受现有版本用于后续研究不改变历史判定。具体边界见 [底层模块说明](CaMRA/locomotion/README.md#当前状态与已知边界)。

| 模块 | 当前状态 | 后续工作 |
| --- | --- | --- |
| [locomotion](CaMRA/locomotion/README.md) | 已训练、导出并冻结 v2；已知低速死区保留 | 复用同一模型、资产和控制契约 |
| [calibration](CaMRA/calibration/README.md) | 说明和实施计划已建立，代码待实现 | 当前任务：背景学习、环境配置、能量与位置误差校准及独立检验 |
| [navigation](CaMRA/navigation/README.md) | 仅有用途说明，代码待实现 | 接入校准结果，完成导航训练及评估 |

本阶段仅做仿真。运行前仍需在执行设备上取得冻结包、配置匹配环境并核对包体；真机通信与部署留到后续。

## 目录与阅读入口

```text
paper/
├── AGENTS.md          # Agent 接手入口：当前任务、阅读顺序、约束
├── README.md          # 工作区总览
├── CaMRA/             # 本项目实现
│   ├── locomotion/    # Go2 底层速度跟踪器
│   ├── calibration/   # 执行误差校准与独立检验
│   └── navigation/    # 上层最小成本到达避障导航器
├── RC-PPO/            # 冻结的原算法参考副本
├── HiRA/              # 固定版本的分层 Reach-Avoid 参考子模块
├── manuscripts/       # 当前论文、历史稿件、文献、绘图及数据
├── doc/               # 实施计划与工作记录索引
└── .local/            # Git 忽略的本机备份
```

- [CaMRA 系统说明](CaMRA/README.md)：方法、模块关系和统一接口。
- [AGENTS.md](AGENTS.md)：接手任务时的最短阅读路径、配置依据与工作边界。
- [当前校准计划](doc/plan/20261103_camra_calibration_implementation.json)：任务依赖、验收与后续逐项工作记录。
- [文档索引](doc/README.md)：当前任务阶段及历史计划、执行证据入口。
- [RC-PPO](RC-PPO/README.md)、[HiRA](HiRA/README.md)：两个参考项目，各自的代码与使用说明。
- [论文资料](manuscripts/README.md)：稿件、图片、数据、参考文献及编译入口。

各目录的 README 解释该目录项目；详细实验经过和命令证据保留在对应计划、交接笔记中。

## 获取完整工作区

```bash
git clone --recurse-submodules https://github.com/haoyangc2001/Paper.git
```

已有检出目录在保存自身修改并更新主仓库后，初始化 HiRA：

```bash
git submodule sync -- HiRA
git submodule update --init --recursive -- HiRA
git submodule status -- HiRA
```

HiRA 使用主仓库锁定的提交，本阶段只读。若已有修改，先检查 `git -C HiRA status --short`；普通 clone 或主仓库 ZIP 不会带入子模块内容。HTTPS 受限且已有 SSH 权限时，可只覆盖本次连接地址：

```bash
git -c submodule.HiRA.url=git@github.com:haoyangc2001/HiRA.git submodule update --init --recursive -- HiRA
```

模型权重、资产和原始日志没有随主仓库克隆。取用与校验见 [底层复现说明](CaMRA/locomotion/docs/reproduction.md)。

## 论文与保留资料

当前稿件：[1029 LaTeX](manuscripts/CCDC/latex/1029/main.tex)、[论文 PDF](manuscripts/CCDC/output/1029/main.pdf)、[编译说明](manuscripts/CCDC/README.md)。在根目录编译：

```powershell
& ./manuscripts/CCDC/scripts/build-1029.ps1
```

`manuscripts/CCDC/data/1029/` 保留数据及来源台账，`manuscripts/figures/` 保存各图的数据、脚本与成图。论文中的示例数值不作为本项目已完成实测的证据。

2026-10-31 的目录迁移映射见 [写作资料说明](manuscripts/README.md)；过程文件的本机可恢复备份见 [.local 清理说明](.local/cleanup_20261031/README.md)，该备份由 Git 忽略。历史稿件、原审稿意见和已有实验记录继续保留。
