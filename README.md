# 四足机器人导航研发工作区

## 当前目标：先在仿真中复现 CaMRA

当前阶段仅在仿真中完成 Go2 分层最小成本到达避障的复现：接入可用的底层速度控制器，采集名义与受扰动执行轨迹，校准并独立检验成本裕度 α 和几何裕度 δ，再实现上层导航与对比实验。真机接口、上机适配和实机试验留到后续阶段。

底层优先复用已有策略，不把重新训练作为开展校准和导航的默认前置条件。当前先沿用 Isaac Lab 框架及已冻结的 `camra_go2_velocity_it2999_v2`，核实模型、资产和接口可重载并完成必要的补充验收；若采用其他现成策略，须使用其配套观测、动作和控制配置，并重新采集校准数据。

当前安排为等待底层训练完成并通过验收，再继续校准与上层导航。现有 v2 控制器的补充复测未通过，具体结果保留在 [底层说明](CaMRA/locomotion/README.md)，不将冻结包校验通过等同于控制器验收通过。

## 目录结构

```text
paper/
├── manuscripts/       # 论文写作、参考资料与绘图
│   ├── CCDC/          # 当前投稿及历史完整稿
│   ├── CORL/          # 早期稿件与原审稿意见
│   └── figures/       # 原“绘图”：各图的数据、脚本与成图
├── RC-PPO/            # 冻结保留的原算法参考实现
├── HiRA/              # Git 子模块：Reach-Avoid 分层导航参考项目
├── CaMRA/             # 四足机器人分层导航系统
│   ├── README.md      # 系统介绍、模块关系与开发顺序
│   ├── locomotion/    # 底层强化学习速度跟踪器
│   ├── calibration/   # 冻结底层后的执行误差校准与检验
│   └── navigation/    # 上层最小成本到达避障导航器
├── .local/            # 本机备份，由 Git 忽略
└── README.md
```

实现代码集中在 [CaMRA](CaMRA/README.md) 下。`CaMRA/locomotion/` 已基于宇树官方框架完成一版控制器训练与冻结；模型包保存在执行服务器，仓库内保留清单及验证入口。`CaMRA/calibration/` 和 `CaMRA/navigation/` 目前仅包含用途说明。后续顺序为：复用并核验现有底层 → 仿真执行误差校准与独立检验 → 上层导航训练及评估。宇树官方现成方案的核查结果见 [CaMRA README](CaMRA/README.md#宇树官方现成方案核查2026-11-03)。

- [CaMRA Locomotion](CaMRA/locomotion/README.md)：基于宇树官方 `unitree_rl_lab`，接收速度指令、输出关节控制，承担平衡、步态协调和动作平滑；上游版本见 [来源记录](CaMRA/locomotion/UPSTREAM.md)。
- [calibration](CaMRA/calibration/README.md)：比较名义轨迹与实际执行，计算并独立检验成本裕度 α 和几何裕度 δ。
- [navigation](CaMRA/navigation/README.md)：根据观测、目标和预算生成速度指令，完成最小成本到达避障导航。

[RC-PPO 参考实现说明](RC-PPO/README.md)介绍原算法的网络、训练流程、环境及现有副本状态。该目录已去除中间的 `src/` 层，原代码保持不变，后续修复和机器人适配在新项目中进行。

[HiRA](HiRA/README.md) 是基于 Reach-Avoid 理论的四足机器人分层避障导航参考项目，原仓库名为 `Go2HierarchicalReachAvoidRL`，远程已改名为 [haoyangc2001/HiRA](https://github.com/haoyangc2001/HiRA)。它以 Git 子模块纳入本仓库，保留独立历史；主仓库固定参考提交，本阶段只读，不修改其中代码。默认分支为 `master`，执行时使用主仓库锁定的提交，不自动追踪最新分支。

## 获取完整工作区

新检出时包含 HiRA：

```bash
git clone --recurse-submodules https://github.com/haoyangc2001/Paper.git
```

已有检出目录在保存自身修改并更新主仓库后，执行：

```bash
git submodule sync -- HiRA
git submodule update --init --recursive -- HiRA
git submodule status -- HiRA
```

如果 HiRA 目录已有独立修改，先检查 `git -C HiRA status --short`，不要用强制更新覆盖。普通 clone 或下载主仓库 ZIP 不会自动带入子模块内容；需要上述初始化操作。`.gitmodules` 使用 HTTPS 地址，不要求 SSH 密钥；私有仓库仍需要相应访问权限。

若设备的 HTTPS 连接受限，但已有可用的 GitHub SSH 权限，可仅对这次初始化覆盖连接地址（不修改共享配置）：

```bash
git -c submodule.HiRA.url=git@github.com:haoyangc2001/HiRA.git submodule update --init --recursive -- HiRA
```

底层实施入口：[实施计划（28 项原任务及 9 项补充任务）](doc/plan/20261031_camra_locomotion_implementation.json)。先阅读其中 `implementation_clarifications` 和 `external_learning_report`，了解配置依据、待验证项和已知历史问题；任务命令按计划要求放在执行主机的 tmux 中运行。当前以仿真复现和现有模型的必要验收为范围，计划中的历史真机说明不作为本阶段前置条件。

## 当前稿件

- [1029版LaTeX主稿](manuscripts/CCDC/latex/1029/main.tex)
- [最新论文PDF](manuscripts/CCDC/output/1029/main.pdf)
- [编译及目录说明](manuscripts/CCDC/README.md)
- [当前稿件说明](manuscripts/CCDC/latex/1029/README.md)

在仓库根目录编译：

```powershell
& ./manuscripts/CCDC/scripts/build-1029.ps1
```

## 保留资料

- `manuscripts/CCDC/images/1029/`：正文图片。
- `manuscripts/CCDC/data/1029/`：绘图数据、配置、来源和推断台账。
- `manuscripts/CCDC/references/1029/`：参考论文PDF及来源。
- `manuscripts/CCDC/scripts/`、`manuscripts/figures/`：编译、绘图及复现脚本。
- `manuscripts/CCDC/submission/1029/`：投稿要求、作者资料、数据替换说明与必要核查。
- `manuscripts/CCDC/latex/1020/`、`manuscripts/CORL/`：历史稿件；原审稿意见位于 `manuscripts/CORL/submission/`。
- `RC-PPO/`：冻结的算法参考实现，后续不直接修改其中代码。

## 过程文件清理

2026-10-31已将计划、阶段审查、重复预览、浏览器快照和临时编译文件集中备份后移出工作目录。当前论文、图片及数据未改动，数据性质及来源记录保留。

本机恢复入口：[备份说明](.local/cleanup_20261031/README.md)。压缩包保留整理前的目录结构，可按清单选择恢复；本机备份由 Git 忽略。过程文件清理已单独提交，可通过 Git 历史恢复。

2026-10-31将写作资料集中到 `manuscripts/`；迁移映射及历史路径说明见 [写作目录说明](manuscripts/README.md)。
