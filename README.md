# 四足机器人导航研发工作区

## 目录结构

```text
paper/
├── manuscripts/       # 论文写作、参考资料与绘图
│   ├── CCDC/          # 当前投稿及历史完整稿
│   ├── CORL/          # 早期稿件与原审稿意见
│   └── figures/       # 原“绘图”：各图的数据、脚本与成图
├── RC-PPO/            # 冻结保留的原算法参考实现
├── locomotion/        # 底层强化学习速度跟踪器
├── calibration/       # 冻结底层后的执行误差校准与检验
├── navigation/        # 上层最小成本到达避障导航器
├── .local/            # 本机备份，由 Git 忽略
└── README.md
```

三个实现目录目前仅包含用途说明，尚未添加代码。开发顺序为：训练底层速度跟踪器，冻结后采集执行误差并校准、检验裕度，最后训练上层导航器。

- [locomotion](locomotion/README.md)：接收速度指令，输出关节控制，承担平衡、步态协调和动作平滑。
- [calibration](calibration/README.md)：比较名义轨迹与实际执行，计算并独立检验成本裕度 α 和几何裕度 δ。
- [navigation](navigation/README.md)：根据观测、目标和预算生成速度指令，完成最小成本到达避障导航。

[RC-PPO 参考实现说明](RC-PPO/README.md)介绍原算法的网络、训练流程、环境及现有副本状态。该目录已去除中间的 `src/` 层，原代码保持不变，后续修复和机器人适配在新项目中进行。

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
