# 上层导航实施交接

## 当前交接：2026-11-12计划修订

当前[实施计划](../../../doc/plan/20261109_camra_navigation_implementation.json)为 **1.3.1，48/67项记录完成**，下一项 **NAV-061**。本次只修订现有任务与说明，未运行服务器、实现、训练或新评估；保留全部任务编号、数组顺序、历史状态与工作记录。

**已存在的实现不等于已经达标。** 名义环境、三网络训练、预算工具及物理接入已建立，但旧三轮开发集成功率为0，`reports/gate_G4.json`为`FAIL_WITH_STABILITY`。NAV-039经用户接受负结果结案；旧报告和冻结候选继续保留，不以新计划改写旧结论。

| 历史产物 | 当前可证明范围 | 本轮责任任务 |
| --- | --- | --- |
| 三轮名义训练、G3/G4与旧部署候选 | 已实现并运行过；共同停止、三估计器、时序及部署价值仍需回修/重验 | NAV-061–065 |
| `reports/budget_selection.json`的90回合 | 单种子经验标签扫描且全无解，不是双critic价值搜索 | NAV-066 |
| `reports/nav042_online_loop.json`的6回合 | 10/50/200 Hz接线pilot；clean随机项和子步事件证据不足以证明新要求 | NAV-067 |
| 原v2校准、旧G2A | 固定控制器和原采样条件的历史证据，不为未来学习策略自动背书 | NAV-043/048 |

恢复顺序为：NAV-061冻结原理、回修、数据和资源协议 → NAV-062对齐时序/预算/终止/动作接口 → NAV-063修正共同停止与三估计器并查生产调用链 → NAV-064有限名义训练 → NAV-065新版actor/同目标双critic及G4 → NAV-066独立价值预算搜索 → NAV-067新版候选clean子步接线 → NAV-043–054正式复查、评估与交付。NAV-067依赖NAV-066，不复用旧候选跳过重验。

训练只用train/development调参及选checkpoint，登记旧选择集暴露；独立选择与`final_*`不回流训练。90回合只覆盖预先指定的单种子/方法；新的独立清单、重复性、pilot或资源额度按计划取得必要授权。旧失败、全部attempt与证据保留，回修主体记录追加到发现问题的当前任务。

固定v2整套执行/校准资源，执行主机固定tiger11（192.168.20.11）+tmux。下文服务器路径、解释器和会话是历史线索，恢复前现场核验，不因能看到共享仓库而推定执行主机；本次没有重新核验其可用性。v2、旧校准、RC-PPO/HiRA和论文不在修改范围。

## 历史快照：2026-11-10（NAV-001–010完成时）

以下保留当时的交接内容；其中“未开始”“下一项NAV-055”等仅描述该历史时点，不是当前任务入口。

更新时间：2026-11-10T10:10:00+08:00。执行主机 tiger11（192.168.20.11）。

### 1. 已完成与未开始

| 阶段 | 任务 | 状态 |
|---|---|---|
| 背景学习 | NAV-001–006 | 完成，产出 onboarding.md 与 source_reference_map.md |
| 服务器与资源 | NAV-007–010 | 完成，产出 runtime_environment.json、runtime_paths.json、v2_resources.json、包骨架 |
| 契约与名义环境 | NAV-012–023（含 055–058） | **未开始**，等 G0 |
| 后续 | NAV-011 及之后 | 未开始 |

**本阶段没有任何仿真、训练、采集或依赖安装以外的动作**；唯一对环境的写入是 editable 安装了本仓库的 `camra-navigation` 包。

### 2. 目录清单

**仓库内（现有 + 本次新建）**

| 路径 | 性质 | 说明 |
|---|---|---|
| `CaMRA/navigation/README.md` | 现有 | 模块说明 |
| `CaMRA/navigation/config/runtime_paths.json` | NAV-008 新建 | 路径、解释器裁定、框架决定、依赖、隔离方案 |
| `CaMRA/navigation/docs/onboarding.md` | NAV-001/002/004/006 | 基线、方法、v2 契约、六阶段矩阵 |
| `CaMRA/navigation/docs/source_reference_map.md` | NAV-003/005 | RC-PPO 调用链、HiRA 分层结构 |
| `CaMRA/navigation/docs/runtime_environment.json` | NAV-007/008 | 主机、tmux、依赖、冻结环境身份 |
| `CaMRA/navigation/docs/implementation_handoff.md` | NAV-010（本文件） | 交接 |
| `CaMRA/navigation/reports/v2_resources.json` | NAV-009 | v2 资源核验报告 |
| `CaMRA/navigation/pyproject.toml` | NAV-010 新建 | 包定义 |
| `CaMRA/navigation/src/camra_navigation/` | NAV-010 新建 | `__init__.py`、`runs.py`（运行身份规则） |
| `CaMRA/navigation/tests/`、`scripts/` | NAV-010 新建 | 空目录，等后续任务填充 |

**仓库外（新建，与旧产物隔离）**

| 路径 | 说明 |
|---|---|
| `/pub/data/caohy/work/camra_navigation` | 工作根（代码快照、中间产物） |
| `/pub/data/caohy/logs/camra_navigation` | 运行输出根 |

两者均不使用、不写入 `logs/camra`（v2 校准）或 `logs/camra_hira`（HiRA 实验）。

### 3. 运行身份与冲突拒绝

`camra_navigation.runs` 已强制三条：

1. run id 必须匹配 `^[a-z0-9][a-z0-9._-]{2,79}$`；
2. 目标目录已存在时**默认拒绝**；即便 `allow_existing=True`，非空目录也拒绝；
3. `run_manifest.json` 不可覆盖，写入 hostname、started_at、cwd、python 与 controller_id。

实测三个反例均被拒绝（重名、非法 id、非空目录），正例可创建并写出 manifest。

### 4. 环境

- 解释器：`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`（Python 3.11.15）
- isaaclab 0.47.2 / isaacsim 5.1.0.0 / rsl-rl-lib 3.0.1 / torch 2.7.0+cu128 / numpy 1.26.0 / onnxruntime 1.31.1
- 关键偏差：torch 为 cu128（冻结记录 cu126）、Python 3.11.15（记录 3.11.18）——已登记，未修正
- 上层框架：**PyTorch**（用户 2026-11-10 批准）
- 包：`camra-navigation 0.1.0`，editable 指向 `src/camra_navigation`

### 5. 下一步入口

**下一个可执行任务是 NAV-055（速度域冲突裁定）**，不是 NAV-011。

- NAV-055 需要：依据原始 applied 命令抽样 + 数据锁，判定有效命令域是 `1.0/0.4/1.0` 还是 `2.0/1.0/2.0`。
- 已确认事实：protocol、v2 contract、power_model.domain 三方一致为 `1.0/0.4/1.0`；仅 `margins_v1.applicability` 的**文本**为 `2/1/2`。
- 该裁定未完成前，G0（NAV-011）不得放行。

### 6. 未决点

| 项 | 负责 | 阻塞 |
|---|---|---|
| 速度域裁定 | NAV-055 | G0 |
| 姿态足迹处理 | NAV-057 | G1 |
| z_ref 物理定义 | NAV-056 | G1 |
| 预算公平性协议 | NAV-058 / NAV-021 | 正式训练 |
| JAX 对照环境与依赖授权 | NAV-033 | G3 |
| 地图净宽与 30 s 可达 | NAV-017 | G1 |

### 7. 恢复方式

```bash
ssh 192.168.20.11 -t tmux attach -t camra-nav-v2-20261110-bootstrap
set -a; . /pub/data/caohy/v2_navigation.env; set +a
cd /home/caohy/repositories/Paper
```

断线后先 `tmux ls` 与查进程，不重复启动同一 run。
