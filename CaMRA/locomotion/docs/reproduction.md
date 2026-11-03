# 换设备复现与产物取用边界（TASK-034）

本文件回答三个问题：**哪些东西在 git 里、哪些不在**，**不在的东西在哪里、怎么取**，
以及**取到之后怎样证明拿对了**。

## 1. 边界

| 内容 | 位置 | 是否入库 | 体量 |
| --- | --- | --- | --- |
| 代码、配置、协议、测试 | 本仓库 `CaMRA/locomotion/` | **是** | 数 MB |
| 冻结交付件 `camra_go2_velocity_it2999_v2` | `$CAMRA_BUNDLE_ROOT/frozen/` | 否（只入 `docs/frozen/` 的清单副本） | 28 MB / 31 文件 |
| 候选包 `r4_it2999_v1`（JIT/ONNX + 验证） | `$CAMRA_BUNDLE_ROOT/candidates/` | 否 | 5.2 MB / 6 文件 |
| 训练检查点 `model_2999.pt` 与训练参数 | `$CAMRA_RSL_LOG_ROOT/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/` | 否 | 4.6 MB |
| 机器人资产 `Go2`（USD + 网格） | `$CAMRA_ASSET_ROOT/Go2` | 否（冻结包内有一份拷贝） | 18 MB / 4 文件 |
| 原始评估日志 TASK-024 / TASK-026 | `$CAMRA_LOG_ROOT/TASK-0xx/` | 否 | 662 MB / 362 文件、10.6 MB / 16 文件 |

**为什么不在 git 里**：模型与原始日志属于二进制大文件，且 `TASK-024` 有 36 万行逐回合 CSV；
把它们放进仓库会让每次 clone 都变慢，也无法用 diff 审阅。冻结包本身已经是"可重载的最小集合"，
它把策略、参数、资产拷贝和判定文本都装了进去，**拿到冻结包就能重载控制器，不需要训练目录**。

**没有任何凭证进入本仓库或索引**：索引只记录路径、大小与 sha256。

## 2. 索引

[`docs/artifacts_index.json`](artifacts_index.json) 是机器可校验的清单，内容由
`python scripts/artifact_index.py build` 生成，**路径按 `CAMRA_*` 解析，换机器不会失效**。

| 条目 | sha256（文件）/ tree_sha256（目录） | 大小 |
| --- | --- | --- |
| `frozen_bundle_v2` | `8923c6fd4f05a921…` | 29,127,046 B / 31 文件 |
| `frozen_bundle_v1_superseded` | `9cf9a07099ed195c…` | 29,124,660 B / 31 文件 |
| `candidate_bundle` | `e3b21eb382a123d8…` | 5,454,281 B / 6 文件 |
| `checkpoint_selected` | `1c8f5aad38c2b763…` | 4,639,669 B |
| `training_run_params` (`env.yaml`) | `922d473cb449b2cb…` | 32,299 B |
| `assets_go2` | `bd2fcb2f235b979f…` | 18,815,207 B / 4 文件 |
| `raw_logs_task024` | `8df2250af159c16a…` | 662,001,271 B / 362 文件 |
| `raw_logs_task026` | `32ea7954eefb787f…` | 10,597,117 B / 16 文件 |
| `protocol` | `0afc2ceadf8f8511…` | 9,167 B（入库） |
| `protocol_supplement` | `f50d8b7f5ca6f815…` | 6,640 B（入库） |

目录条目是**树散列**：对目录内每个文件算 sha256，再对排序后的 `相对路径:sha256` 列表取 sha256，
所以内容被替换、文件被增删、目录结构被改动都会变。完整散列与逐条说明见 JSON 文件。

## 3. 取用

都在已授权主机 `caohy@192.168.20.12` 上、走 ssh 内网，不经过任何外部服务：

```bash
# 只取重载控制器所需的最小集合（推荐给上层模块）
rsync -a caohy@192.168.20.12:/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2/ \
      ./camra_go2_velocity_it2999_v2/

# 需要重跑评估：再取检查点与资产
rsync -a caohy@192.168.20.12:/pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/ \
      /mnt/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/
rsync -a caohy@192.168.20.12:/pub/data/caohy/unitree_model/Go2/ /mnt/assets/unitree_model/Go2/

# 需要复算历史判定：再取原始日志（662 MB）
rsync -a caohy@192.168.20.12:/pub/data/caohy/logs/camra/TASK-024/ /mnt/logs/camra/TASK-024/
```

取到之后先把新路径写进 `config/repro.env`（或导出 `CAMRA_*`），再校验：

```bash
python scripts/artifact_index.py verify   # 逐条比对，任一不符返回 4
python scripts/check_repro_readiness.py --level full-repro
```

## 4. 换设备四步

1. `git clone` 本仓库，`pip install -e CaMRA/locomotion/source/unitree_rl_lab`（另需 torch / isaaclab / isaacsim / rsl-rl-lib，版本见 README）。
2. 改 `config/repro.env` 里的 `CAMRA_*`（只改这一个文件，不碰源码）。
3. `python scripts/check_repro_readiness.py --level eval-now` → 必须是 `VERDICT: ready`。
   返回 `infra_blocked`(2) 是资源缺失、`env_incompatible`(3) 是环境不对，**两者都不是控制器不合格**。
4. `scripts/repro/finaltest.sh <网络路径或本地检查点> both`，输出默认落在 `$CAMRA_LOG_ROOT/repro/finaltest/`，
   与历史产物目录分开，不会覆盖既有证据。

## 5. 明确没做的事

- 没有把模型或原始日志加入 git，也没有上传到任何新的外部服务；
- 没有替换资产内容：只是把资产根目录改成可配置，解析到的仍是同一棵树（`assets_go2` 树散列不变）；
- 没有改任何阈值、协议或训练超参；
- 索引里的绝对路径是交付主机的路径。换机器后 `verify` 按新解析到的路径重新计算，**只比对内容散列，不比对路径字符串**。
  清单存在不等于包体存在：`verify` 与 `check_repro_readiness.py` 都会真的去读文件，缺失即报 `MISSING`。
