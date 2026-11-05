# HiRA 控制器包 hira_go2_18500 v1

不可变。任何权重、资产、增益、观测、时钟的改动都必须产生新的 controller 版本并重过受影响的门禁（HIRA-020 约束）。

- 策略：`policy.pt`（TorchScript，sha256 721a891a22da99f2…）
- 描述：`controller.json`
- 清单：`manifest.json`（bundle_id 覆盖描述内容与逐文件散列）

重载与校验：`from hira_go2.bundle import load_bundle; load_bundle(path)`
