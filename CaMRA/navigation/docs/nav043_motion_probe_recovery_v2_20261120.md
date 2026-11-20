# NAV-043：采集接口回修与一次恢复提案

本轮原12工况只启动1个应用和M01，完整完成0/12。895行实际状态、机械功和调用账本已由主agent只读复核；35个q区间和223份完成命令证明可复算，末893–895没有完成证明。原失败、输出、持久锁和终态保存缺口全部保留，详见[首例失败](nav043_motion_probe_first_failure_20261120.md)。

根因是采集程序把理想0.005秒和Isaac实际float32回调的小数表示作过严的累计比较；第895步累计差刚超过1e-7秒。另有清理wait超时遮盖首错、未写出HELD的问题。不能把它们当作导航策略失败或物理漏步。

已准备隔离v2：

- `simulation_clock_v2.py`和`motion_capture_v2.py`同时保存真实时钟、实际整数step索引，逐步拒绝缺步、重步、陈旧数据或不符的dt。只允许预先明确的float64/float32回调表示，J和q仍用原0.005秒单位。
- 新恢复启动器、采集接线和保存审阅器形成独立调用链；原v1封存文件不修改。安装的SimulationContext源码已确认提供`current_time_step_index`，但新版真实Isaac调用链仍未执行。
- `nav043_process_cleanup_v2.py`保留TERM/KILL及wait错误、实际退出码和未确认状态；父进程单独保存原始应用错误、HELD和清理结果，不补造缺失child记录。

主agent在tiger11/tmux `%688`完成20项CPU基本检查（2.23秒），包含两种时钟表示的完整2200步、原895步反例、索引/时钟异常、中断尾段、重复wait超时，以及采集到保存审阅的实际代码调用。没有测试子agent；这些是合成检查，不是物理采集或G5通过。来源/历史/实际缺批准拒绝核对见[准备报告](../reports/nav043_motion_probe_recovery_preparation_v2.json)。

具体恢复包仍是[PROPOSED](../config/nav043_motion_probe_recovery_proposal_v2.json)，pin为`1a5afe8d2b801a105a714cdb742cbbdf476d49a2eb376e485adc344a5e4c984f`；[支持清单](../config/nav043_motion_probe_recovery_readiness_v2.json)为PREPARED_ONLY，pin为`6093d64f9d889e8fc8f21803a47a81c8f7d7192f292d6c1bfde595ec448743b8`，都不是APPROVED。

拟只增加1次应用：在原保留目录下的`recovery_v2`子目录，重新完整采M01一次，再按原顺序采尚未开始的M02–M12；每例仍1秒warmup加10秒测量，单环境、固定v2、tiger11/tmux和物理GPU6。原M01不覆盖，也不能把原部分测量当完整工况。

累计应用上限由1变2，累计显式step上限仍32000：原895次，加新包最多31105次；新包26400测量/预热步和最多4705构建/复位/清理显式步。原生内部初始化步数仍不可观测，不写0。

累计墙钟上限仍2小时。原launcher报告104.594秒，但退出存在缺口，因此采用从原OWNER时间到保存审阅确认两PID均不存在的保守上界，再向上占用900秒；新应用最多6300秒（105分钟），包含启动与清理。不能只扣104.594秒而假设其它清理时间为0。

实际入口为`run_nav043_motion_probe_recovery_v2.py`，一次性包装为`.worklog-testing/NAV-043/motion-recovery-v2/launch-once.sh`。缺独立`nav043_motion_probe_recovery_approval_v2.json`时，各入口在GPU检查、应用创建及输出/锁创建前拒绝。取得新增一次应用的明确批准后，仍须先预检资源/身份/来源并提交，再执行一次；首个新实际失败保存停下，不自动重试或换目录。

本次没有第二应用、新物理步、拟合、导航、W查询、网络更新或校准。NAV-043=false、计划1.5.0/55/67历史完成和G5未过保持；旧45秒生产链及固定v2/功率/足迹/alpha/delta和数据锁不改。12工况完成后也只支持下一步运动模型范围评审，不能直接宣称90秒或部署通过。
