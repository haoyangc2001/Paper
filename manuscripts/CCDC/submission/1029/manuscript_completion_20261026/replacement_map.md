# 真实数据替换清单

正文已经是完整内部模拟稿，以下内容不可因为句式完整而视作已测事实。替换时同时核对定义、配置、分母、区间及结论，而非只改数字。

| 对象 | 当前输入 | 正文/图表消费者 | 真实替换要求 |
|---|---|---|---|
| 三网络和低层 | protocol.low_level / high_level；method_contract | IV-B/IV-D、附录A/C、图一网络说明 | 核对实际观测、动作裁剪前概率、共享停止、loss符号、频率、reward、随机化和权重冻结；当前代码核对只证明RC-PPO原实现。 |
| cost及单位 | protocol.measurement.cost | III-B、附录B、所有Cost/BVR、图三 | 替换逐步cost及正时间基项；若改为真实电能须重新计算所有cost、budget和alpha，不保留a.u.数值。 |
| footprint及delta | protocol.measurement.geometry | CR/MC、校准、图一/三/四 | 核对sim disc与hardware实际包络；只膨胀一次；不能按各方法自己的规划图统计碰撞。 |
| 样本/统计 | protocol.statistics | V-A、附录D、图二、各表 | 当前每场景300、每方法900；保留事件交叉关系。训练种子不确定性需要新分层方案，不能用当前条件区间冒充。 |
| 主比较 | main_results.json；continuous_sd.csv | 摘要、表一、图二、消融FULL/ZERO、预算ratio1、裕度四锚点 | 用真实场景计数/样本计算中心及CI；重新等场景汇总；图二empirical模式不再强制旧中心。 |
| 图二区间 | 各图data/manuscript_completion_20261026 | 图二及附录D | 旧mc_extension_v2是人工范围，新版是合成充分统计量算出的CI；二者均不是实测。实测换真实均值/方差/计数或合适分层bootstrap。 |
| 轨迹示例 | 轨迹项目data/versions/v2 | V-B、图三、总览示意 | 同初始条件/命令的配对日志；重新积分nominal和executed，保留双账本区别；当前示例两方法都未超预算。 |
| 校准及留出 | 校准项目paired_rollouts.csv / protocol | IV-C、V-C、附录B、图一/三/四中alpha/delta | 真实配对全窗口、20条件分配、排除规则；校准后冻结，留出只检验；重算Q90/并列覆盖/联合覆盖/分层bootstrap。若alpha/delta改变，高层重训及所有锚点受影响。 |
| 消融 | ablation_budget_results.ablation | 表二、图五、V-C及裕度四锚点 | 完整/双零复用同条件主比较；各删项独立重训；不要重新加无分层行。数字更新要同步所有差值标注。 |
| 预算曲线 | 预算项目新版96点与追加审查 | V-D、图六、附录C-C | 区分选择集与最终响应；ratio1共享主表，RA固定轨迹；重算三条件最小通过网格，不能连续插值声称最优。 |
| 裕度扫描 | margin_sweep.csv / results | V-D中大裕度结果 | 每设置重训、同低层、同预算；替换80指标，四锚点复用；不能用插值补实验。 |
| 扰动 | protocol.perturbation / results | V-E | 核对25%相对名义的幅值定义；三场景各300、条件分配和固定策略；重新计算两方法五指标。 |
| 硬件 | protocol.hardware / hardware_results | V-F、附录A/B/D、图七说明 | 核实真正的机器/传感/算力/频率、15m/60s/230预算、30例交叉计数与均值；旧照片仅支持定性展示。换真实照片时核对轨迹与统计是否同批。 |
| 作者与投稿 | 原匿名模板 | 标题；总页数 | 作者/单位/邮箱不推断；当前10页超过8页上限，另行压篇而非改官方字号/页边距。 |

## 推荐替换顺序

先固定真实成本、几何、控制接口和终止协议，再整理配对校准及留出，随后高层训练/独立预算选择/最终评估。结果替换按主比较→共享锚点→消融/扫描→扰动/硬件→正文结论→图表重绘→全稿审查进行。

单一数字改动示范已运行，见replacement_check.md。示范只改临时副本，不生成另一份main_preview，也不覆盖正式输入。当前审计脚本中的固定期望值是本轮内部示例的回归基线；真实替换必须有版本化协议和新证据，不能为过检查直接删除断言。
