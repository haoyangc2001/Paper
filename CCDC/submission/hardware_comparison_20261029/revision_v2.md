# v2：场景依据与全组结果重审
日期：2026-10-29。用户质疑真机86%并授权重新分析修整。TASK-023--029为新增任务，原001--022和review_A--E等仅代表v1历史。
边界：两个方法各50、同15m/60s/230a.u.路线预算、低层50Hz/高层5Hz及alpha=.18/delta=.12不变；不修改仿真，不运行真实实验，不提交Git。

## 为什么重审
v1参考旧26/30=86.7%，但旧值也为作者示例，不能构成验证证据。固定场景不等于已证明容易；v1的完整CR4%甚至低于普通仿真CR8%，解释依据不足。v1数学兼容通过不等于成功率可信。
真实成功率无法由现有合成仿真数据唯一推断。v2只是难度假设更明确的另一组示例，不声称是预测、测量或被校准过的概率，也不规定真机SR必须低于仿真均值。

## 可用的相邻任务参照
源：CCDC/data/manuscript_completion_20261026/main_results.json与protocol.json、main.tex 5.1--5.5。
完整/RC-PPO：普通SR85/84、CR8/7；窄道SR73/72、CR15/15；密集SR62/56、CR22/25；增强执行扰动聚合SR65/58、CR22/27。不同条件不可池化，也不可拟合统一折减比例。
硬件包含窄道、细椅腿、移动障碍和定位失效；不再采用“普通固定室内很容易”作为主要依据。
几何：硬件额外通道宽约.30m，delta在两侧共.24m，居中名义剩余余量约.03m/侧。转身时.70×.32m轮廓限制更强。仿真半径.20m、通道.70--.95m，对应余宽.30--.55m。硬件不应被无理由视为更宽松。
时间：约15m/60s要求平均推进至少.25m/s（不含绕行）；仿真窄道起终点距离4--6m/30s。60s并不能单独证明更容易，等待与绕障会进一步消耗时间。两种距离含义不同，不用此比值计算SR。

## 修订方式
先逐项审查接触、定位中止、超时和独立接管，成功作为剩余计数；再独立审核预算交叉、Cost和MC的均值/样本SD。先完整后基线，过程中回看全部已确定指标。
保留共同定位瓶颈，不机械扩大方法优势；允许完整方法Cost较高。每项v2条目通过revision_of指向v1，保留全部历史。任何更低数值本身不代表更“真实”。

## 基线与备份
开始时工作区为上轮未提交变更，无新用户冲突；新表单栏，原PDF10页。
v1_results与v1_validation保存在CCDC/data/hardware_comparison_20261029/history/；原28行台账仅追加不改写。
本轮修改前关键文件与PDF页图哈希：
[
  {
    "method": "HD-MCRA",
    "scenario": "moderate",
    "metric": "SR",
    "estimate": 85.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "moderate",
    "metric": "CR",
    "estimate": 8.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "moderate",
    "metric": "BVR",
    "estimate": 5.0
  },
  {
    "method": "RC-PPO",
    "scenario": "moderate",
    "metric": "SR",
    "estimate": 84.0
  },
  {
    "method": "RC-PPO",
    "scenario": "moderate",
    "metric": "CR",
    "estimate": 7.0
  },
  {
    "method": "RC-PPO",
    "scenario": "moderate",
    "metric": "BVR",
    "estimate": 8.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "narrow",
    "metric": "SR",
    "estimate": 73.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "narrow",
    "metric": "CR",
    "estimate": 15.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "narrow",
    "metric": "BVR",
    "estimate": 9.0
  },
  {
    "method": "RC-PPO",
    "scenario": "narrow",
    "metric": "SR",
    "estimate": 72.0
  },
  {
    "method": "RC-PPO",
    "scenario": "narrow",
    "metric": "CR",
    "estimate": 15.0
  },
  {
    "method": "RC-PPO",
    "scenario": "narrow",
    "metric": "BVR",
    "estimate": 14.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "dense",
    "metric": "SR",
    "estimate": 62.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "dense",
    "metric": "CR",
    "estimate": 22.0
  },
  {
    "method": "HD-MCRA",
    "scenario": "dense",
    "metric": "BVR",
    "estimate": 13.0
  },
  {
    "method": "RC-PPO",
    "scenario": "dense",
    "metric": "SR",
    "estimate": 56.0
  },
  {
    "method": "RC-PPO",
    "scenario": "dense",
    "metric": "CR",
    "estimate": 25.0
  },
  {
    "method": "RC-PPO",
    "scenario": "dense",
    "metric": "BVR",
    "estimate": 20.0
  }
]

Path : C:\Users\smile\Desktop\paper\CCDC\latex\1020\main.tex
Hash : AD0DB833FAC03534A7DF46127196872C1E8716EBCF3DC016A68B71A013B09A82

Path : C:\Users\smile\Desktop\paper\CCDC\data\hardware_comparison_20261029\inference_ledger.jsonl
Hash : 84667B1CFAC36D97064E47BD8E03C8BF08A4993098851E5236E5CCC7A34B8A1C

Path : C:\Users\smile\Desktop\paper\CCDC\data\hardware_comparison_20261029\protocol.json
Hash : 8A414F85DAAADDE0D7B8ED4F4B9DB4ED107F119410FD985093E99B767DE40004

{"pages": 10, "rasters": ["122f4052971bf1e229a05f680bc57171ea5ff6d9e595d8a25bf608e0032ea58c", "05dd649c253250a3e82cc02429e21dda88d9c6602a17e3ed1554d2a5d57d28a6", "3f92c53253b18751eca93b10a7b6480765ba5e035d87a388934a905945bf71f8", "ed67799bb8db47ad085155f71e23f810b66bdac173c389552068476efa86c52d", "f8e60489fd50883732ea32785b0df890aa179036dd4fe54ac24edecdbb8c2260", "6ba63154cfae9721ea6e7cc32c116af22731f64c07065d5fcb9ea3c519966e59", "2edd7df9967a65b592e6d94925ea704a4a68eeb11a26f1abd303de7bccbb8506", "d0cd5871432eac1135880a2687bfaf7ecd7dadd3c3c9ba6499f07dcb626e4bd1", "b738ffe0e740d490350fa258d1af3e4da201f401c330d8f1aeee4243a28d0698", "a497107f05fd8859f01dcec9a96f6a981f0bd4114c7fc40e3eab98d11498bda3"]}

## 完整方法阶段复查（TASK-025）
14项v2记录已逐项追加。先审查8接触、3定位、4超时和0独立接管，成功补集35/50=70%。CR16%，其余失败14%；这是相邻难度的示例判断，不是73%与65%的插值公式。
BVR=7/50=14%：3成功、3超时、1晚期接触；另1超时可在216--230内，不再假设超时必超预算。
Cost182/SD54，MC.086/SD.057。相较v1，失败截断增多可拉低成本均值同时增大离散度；净空零点从2变8，不能保留原MC而不复查。

构造性检查（仅证明矩与计数兼容，不产生试验日志）：
{
  "cost": {
    "tail_count": 7,
    "tail": 250,
    "within_timeout": 223,
    "within_count": 1,
    "remaining": 42,
    "center": 169.6904761904762,
    "lower": 120.2962948793917,
    "upper": 219.0846575015607
  },
  "MC": {
    "zeros": 8,
    "low_count": 14,
    "low": 0.0373670693353895,
    "high_count": 28,
    "high": 0.13488789390373382
  }
}
成本：7个250，1个223分给未超预算超时，其余42个对称两值。可把8次接触中的1次分给250组、其余分给低组。
净空：8个0，其余14个低值和28个高值，均正且小于.15m，满足均值和样本SD。
因此计数、预算与净空可同时分配，没有发现需要回写输入的矛盾；所有判断仍需真实数据替换。

## 基线逐项审查
R2-01--14均完成。10接触+3定位+5超时+0接管，成功为32/50=64%；CR20%。与完整相比，3个额外失败来自2个接触和1个超时，共享定位中止未增加。
BVR11/50=22%：5成功、4超时、2接触；1超时未超预算。成本176/SD62，净空.067/SD.056。
{
  "cost": {
    "tail_count": 11,
    "tail": 255,
    "within_timeout": 223,
    "remaining": 38,
    "center": 151.89473684210526,
    "lower": 101.78549330272779,
    "upper": 202.00398038148273
  },
  "MC": {
    "zeros": 10,
    "low_count": 20,
    "low": 0.03436591855668469,
    "high_count": 20,
    "high": 0.13313408144331532
  }
}
证人仅检查矩/尾部/零点兼容，不作为传感器数据。SD不必全部上升：净空均值下降和正值压缩允许基线MC的绝对SD从.058略降.056，而相对离散度仍较高。

## 跨方法与统计复查（TASK-027）
v2中心值（SR/Cost/CR/BVR/MC）：
- 完整70% / 182a.u. / 16% / 14% / .086m。
- 零裕度64% / 176a.u. / 20% / 22% / .067m。
不是将两行按同一比例折减；计数、尾部和矩均有独立复核。相同成功差6个百分点仅对应3个终止结果差异，不因保留差值宣称精度或显著性。
不再无依据宣称场景容易；两方法SR较窄道示例低，CR处于窄道与增强执行扰动量级；不同场景不池化，也不把这种排序当成真实性检验。

检查器修正：去除43/50、40/50和旧失败类别硬编码；校验台账revision_of的字段/旧值；所有率区间、连续区间均从当前输入验证。允许非超预算超时，纳入216基础成本下界。
更强的成本证人把非超预算成功和早期失败分组，而非仅对所有值对称分配：
- 完整：7个250；1超时223；10个提前失败81.3315；32个成功197.3027。
- 基线：11个255；1超时223；11个提前失败73.3886；27个成功183.8787。
所有成功成本超过15m代理成本理论下界98.5901；晚期接触可分配至高尾，接触净空始终0。这里的分组只是可实现性证人，不是生成或观测到的运行样本。
MC证人由固定均值/SD/零点搜索可行两点分布，检查正净空≤.15m；不由脚本决定输入或模拟噪声。

小批Wilson0/25/50及t49数值积分通过；两方法全部计数/交叉关系/10指标区间通过。
原28行台账字节哈希未变；28条v2追加均有revision_of；v1快照与原台账对应；17个保留文件哈希不变。无未决数据矛盾。

## 稿件与说明同步（TASK-028）
5.6表III仍单栏，顺序与主表一致；完整方法浅蓝，最低Cost仍按实际值加粗基线176。
正文成功35/50与32/50，其他失败4超时8接触和5超时10接触，共同3定位中止；BVR22→14、Cost176→182。
附录逐项同步全部10组区间、SD、MC精确值以及7/11超预算交叉计数；明确每方法另1超时未超预算。表中MC .09/.07仅为两位显示，不能反推出实际差值.019。
当前数据README、论文README、替换映射及计划说明均标记v2，delivery_review.md保留首版历史并指向本文件；v1原始结果、原28条台账和历史审查保留。
只读--check-tex通过，核对10表值、成功/失败文字、全部区间及预算交叉；原主稿真机节之前的内容逐字符不变，17个保护文件哈希不变，git diff --check通过。

## 编译、视觉复查与交付（TASK-029）
使用现有 build-1020.ps1 完成 pdflatex/BibTeX/两轮 pdflatex，退出码0。PDF为10页，9476599字节；参考文献仍在第7页，附录仍自第8页开始。本轮没有扩大页数，但当前含附录版本仍超过已记录的8页投稿上限，不能将编译成功等同于满足投稿页数要求。

最终日志保留7处原有 Underfull hbox，没有 Overfull、未定义引用、重复标签或新编译警告。逐页72dpi栅格哈希对比v1：仅第6和第10页改变，其余8页逐像素一致。Poppler以125dpi渲染并检查两个变化页：表III位于真机小节第6页左栏下方，Fig.7仍在右栏上方，计数和表值清晰；附录第10页的新区间及交叉计数无截断、叠字或溢出。最后一页右栏保留原有短栏布局，未为数据修订改动模板。

视觉证据：
- CCDC/output/1020/review/hardware_comparison_20261029/v2/page-06.png
- CCDC/output/1020/review/hardware_comparison_20261029/v2/page-10.png

最终 --check-tex 审计和 git diff --check 通过；保留文件、原始台账与v1快照检查通过。结果仍为合成示例，empirical_validation=false，表题保留统一Synthetic说明。没有新实机/仿真运行，没有Git提交或推送；HEAD保持5d0d35e98cce15652c3f07d19ba9d9fbf8806550。
