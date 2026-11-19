# EECR 六页正文精简交付（2026-11-18）

已按用户本次明确授权，在16篇引用的1118版本上建立独立正文修订 `1118_concise`。最终PDF恰好6页，包含全部16篇参考文献。原1118、1114、CCDC和ICACAR稿件保留。

压缩采用36块句法和重复说明整理，重点是引言动机、相关工作、分层控制/校准流程及实验导语。PDF提取词数约从5506变为4628；该数受换行连字符影响，仅作篇幅参考，不是学术内容量的判据。摘要、关键词、作者、全部7组展示公式、7幅图/8个图像文件、3张表和图注逐字保持；正文全部数值的写法与出现次数不变，16篇文献数据库不变。

## 当前入口

- 主稿：`manuscripts/EECR/latex/1118_concise/main.tex`
- PDF：`manuscripts/EECR/output/1118_concise/main.pdf`
- 逐段语义/逻辑核对：`manuscripts/EECR/output/1118_concise/qa/semantic_review.md`
- 逐页目视核对：`manuscripts/EECR/output/1118_concise/qa/visual_review.md`
- 源码与PDF检查：`manuscripts/EECR/output/1118_concise/qa/verification.json`
- 本目录保留完整差异、36项原文/改文对照、开工哈希和构建日志。

构建：`& ./manuscripts/EECR/scripts/build-1118_concise.ps1`，通过latex-paper-en的compile.py，默认禁用shell escape。

验证：`& D:/codesoftware/Miniconda/python.exe ./manuscripts/EECR/scripts/verify-1118_concise.py`。

## 验收与边界

源码按36项记录可精确重建；受保护旧文件检查未发现异常修改。官方类、10pt、Letter、双栏与原有页面设置不变。无未定义引用、无Overfull，字体嵌入、页外文字检查及6页目视核对通过。

语义核对尤其保留：经验覆盖不等于成功概率、不作随机严格安全保证；同一冻结接口/配对校准；双critic相同停止时域与actor固定后重拟合；预算选择和最终评估分开；超预算继续评价、全部失败纳入分母；成本与成功率联合解释；单例不混入主表；扰动下不重拟合；硬件与平面结果不合并。

本次完成文字精简与原稿含义/格式一致性核对，没有运行新实验、改实验数字或新增科学保证。论文仍需要作者按既定投稿流程进行最终学术审阅；此次没有发送或提交稿件。
