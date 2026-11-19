# EECR 2027 官网要求与交接

核查日期：2026-11-14。以此次重新下载的官网页面及官方模板为依据，文件索引见official_sources/index.json和../../latex/template/download_manifest.json。

## 时间、地点与渠道

| 项目 | 官网安排 |
| --- | --- |
| 会议 | 2027 13th International Conference on Electrical Engineering, Control and Robotics |
| 会议地点 | 深圳；场地页列深铁塘朗城君璞酒店（GENPLA HOTEL SHENZHEN NANSHAN） |
| 会期 | 2027年4月23–25日 |
| 全文截止 | 2026年12月10日 |
| 录用通知 | 2027年1月10日，早于用户要求的3月1日；属于计划而非个案保证 |
| 注册截止 | 2027年1月30日 |
| 在线投稿 | https://easychair.org/conferences/?conf=eecr2027 |
| 咨询邮箱 | eecr_conf@vip.163.com |

[投稿页](https://www.eecr.org/sub.html)要求通过在线系统提交。邮箱作为咨询渠道，不把发送咨询视为正式投稿。不必为全文投稿先单独提交摘要；仅摘要录用只能报告，不会出版，不能代替全文录用。

## 要求及迁移结果

| 项目 | 官网依据 | 当前结果 |
| --- | --- | --- |
| 模板 | 投稿页链接LaTeX ZIP及Word模板 | 均已从官网重新下载，原包保留；LaTeX原包另解压保存 |
| 排版 | 官方LaTeX样例使用IEEEtran conference；Word模板明确US Letter | 现稿为IEEEtran、双栏、10pt、US Letter；官方类文件与CCDC、ICACAR逐字节相同 |
| 语言 | "Papers should be written in English" | 保留原英文稿 |
| 页数 | "limited to 4-10 pages" | 当前7页，符合公开页数范围 |
| 原创 | "Submitted papers are expected to contain original work executed by the authors..." | 原稿内容原样迁移，实际原创、数据来源及论断仍由作者核实 |
| 同时投稿 | 已查首页、投稿页、征稿页、注册页及模板未找到明确禁止或允许同时在审的声明 | 不能认定允许；同稿当前仍在CCDC审稿，咨询草稿已披露这一事实 |
| 匿名与标题页 | 已查公开页面未明确要求匿名主稿或独立标题页 | 准备署名核对稿和投稿字段草稿；没有擅自引入ICACAR的特定要求。最终依系统与编辑说明核定 |
| 参考文献、摘要、数值及图表 | 没有查到需要改写这些内容的EECR特定要求 | 保持原文、原格式、原顺序及原数值；不因一般风格建议改写内容 |
| 出版与检索 | 2027首页写收入EECR 2027会议论文集；官方CFP称EI/Scopus等检索；2026历史页链接IEEE Xplore | 2027出版方未由已取得文字明确，不能由2026沿用推断，也不承诺检索或实际发表日期 |

官方链接：[主页](https://www.eecr.org/)、[征稿范围](https://www.eecr.org/cfp.html)、[投稿](https://www.eecr.org/sub.html)、[注册](https://www.eecr.org/reg.html)、[联系](https://www.eecr.org/contact.html)、[场地](https://www.eecr.org/venue.html)、[2026历史页](https://www.eecr.org/2026.html)。

## 费用（依据官网图片人工读取）

[官网费用表](https://www.eecr.org/assets/images/fee.png)保存为official_sources/registration_fee.png，SHA256为37648d66f16968f34b6970182c2cd64bbd693cdfa6287affc9c95db252eeb702。表中美元与人民币是各自列示的价格，不能当作汇率换算。

| 类别 | 基础费（最多6页） | 本稿额外1页 | 本稿合计 |
| --- | --- | --- | --- |
| 普通作者 | 4050元／560美元 | 450元／60美元 | 4500元／620美元 |
| 学生作者 | 3600元／500美元 | 450元／60美元 | 4050元／560美元 |
| IEEE/TPC会员 | 3600元／500美元 | 450元／60美元 | 4050元／560美元 |
| IEEE学生会员 | 3450元／480美元 | 450元／60美元 | 3900元／540美元 |

学生价限全日制学生第一作者，并需有效在读证明；本次没有代作者确认资格。至少一位作者需完成录用论文注册。退款政策为距会期60天以上退70%、30–60天退50%、30天内不退；具体执行以会议确认及注册条款为准。当前没有付款。

## 原样迁移与核查

来源：manuscripts/CCDC/latex/1029/main.tex、references.bib及images/1029中正文使用的8份资源。main.tex只有会议说明注释和8个图片路径改变；反向还原后与原文件字节一致。bib与图像哈希完全一致；迁移差异保存在migration_source.diff。

已通过新EECR独立路径构建。生成7页PDF，23篇参考文献；逐页文本与108dpi渲染均与既有CCDC PDF相同。字体均已嵌入，没有越界文本块、未定义引用或overfull box。关闭shell escape导致的epstopdf提示不影响本稿已有PDF/JPG/PNG资源。PDF文件哈希因创建元数据可能不同，内容一致通过文本及渲染分别验证。

保护清单记录CCDC与ICACAR共690个原文件，核查未发现变动。官方页面、模板和费用图也保存了下载时间、网址及SHA256。构建/核查入口与结果见README及output/1114/qa/verification.json；人工版面记录见visual_review.md。

## 正式提交前仍需核实

1. 同稿正在CCDC审稿。EECR网页未明示禁止不等于允许；应取得关于并行审稿的明确回复，并同时满足CCDC要求。[IEEE官方要求](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/ethical-requirements/)写明等待同行评审决定时不要提交其他出版物。仅告知编辑或单边同意不足以豁免另一会议要求。
2. CCDC来源README及此前ICACAR交接均注明部分统计仍为内部推断示例，需用真实实验记录核实。此次按用户要求保持原值，迁移核验不能证明这些数值或科学结论真实有效。
3. 对照投稿系统核定是否匿名、所需文件格式、通讯作者、作者及单位批准、利益冲突、经费和AI披露等实际字段；没有要求的内容不凭空补声明。未登录系统，未代勾保证。
4. 核实2027最终出版方及学校对全文录用通知的认定。官网通知日期不是保证提前发表或检索。

editor_inquiry_draft.txt为政策咨询草稿，不附论文，不视为投稿，尚未发送。本次未投稿、未发邮件、未注册或付款、未撤回CCDC，也未提交Git。新版目录是授权的本地材料准备，不是投稿批准或实验结果审计。
