# PDF版面复查（TASK-021）
构建命令：CCDC/scripts/build-1020.ps1。PDF技能authoring marker edit/1/pdf已在首次编译前成功运行一次。
工具：已有TeX Live 2026、Poppler pdftoppm、Conda Python/PyMuPDF；未安装依赖。
编译成功，三遍LaTeX及BibTeX完成；最终main.log无undefined、multiply defined、Overfull。7个既有Underfull hbox保留，数量与badness值与原稿相同，无新增。

页数：修改前10，修改后10；已有8页上限问题仍存在，未修改模板、字号、栏宽、图尺寸或压缩无关内容。
位置：表III在第6页左栏底，介绍段后，真机失败分析在同页右栏；照片仍在右栏上部。单栏表未进入Limitations或Conclusion。
参考文献第7页，附录第8页开始；新附录统计在第10页。第7页右栏及末页尾部空白来自既有双栏和附录另页规则，不做超范围重排。

视觉检查：
- 第6页：表题/表头、两行数值、浅蓝背景和粗体清楚；无裁切、重叠或异常拉大标题间距。照片原样。
- 第7页：结论续行和参考文献顺序正常，无引用缺失；附录新页符合约定。
- 第8--9页：附录公式、栏间距与章节转接正常。
- 第10页：新增协议与区间文字连续，左右栏无越界；跨栏续文正常。
- 第1--5页：与HEAD原PDF逐页渲染像素哈希相同，不需重排或修改。

预览：
- C:/Users/smile/Desktop/paper/CCDC/output/1020/review/hardware_comparison_20261029/page-06.png
- C:/Users/smile/Desktop/paper/CCDC/output/1020/review/hardware_comparison_20261029/page-07.png
- C:/Users/smile/Desktop/paper/CCDC/output/1020/review/hardware_comparison_20261029/page-08.png
- C:/Users/smile/Desktop/paper/CCDC/output/1020/review/hardware_comparison_20261029/page-09.png
- C:/Users/smile/Desktop/paper/CCDC/output/1020/review/hardware_comparison_20261029/page-10.png
保留本次5张页面预览作为验收证据，位于output/review/hardware_comparison_20261029；当前为未跟踪文件，未自动加入Git。
论文PDF：CCDC/output/1020/main.pdf。
