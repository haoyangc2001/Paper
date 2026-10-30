# 裕度校准与独立检验图：参考原图

本目录保存本轮调研的公开原文、相关页文本、原图截取和出处。没有生成校准或检验假设数据，没有修改论文或已有绘图数据。

## 参考资料

| 资料 | 版本、原图及阅读页 | 借鉴内容 |
| --- | --- | --- |
| Deep Reinforcement Learning at the Edge of the Statistical Precipice | arXiv:2108.13264；复用既有原文，PDF 第7页，Fig. 7 | 成对分布曲线、共享图例、轻网格和参考横线。原图为超阈值比例，拟绘图为不超过误差阈值的比例，曲线方向不同。 |
| Safe Planning in Dynamic Environments using Conformal Prediction | arXiv:2210.10254v2；PDF 第10页，Fig. 2 和相邻实验说明 | 误差直方图与阈值；独立划分校准和测试数据。其 ORCA 案例使用2,000组校准、500组测试，文中报告498/500测试轨迹满足所检验条件。这是参考论文的结果，不是本论文结果。 |
| Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners / KnowNo | arXiv:2307.01928v2；PDF 第6页，Fig. 3 | 直接报告目标水平与实际统计值的偏差。借鉴其问题表达，不照搬多方法柱状图或将其任务成功率等同本论文误差覆盖率。 |
| A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | arXiv:2107.07511v6；PDF 第4页 Fig. 2、第15–16页 §3.3 | 分位数阈值的解释、独立验证覆盖率、有限样本波动。其共形修正和理论保证不是本论文当前普通经验分位数方法的现成保证。 |

原文 URL、文件哈希和读取路径见 `source_manifest.json`。原图裁剪位置、页码和图号见 `figure_cases.json`。

## 原图入口

- `figures/reference_cases.png`：三组案例速览。
- `figures/rliable_fig7.png`：分布曲线参考。
- `figures/safe_planning_fig2.png`：误差分布与阈值参考。
- `figures/knowno_fig3.png`：目标与实际偏差参考。
- `previews/`：相关原文完整页，使用 Poppler 渲染以检查图注与版面。
- `text/`：带 PDF 页码的原文抽取文本。

原图仅供内部阅读和设计参考，不插入本论文，不作为本论文实验结果。

## 复查

```powershell
& 'D:/codesoftware/Miniconda/python.exe' './CCDC/references/calibration_review_20261025/review_calibration_references.py'
& 'D:/codesoftware/Miniconda/python.exe' './CCDC/references/calibration_review_20261025/review_calibration_references.py' render
```

第一个命令只在原文副本不存在时下载，并重新提取文本；第二个命令重绘参考截图片，不产生本论文数据。
