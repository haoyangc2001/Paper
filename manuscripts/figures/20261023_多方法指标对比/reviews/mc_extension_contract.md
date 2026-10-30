# MC五列版数据契约与逐点来源映射

MC为全部试验全程最小物理机身间距的场景均值，单位m；非膨胀规划集距离，不是delta。左右端点是人工假设估计范围，不是逐试验极值、SD/SE/CI。原84点保留，新版JSONL为唯一点数据入口；三个场景等权，不平均端点。

| ID | 场景 | 方法 | MC(m) | SR(%) | Cost | CR(%) | BVR(%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MC001 | moderate | HD-MCRA | 0.40 | 85 | 64 | 8 | 5 |
| MC002 | moderate | RC-PPO | 0.38 | 84 | 67 | 7 | 8 |
| MC003 | moderate | RA | 0.43 | 76 | 76 | 6 | 12 |
| MC004 | moderate | PPO_beta_L | 0.29 | 87 | 70 | 11 | 11 |
| MC005 | moderate | PPO_beta_H | 0.24 | 57 | 92 | 16 | 16 |
| MC006 | moderate | CPPO_beta_L | 0.45 | 55 | 98 | 5 | 15 |
| MC007 | moderate | CPPO_beta_H | 0.34 | 89 | 75 | 9 | 10 |
| MC008 | narrow | HD-MCRA | 0.31 | 73 | 80 | 15 | 9 |
| MC009 | narrow | RC-PPO | 0.30 | 72 | 84 | 15 | 14 |
| MC010 | narrow | RA | 0.34 | 61 | 95 | 13 | 20 |
| MC011 | narrow | PPO_beta_L | 0.19 | 67 | 91 | 24 | 21 |
| MC012 | narrow | PPO_beta_H | 0.15 | 45 | 105 | 31 | 27 |
| MC013 | narrow | CPPO_beta_L | 0.36 | 39 | 118 | 14 | 22 |
| MC014 | narrow | CPPO_beta_H | 0.25 | 68 | 96 | 19 | 17 |
| MC015 | dense | HD-MCRA | 0.23 | 62 | 121 | 22 | 13 |
| MC016 | dense | RC-PPO | 0.22 | 56 | 124 | 25 | 20 |
| MC017 | dense | RA | 0.25 | 43 | 141 | 20 | 28 |
| MC018 | dense | PPO_beta_L | 0.12 | 49 | 138 | 34 | 30 |
| MC019 | dense | PPO_beta_H | 0.09 | 25 | 154 | 40 | 38 |
| MC020 | dense | CPPO_beta_L | 0.28 | 19 | 158 | 18 | 30 |
| MC021 | dense | CPPO_beta_H | 0.18 | 52 | 146 | 27 | 24 |

来源：CORL/scripts/generate_experiment_results_v3.py；21个中心与main_metric_wide工作簿及表一7个均值已核对。点ID固定，原P001至P084不重编号。五列顺序SR/Cost/CR/BVR/MC。MC采用两位小数，米；其他指标保留旧精度。端点有限、非负、覆盖中心，无概率式MC上限；中心禁止静默修改。
