# EECR submission metadata draft

Title: Engineering Minimum-Cost Reach-Avoid Control for Quadruped Navigation

Authors (order copied from CCDC):

1. Haoyang Cao. Institute of Software, Chinese Academy of Sciences; University of the Chinese Academy of Sciences. Beijing, China. caohaoyang24@mails.ucas.ac.cn
2. Yuhua Guo. Space Star Technology Co., Ltd, China Academy of Space Technology; State Key Laboratory of Space Information System and Integrated Application, Beijing Institute of Satellite Information Engineering. Beijing, China. guoyh@radi.ac.cn
3. Bai Xue. Key Laboratory of System Software, Institute of Software, Chinese Academy of Sciences. Beijing, China. xuebai@ios.ac.cn

Corresponding author: PENDING AUTHOR CONFIRMATION. Do not infer from author order or the inquiry sender.

## Abstract (copied from CCDC, typographic escapes normalized)

The deployment of minimum-cost reach-avoid control on a physical quadruped robot is investigated. The challenges of coupled locomotion-navigation learning and model-execution mismatch are addressed through hierarchical control and two data-calibrated margins. Locomotion stabilization is separated from high-level navigation learning by a pretrained, frozen velocity tracker. Before high-level learning, nominal budget accounting is corrected by the cost margin, and the unsafe set is expanded by the geometric margin. Across three navigation scenarios, a success rate of 73.3% and a budget violation rate of 9.0% are obtained with the resulting calibrated minimum-cost reach-avoid (CaMRA) system, compared with 70.7% and 14.0% for uncalibrated RC-PPO. The distinct roles and deployment limits of these corrections are examined through ablation, held-out calibration, and physical-robot studies. An execution-calibrated pipeline is established to connect nominal minimum-cost reach-avoid planning with quadruped closed-loop execution.

## Keywords

quadruped navigation, reach-avoid control, safe robot learning


## Preparation status

Copied from the CCDC author information, abstract and keywords. This is a local field-entry aid, not an official EECR title page or a signed declaration. Corresponding author and any required approvals remain to be confirmed.

On 14 November 2026, OpenAI Codex assisted with downloading EECR templates, copying the existing LaTeX manuscript into an independent directory, local compilation, verification scripts, and administrative drafts. Substantive manuscript text, numerical results and figure assets were preserved. Historical AI use and data provenance were not established by this preparation; any required disclosure must reflect the authors' actual use.
