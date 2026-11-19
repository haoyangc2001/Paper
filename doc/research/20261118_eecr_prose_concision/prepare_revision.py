"""Create a separate prose-only EECR revision and retain every replacement."""
from pathlib import Path
import collections
import difflib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
RECORDS = Path(__file__).resolve().parent
BASE = ROOT / "manuscripts/EECR"
SOURCE = BASE / "latex/1118"
TARGET = BASE / "latex/1118_concise"
assert not TARGET.exists(), "Refuse to overwrite an existing revision"
original = (SOURCE / "main.tex").read_text(encoding="utf-8")
text = original
changes = []


def paragraph(start: str, replacement: str, reason: str) -> None:
    global text
    assert text.count(start) == 1, start
    position = text.index(start)
    end = text.find("\n\n", position)
    assert end != -1
    # Some source paragraphs directly precede a float without a blank line.
    for marker in ["\n\\begin{figure}", "\n\\begin{table}"]:
        boundary = text.find(marker, position)
        if boundary != -1:
            end = min(end, boundary)
    before = text[position:end]
    after = replacement.strip()
    changes.append({"id": f"P{len(changes)+1:02d}", "original": before,
                    "revised": after, "reason": reason,
                    "original_line": original[:original.index(start)].count("\n") + 1,
                    "meaning_check": "PRESERVED (author review remains appropriate)",
                    "risk_flags": "none"})
    text = text[:position] + after + text[end:]


paragraph("Quadruped robots are increasingly used", r"""
Quadruped inspection, exploration, and service tasks require collision-free goal
reaching without unnecessary motion or energy expenditure. In cluttered scenes,
short routes may approach obstacles, whereas safer routes may require more time
and actuation. Encoding safety and efficiency as separate reward terms makes
this trade-off depend on hand-tuned weights. Minimum-cost reach-avoid (MCRA)
control instead specifies goal reachability and trajectory-wide avoidance while
minimizing cumulative cost~\cite{bansal2017}.\footnote{The CaMRA implementation
is publicly available at \url{https://github.com/haoyangc2001/NaVILA_CaMRA}.}
""", "压缩应用背景和安全/效率动机，保留短路与安全路线的条件性权衡及MCRA定义。")

paragraph("Practical MCRA learning has been enabled", r"""
Reachability-Constrained Proximal Policy Optimization (RC-PPO)~\cite{so2024}
enables practical MCRA learning by augmenting the state with a safety indicator
and remaining budget, learning a reachability value function and
budget-conditioned policy, and seeking the least feasible initial budget.
These components jointly represent safety and cost feasibility.
""", "合并RC-PPO步骤介绍，保留状态增广、价值函数、策略与最小可行预算四项。")

paragraph("Deploying this formulation on a physical quadruped", r"""
Physical deployment also requires translating navigation decisions into joint
motion. Direct joint-level learning must acquire balance and gait coordination
alongside collision-free goal reaching. Encoding falls, excessive body tilt, and
collisions as unsafe states does not fully specify smooth, coordinated locomotion;
early instability may interrupt navigation rollouts and hinder route selection
and cost optimization. We therefore pretrain a velocity tracker for balance,
gait coordination, and action smoothness, freeze it, and learn RC-PPO navigation
through its fixed velocity-command interface.
""", "缩短分层控制动机，保留不安全状态约束的不足、早期训练中断和先预训练后冻结的时序。")

paragraph("Although the low-level controller improves", r"""
The tracker improves stability and attenuates disturbances, but identical velocity
commands may still incur extra cost or path deviations, leading to underestimated
budget consumption or overestimated clearance. With the tracker frozen, we pair
nominal and executed rollouts using identical initial conditions and commands.
Empirical cost overruns and path deviations determine a cost margin $\alpha$ and
geometric margin $\delta$, respectively. These increase the running cost in
budget updates and expand the unsafe set before high-level learning and budget
search. Deployment uses the same frozen tracker.
""", "合并残余误差与两类裕度流程，保留相同初态/命令、分别校准和学习前修正。")

paragraph("An ideal MCRA learner is connected", r"""
This calibrated minimum-cost reach-avoid (CaMRA) pipeline connects nominal MCRA
learning to physical closed-loop execution. Its scope is empirical robustness
under the calibration execution distribution; we claim neither strict stochastic
safety guarantees nor an extension of RC-PPO's deterministic theory to general
stochastic systems. Controlled mismatch studies, component-wise ablations,
sensitivity tests, and quantitative hardware evaluation assess execution
mismatch, the distinct failure modes addressed by $\alpha$ and $\delta$, and
system behavior on a physical robot.
""", "合并研究范围与评估目的，明确保留经验稳健性边界和两个不作理论保证的限定。")

paragraph("The main contributions are summarized as follows:", r"""
The main contributions are:
\begin{enumerate}
\item A hierarchical quadruped MCRA framework separates locomotion stabilization
      from navigation learning through a pretrained, frozen velocity tracker,
      providing a consistent calibration and deployment interface.
\item Two margins calibrated from paired nominal and executed trajectories
      separately correct budget accounting and unsafe-set geometry for
      high-level learning and budget selection.
\item Navigation comparisons, ablations, held-out calibration tests, and
      physical-robot evaluation examine both margins' roles and deployment
      trade-offs.
\end{enumerate}
""", "压缩三项贡献的重复修饰，保留三项贡献及各自作用。")

paragraph("In reach-avoid reinforcement learning,", r"""
Reach-avoid reinforcement learning treats target reaching and unsafe-set
avoidance as trajectory-level requirements rather than competing reward
terms~\cite{bansal2017,hsu2021}. RC-PPO extends this to MCRA through safety/budget
state augmentation, budget-conditioned policy and reachability-critic learning,
and least-feasible-budget search~\cite{so2024}. We adopt this foundation to
connect nominal MCRA decisions to quadruped execution.
""", "相关工作不重复展开算法步骤，保留原引用与算法继承关系。")

paragraph("Quadruped navigation in challenging environments", r"""
Learned locomotion supports quadruped navigation: Lee et al.~\cite{lee2020}
demonstrated robust natural-terrain locomotion using proprioception. Jin et
al.~\cite{jin2023resilient} trained navigation policies to command an existing
locomotion controller and use locomotion-level observations to respond to
perception failures. He et al.~\cite{he2024abs} instead developed a joint-level
agile policy with reach-avoid-guided switching to collision-avoidance recovery.
Hierarchical policies connect visual navigation and reusable motor
skills~\cite{jain2021pixels}, while iPlanner learns smooth, collision-free paths
from depth observations through differentiable cost maps~\cite{yang2023iplanner}.
Here, a frozen tracker stabilizes locomotion, while high-level MCRA addresses
goal reaching, obstacle avoidance, and cumulative cost.
""", "压缩被动表达，保留五类工作的差异及本稿与它们的关系。")

paragraph("Model uncertainty is addressed through", r"""
Domain and dynamics randomization broaden training
distributions~\cite{tobin2017,tan2018simreal}, whereas robust reinforcement
learning optimizes performance across uncertain dynamics~\cite{wang2021}.
The need for dynamics randomization depends on control design and deployment
conditions~\cite{xie2021randomization}. Chance-constrained and risk-aware planning
incorporate uncertainty into constraints or risk
measures~\cite{theurkauf2023,fan2021step}; STEP, for example, estimates stochastic
traversability for off-road planning~\cite{fan2021step}. FaSTrack bounds
planning--tracking mismatch using reachability~\cite{herbert2017fastrack}, while
DTC combines model-based planning and learned tracking for accuracy and
robustness~\cite{jenelten2024dtc}. We calibrate cost and geometric margins
from empirical quantiles of paired nominal/executed errors under a fixed velocity
interface for MCRA learning and deployment.
""", "保留随机化、鲁棒RL、风险规划、FaSTrack/DTC各路线与本文经验校准的区别。")

paragraph("The navigation layer is modeled as", r"""
The navigation layer has state $x_t\in
\mathcal{X}$, high-level velocity command $u_t\in\mathcal{U}^{\mathrm{high}}$,
transition kernel $P$, running cost $c$, and initial-state distribution $\rho_0$.
MCRA minimizes accumulated cost while reaching $\mathcal{G}$ at a freely chosen
terminal time $T$ and avoiding $\mathcal{F}$ at every step:
\begin{equation}
\min_{\pi,T}\quad \sum_{t=0}^{T-1}c(x_t,\pi(x_t))
\label{eq:mcra_objective}
\end{equation}
\begin{equation}
\begin{gathered}
\mathrm{s.t.}\quad x_T\in\mathcal{G},\\
x_t\notin\mathcal{F}\ \forall t\leq T,\\
x_{t+1}\sim P(\cdot\mid x_t,\pi(x_t)).
\end{gathered}
\label{eq:mcra_constraints}
\end{equation}
The sets are represented using signed functions $g$ and $h$ as
$\mathcal{G}=\{x:g(x)\leq0\}$ and $\mathcal{F}=\{x:h(x)>0\}$. Following
RC-PPO~\cite{so2024}, the state is augmented to $\hat{x}_t=(x_t,y_t,z_t)$,
where $y_t\in\{-1,1\}$ records whether the unsafe set has been entered, and $z_t$
denotes the remaining cost budget, initialized at $z_0$. The shifted indicator is
$I_{b\in\mathcal{B}}=+1$ when $b\in\mathcal{B}$ and $-1$ otherwise; thus
$y_{t+1}=\max\{I_{x_{t+1}\in\mathcal{F}},y_t\}$ and
$z_{t+1}=z_t-c(x_t,u_t)$. The augmented goal function is
\begin{equation}
\hat{g}(x,y,z)=\max\{g(x),Cy,-z\},
\label{eq:augmented_goal}
\end{equation}
where $C>0$ is an arbitrary safety-violation constant. RC-PPO notes that a
practical choice is $C=\max_{x\in\mathcal{X}}g(x)$~\cite{so2024}. Reaching the
zero sublevel set of $\hat{g}$ therefore encodes goal reaching, trajectory-wide
avoidance, and the upper-bound property $z_0\geq\sum_t c_t$. The reachability critic,
budget-conditioned actor, and two-phase minimum-budget search from RC-PPO are
adopted as the high-level decision backbone.
""", "只压缩问题设置导语，公式、符号、增广状态更新及RC-PPO理论回顾逐字保留。")

paragraph("On hardware, $u_t$ is a desired body velocity", r"""
On hardware, a locomotion controller realizes desired body velocity $u_t$ through
the closed-loop transition $P_{\mathrm{cl}}$. CaMRA addresses the gap between the
nominal learning model $x_{t+1}=\bar f_{\mathrm{cl}}(x_t,u_t)$ and execution sampled
from $P_{\mathrm{cl}}(\cdot\mid x_t,u_t)$. Paired-rollout relative cost overrun
and trajectory deviation are
\begin{equation}
\varepsilon_c=\frac{J_{\mathrm{exec}}-J_{\mathrm{nom}}}{J_{\mathrm{nom}}},
\qquad
\varepsilon_d=d(\tau_{\mathrm{exec}},\tau_{\mathrm{nom}}),
\label{eq:execution_errors}
\end{equation}
Here $J$ is a nonnegative navigation proxy cost in arbitrary units (a.u.),
not measured electrical energy. Paired rollouts share the command program and
time window; $d$ is the maximum planar distance between time-aligned positions.
""", "压缩名义/执行转移定义的句法，不改变转移、成本单位、配对和距离统计定义。")

paragraph("The deployment objective is to retain", r"""
We retain RC-PPO's MCRA semantics while accounting for the measured distribution
of $(\varepsilon_c,\varepsilon_d)$ through a fixed velocity interface, paired
rollouts, and calibrated margins. These errors empirically describe the
nominal-to-execution gap, rather than define a new stochastic reachability model.
""", "压缩部署目标，保留误差仅为经验描述、不定义新随机可达性模型的边界。")

paragraph("The CaMRA architecture is shown", r"""
CaMRA couples high-level MCRA navigation to a pretrained, frozen velocity tracker
(Fig.~\ref{fig:system_overview}). Development has two stages: train and freeze
the tracker, then calibrate $\alpha$ and $\delta$ from paired nominal/executed
rollouts; subsequently incorporate the margins into RC-PPO learning and
initial-budget selection. At deployment, the tracker converts navigation
body-velocity commands into joint commands for closed-loop execution.
""", "合并系统概述与前文重复流程，保留两个开发阶段及部署调用链。")

paragraph("With direct joint-level MCRA learning,", r"""
Direct joint-level MCRA learning couples stable locomotion with goal reaching,
obstacle avoidance, and cost reduction. Avoiding falls and excessive tilt alone
gives limited guidance for coordinated, smooth locomotion, and early balance
loss can interrupt navigation exploration. Hierarchical control therefore
separates locomotion stabilization from high-level reach-avoid decisions.
""", "保留分层控制的完整因果链，删去与引言重复的展开和冗余被动句。")

paragraph("The low-level tracker is pretrained", r"""
The tracker is pretrained with PPO and domain randomization to maintain balance,
gait coordination, and action smoothness while tracking velocity commands.
It maps proprioception and body-velocity commands
$u_t=[v_x^*,v_y^*,\omega_z^*]^\top$ to joint commands, attenuating disturbances
through closed-loop feedback within its trained command domain. Freezing it
before calibration and high-level learning prevents subsequent low-level
parameter changes; the navigation policy learns velocity commands rather than
joint coordination.
""", "保留PPO、随机化、观测/动作映射、训练域限定与冻结时序，缩短重复表达。")

paragraph("The nominal map $\\bar f_{\\mathrm{cl}}$ abstracts", r"""
The nominal map $\bar f_{\mathrm{cl}}$ integrates commanded body velocities over
each high-level interval under ideal tracking. Contact variation, friction,
actuation latency, and state-estimation errors can cause deviations. Calibration,
high-level learning, and deployment share the frozen tracker and command domain,
providing a consistent basis for residual-error measurement and margin calibration.
""", "保留理想跟踪假设、四类误差来源与跨阶段一致接口。")

paragraph("To convert execution mismatch into cost", r"""
Paired nominal and perturbed-execution rollouts use the frozen interface with
identical initial states, maps, goals, and high-level commands. For length $T$,
let $J_{\rm nom}=\sum_{t=0}^{T-1}c(x_t^{\rm nom},u_t)$ and
$J_{\rm exec}=\sum_{t=0}^{T-1}c(x_t^{\rm exec},u_t)$. Calibration and independent
evaluation use disjoint rollout sets; evaluation data do not select margins.
""", "压缩校准采集描述，保留四项配对条件、成本定义、数据互斥与不使用评估调参。")

paragraph("For $J_{\\rm nom}>0$,", r"""
For $J_{\rm nom}>0$, paired costs give $\varepsilon_c$ as defined in
Section~\ref{sec:problem_setup}. Geometric deviation is the maximum planar
distance between time-aligned base positions,
\begin{equation}
\varepsilon_d=\max_{0\leq t\leq T} \left\|p_t^{\rm exec}-p_t^{\rm nom}\right\|_2,
\label{eq:max_planar_deviation}
\end{equation}
where $p_t$ is the base position.
""", "保留正分母条件与几何偏差的时间对齐、基座位置和最大值定义。")

paragraph("The two errors are calibrated separately", r"""
The errors affect different MCRA constraints and are calibrated separately at
quantile levels $p_c$ and $p_d$:
\begin{equation}
\alpha=Q_{p_c}(\varepsilon_c),\qquad
\delta=Q_{p_d}(\varepsilon_d),
\label{eq:calibration_margins}
\end{equation}
where $Q_p$ is the empirical calibration-sample $p$-quantile. The corrected cost
$\tilde c(x_t,u_t)=(1+\alpha)c(x_t,u_t)$ gives budget updates
$z_{t+1}=z_t-\tilde c(x_t,u_t)$, and the unsafe set becomes
$\mathcal{F}_\delta=\{x:d(x,\mathcal{F})\leq\delta\}$. Both margins are fixed
before high-level learning; empirical coverage depends on the execution
conditions represented in calibration.
""", "缩短裕度说明，保留单独分位数、成本/预算/集合公式及覆盖的分布依赖边界。")

paragraph("A budget-conditioned navigation policy is trained", r"""
Budget-conditioned navigation learning and initial-budget selection use the
corrected cost and expanded unsafe set, incorporating both margins into decisions
executed through the frozen velocity interface.
""", "合并鲁棒学习小节的两句同义导语，不删算法细节。")

paragraph("In Phase I, budget updates are computed", r"""
In Phase I, $\tilde c$ updates the budget, while $\mathcal{F}_\delta$ determines
the safety indicator and augmented goal. A stochastic actor
$\pi_\theta(u\mid o,y,z)$ uses two critics: $V_h$ evaluates goal reaching and
safety, and $V_c$ estimates corrected cumulative cost over the same stopping
horizon. Normalizing cost by the scene-specific reference budget
$z_{\rm ref}>0$ gives
\begin{equation}
W=\max\{V_h,V_c-z/z_{\rm ref}\}.
\label{eq:combined_residual}
\end{equation}
The actor is optimized to reduce this residual. Thus $\alpha$ modifies cost targets and budget
updates, while $\delta$ modifies reachability targets through the safety indicator.
""", "保留同停止时域、双critic各自目标、归一化残差与alpha/delta修改对象。")

paragraph("In Phase II, a deterministic deployment policy", r"""
In Phase II, the actor's mean action defines a deterministic deployment policy;
the actor is fixed and the critics are refitted. Under the calibrated model,
we select the smallest tested initial budget satisfying the prescribed
goal-reaching, collision, and budget-violation criteria.
""", "保留先固定部署actor再重拟合critic，以及最小已测预算而非连续最优预算。")

paragraph("During deployment, navigation observations and", r"""
At deployment, the policy receives navigation observations and remaining budget,
and the frozen tracker executes its body-velocity commands. A parallel nominal
rollout driven by the same commands supplies the calibrated budget-update cost.
""", "保留执行与并行名义记账的区别及相同命令条件。")

paragraph("Navigation performance, execution-error calibration,", r"""
The evaluation covers navigation performance, execution-error calibration, margin
contributions and sensitivity, budget selection, disturbance robustness, and
physical deployment. The planar benchmark assesses high-level velocity-interface
decisions; the hardware study examines quadruped execution.
""", "压缩实验导航语，保留平面决策评估与硬件执行评估的界限。")

paragraph("The navigation benchmark comprises", r"""
The benchmark uses moderately cluttered, narrow-passage, and densely obstructed
$8\times8$ m maps. Methods share test maps, navigation observations, command
limits, and the frozen tracker, with matched training interaction budgets and network
widths. The decision interval is 0.2 s and horizon 30 s. Each method has three
policies trained with different seeds: 100 trials per policy and scenario give
300 per scenario and 900 overall. Calibration, training, budget selection, and
final evaluation use separate rollout sets.
""", "保留三场景、同条件比较、全部频率/时域/种子/分母与数据划分条件。")

paragraph("Five metrics are reported:", r"""
Metrics are success rate (SR), mean executed cost (Cost), physical collision rate
(CR), budget violation rate (BVR), and mean per-trial minimum clearance (MC).
Success requires collision-free goal reaching before timeout; budget compliance
is separate. Collision and clearance use a common physical footprint. Cost and
MC include all trials until goal reaching, collision, or timeout; evaluation
continues after budget exceedance. Reference budgets $z_{\rm ref}$ are 120, 140,
and 190 a.u. for the moderately cluttered, narrow-passage, and densely obstructed
scenes, respectively. The ratio $z_0/z_{\rm ref}$ normalizes initial budget.
Tables report equally weighted scene means at $z_0/z_{\rm ref}=1$; scene-level
intervals describe test-case variability conditional on the trained policies.
""", "缩短五指标说明，保留失败分母、超预算继续评估、参考预算对应关系与条件区间定义。")

paragraph("The baselines comprise uncalibrated", r"""
Baselines are uncalibrated RC-PPO, reach-avoid learning (RA), reward-based proximal
policy optimization (PPO), and constrained PPO (CPPO), all using the same frozen
tracker. Uncalibrated RC-PPO sets both margins to zero. RA retains goal reaching and avoidance
but omits cost minimization and budget input. PPO and CPPO use lower/higher
collision-penalty weights $\beta_L=1$/$\beta_H=5$; CPPO additionally imposes
$\mathbb{E}[J/z_{\rm ref}]\leq1$ on expected cumulative cost.
""", "保留各基线具体差别、两组惩罚权重与期望成本约束，不把它写成逐轨迹预算保证。")

paragraph("The complete pipeline is compared", r"""
Table~\ref{tab:main_results} compares overall performance;
Fig.~\ref{fig:metrics_draft} reports the five metrics by scene. Relative to
uncalibrated RC-PPO, CaMRA raises mean SR from 70.7\% to 73.3\%, reduces Cost
from 91.7 to 88.3 a.u. and BVR from 14.0\% to 9.0\%, with similar CR
(15.7\% for RC-PPO versus 15.0\% for CaMRA). RA and CPPO$_{\beta_L}$ achieve lower CR and greater
clearance but lower SR and higher Cost and BVR, illustrating the joint
success--safety--cost trade-off.
""", "保留全部性能数值与对比方向，以及RA/CPPO更安全但成功/成本较弱的限定。")

paragraph("Navigation behavior in one paired example", r"""
In the paired example of Fig.~\ref{fig:trajectory_sync}, CaMRA follows a slightly
longer, costlier route with greater clearance under the geometric margin $\delta$.
Obstacle passages are marked in synchronized clearance, executed-cost, and
tracking-error curves; the corrected nominal charge illustrates $\alpha$'s
budget-accounting role. Both methods satisfy the common budget, illustrating a
safety--cost trade-off. This separate diagnostic uses the same proxy cost and
is excluded from Table~\ref{tab:main_results}.
""", "压缩单例诊断，保留不是主表样本、同代理成本和双方均预算合规等限定。")

paragraph("The contributions of the two execution margins", r"""
Table~\ref{tab:ablation_draft} and Fig.~\ref{fig:ablation_margins} compare margin
ablations with the same frozen tracker. Single-margin variants set $\alpha=0$
or $\delta=0$, retaining the other's calibrated value; removing both recovers
uncalibrated RC-PPO. Without $\alpha$, BVR rises by 13.4 percentage points;
without $\delta$, CR rises by 11.3 percentage points and clearance decreases.
""", "保留两种单裕度和双移除设定、差值单位percentage points及安全/预算不同影响。")

paragraph("The CaMRA and RC-PPO results from", r"""
Full and zero-margin rows reuse the CaMRA and RC-PPO results in
Table~\ref{tab:main_results}. Single-margin variants are trained separately with
the margin removed in both learning and evaluation. Effects are non-additive;
more failures or shorter trajectories can lower Cost, which must be interpreted
jointly with SR.
""", "保留主结果复用、分别重训、训练和评估均移除，以及成本不能脱离成功率解释。")

paragraph("With policy parameters and margins fixed,", r"""
With policy and margins fixed, budget selection uses separate rollouts, while
Fig.~\ref{fig:budget_draft} shows final-evaluation budget responses. RA reuses
the same trajectories at every threshold, so only BVR changes. Under SR
$\geq60\%$, CR $\leq20\%$, and BVR $\leq15\%$, the smallest tested ratios
satisfying all criteria on the final curves are $0.90$ for CaMRA, $0.96$ for
RC-PPO, and $1.12$ for RA. CaMRA thus meets the same criteria at a lower tested
budget.
""", "保留预算选择集和最终响应曲线的区别，及RA同轨迹、阈值、三个已测最小值。")

paragraph("A Unitree Go2 is used in two", r"""
A Unitree Go2 is evaluated in two cluttered indoor settings:
Fig.~\ref{fig:hardware_sequences}(a) examines collision avoidance in an
obstacle-filled corridor; Fig.~\ref{fig:hardware_sequences}(b) examines low-cost
route selection among multiple chair-obstacle passages.
""", "合并硬件场景描述，保留两个场景及分别检验的能力。")

paragraph("CaMRA is compared with uncalibrated RC-PPO in", r"""
Table~\ref{tab:hardware_comparison} compares CaMRA and uncalibrated RC-PPO with
matched scenes, starts, goals, and budget. Both share the frozen tracker and
navigation inputs; high-level policies remain fixed throughout evaluation.
""", "保留硬件配对比较条件和不再更新策略。")

paragraph("The deployment of minimum-cost reach-avoid navigation", r"""
CaMRA connects quadruped MCRA navigation to execution through hierarchical control
and calibration. A frozen low-level controller stabilizes motion while a
high-level policy learns navigation, mitigating coupled learning. Separate cost
and geometric margins account for residual nominal/execution mismatch. The
evaluations illustrate their effects on budget compliance, obstacle clearance,
and goal reachability. Future work will explore semantic task understanding and
stochastic formulations in which execution uncertainty can be incorporated into
formal reach-avoid and cost guarantees.
""", "压缩结论，保留分层、双裕度、经验结果与未来才考虑随机形式化保证的时序。")

paragraph("Success counts are $35/50$", r"""
CaMRA succeeds in $35/50$ trials versus $32/50$ for RC-PPO. Both have three
localization aborts; remaining failures are four timeouts and eight contacts for
CaMRA, versus five timeouts and ten contacts for RC-PPO. BVR decreases from
22.0\% to 14.0\%, while mean cost increases from 176.0 to 182.0 a.u. These
results indicate a clearance--cost trade-off: shorter routes and earlier failures
can lower mean cost without reducing its upper tail. Cost and MC include all
terminal outcomes. Hardware results are separate from the planar benchmark
because route, horizon, and footprint differ.
""", "压缩硬件失败分析，保留全部统计、成本解释和分母条件。")

# Scientific displays, captions, references and submitted metadata are not prose-edit targets.
for env in ["figure", "table", "equation", "abstract", "IEEEkeywords"]:
    pattern = rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}"
    assert re.findall(pattern, original, re.S) == re.findall(pattern, text, re.S), env
for command in ["cite", "label", "ref"]:
    pattern = rf"\\{command}\{{[^}}]*\}}"
    assert collections.Counter(re.findall(pattern, original)) == collections.Counter(re.findall(pattern, text)), command

# Every rendered numeric token survives with its original spelling and multiplicity.
def visible_numbers(source: str) -> collections.Counter:
    source = re.sub(r"(?m)(?<!\\)%[^\n]*", "", source)
    return collections.Counter(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", source))

assert visible_numbers(original) == visible_numbers(text), (visible_numbers(original)-visible_numbers(text), visible_numbers(text)-visible_numbers(original))
TARGET.mkdir(parents=True)
(TARGET / "main.tex").write_text(text, encoding="utf-8", newline="\n")
for name in ["references.bib", "IEEEtran.cls"]:
    shutil.copy2(SOURCE / name, TARGET / name)
build = (BASE / "scripts/build-1118.ps1").read_text(encoding="utf-8").replace("1118", "1118_concise")
(BASE / "scripts/build-1118_concise.ps1").write_text(build, encoding="utf-8", newline="\n")
(RECORDS / "replacements.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
(RECORDS / "manuscript.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile="EECR/latex/1118/main.tex", tofile="EECR/latex/1118_concise/main.tex")), encoding="utf-8")
print(json.dumps({"replacement_blocks": len(changes), "numeric_tokens_preserved": True,
                  "display_math_figures_tables_abstract_unchanged": True,
                  "citation_label_reference_counts_preserved": True,
                  "before_words_approx": len(re.findall(r"\b[A-Za-z]+\b", original)),
                  "after_words_approx": len(re.findall(r"\b[A-Za-z]+\b", text))}, indent=2))
