# 0914 CoRL rewrite workspace

This directory is a new manuscript workspace copied from the CoRL template used by
`latex/0820_submitted`. The 0820 directory is read-only and remains the comparison
baseline.

The 0914 manuscript is an engineering paper about deploying the RC-PPO/MCRA idea on
a real quadruped. Its central story is the deployment gap between the idealized
high-level model and the closed-loop execution of a learned locomotion controller.
The planned design covers a frozen velocity interface, data calibration of the
execution mismatch, conservative cost and safety corrections (`alpha` and `delta`),
and minimum-budget search. Low-level PPO is implementation background, not the main
contribution.

The final manuscript should remain close to the 0820 submission in content volume
(approximately the same 11-page scale after a clean build). Added space should be
used for engineering mechanisms, controlled experiments, and reproducibility rather
than unrelated reinforcement-learning background.

Build from this directory with:

```text
latexmk -xelatex -interaction=nonstopmode -file-line-error -outdir=../../output/0914 main.tex
```

All quantitative claims must be traceable to source material, configuration, code,
or experiment logs. Unknown values must be marked `TBD` and resolved before the
corresponding section is finalized. Do not infer confidence intervals, random-seed
statistics, or new results from estimates. Robustness claims are empirical claims
under the calibrated deployment distribution unless a separate proof establishes
stronger guarantees.

## Fact and evidence ledger

The following boundaries are enforced while drafting the manuscript.

### Confirmed source facts

- The source manuscript is the 0820 CoRL submission and uses the title
  “Hierarchical Minimum-Cost Reach-Avoid Learning for Safe Quadruped Navigation”.
- The RC-PPO reference defines MCRA through goal reaching, avoidance and cumulative
  cost, and uses the augmented state `(x,y,z)`, a reachability critic, a stochastic
  Phase-I policy, deterministic Phase-II deployment, and one-dimensional budget
  search. These are inherited foundations and must be cited to So et al. (`so2024`).
- The 0820 manuscript reports three simulated scene categories, the metrics SR, CR,
  Average Cost, BVR and MC, and a Unitree Go2 deployment study. These statements are
  legacy-report facts and are not evidence for any new result until their raw logs
  and protocol are checked.
- The 0820 manuscript reports a representative Go2 count of 26 successful trials
  out of 30 over an approximately 15 m route. The corresponding trial log,
  failures, confidence interval and exact environment protocol are not present in
  the current source tree.
- The required CoRL style, bibliography style and reference database are copied
  unchanged from `latex/0820_submitted`.

### Legacy values not yet admissible as new evidence

The tables and curves embedded in the 0820 submission are preserved as a record of
what was submitted. They must not be reinterpreted as independently reproduced
measurements. In particular, `scripts/generate_experiment_results_v3.py` labels its
workbook as an “Estimated v3 experiment package” and states that new values are
estimated for internal consistency. Its outputs therefore cannot supply confidence
intervals, seeds, calibration quantiles or new claims.

### TBD before final quantitative claims

- Exact instantaneous cost function, units, weights and normalization (`C` included).
- Simulator name/version, terrain and map generation, observation and action rates,
  low-level and high-level control frequencies, episode horizon, and termination
  handling.
- Low-level tracker interface ranges, training protocol and the definition of the
  nominal closed-loop transition.
- Source, size and split of the execution-mismatch calibration data; pairing of
  nominal and executed trajectories; definitions of cost-overrun and path-deviation
  errors; selected quantiles for `alpha` and `delta`.
- Exact implementations and hyperparameters for RC-PPO, RA, PPO/CPPO baselines,
  including `beta_L`, `beta_H`, and the “w/o Hierarchy” variant.
- Number of evaluation trials and random seeds, confidence-interval procedure,
  per-scenario results, failure categories, and raw Go2 logs.
- Whether the submitted figures are backed by reproducible data files and whether
  the robot footprint is already included in obstacle inflation.

Until these items are resolved from code, configuration or logs, the manuscript
uses `TBD` or describes a protocol without reporting a number. No estimate from a
plot, summary script or legacy table is promoted to a new experimental conclusion.

## Ownership boundary

RC-PPO supplies the MCRA formulation, augmented-state reduction, reachability
learning and two-phase minimum-budget procedure. The low-level PPO tracker is a
system dependency that supplies a velocity execution interface. The 0914 paper's
new engineering scope is the closed-loop deployment interface, the paired
execution-mismatch data, the cost and safety calibration (`alpha` and `delta`), and
the experiments that test those mechanisms on a quadruped.

## Literature difference matrix

| Line of work | Primary object | Uncertainty treatment | Relation to 0914 |
| --- | --- | --- | --- |
| RC-PPO / MCRA (`so2024`) | Minimum-cost reach-avoid learning | Idealized deterministic transition and augmented budget state | Supplies the inherited objective, critic construction and two-phase solver |
| Safe quadruped navigation (`miki2024`, `weerakoon2024`, `hoeller2024`) | Locomotion or navigation performance | Platform-specific robustness, not MCRA budget calibration | Motivates the hardware and interface constraints |
| Risk-aware planning (`fan2021step`, `schmittle2024`) | Probabilistic traversability or multi-sample path planning | Explicit risk/sensing uncertainty in planning | Provides a comparison boundary; these methods do not use the same MCRA budget-and-avoid-set calibration pipeline |
| HD-MCRA (this paper) | Deployable MCRA navigation on a quadruped | Empirical quantile corrections for paired execution cost and path errors | Adds the closed-loop interface and maps measured mismatch to `alpha` and `delta`; no new formal stochastic guarantee is claimed |

The matrix is a scope comparison, not a claim of absolute novelty. Any stronger
“first” or “only” statement requires a separate literature audit.

## Draft-data convention

At the author's request, the current manuscript uses analytically consistent
placeholders to review the argument before sensitive experiment logs are gathered.
Every such value is marked in the LaTeX source with `assumed data`. These values are
not experimental evidence and must be replaced as a batch with traceable trial
records before submission. In particular, the draft values for `alpha`, `delta`,
calibration sample counts, control rates, baseline metrics, sensitivity curves,
failure categories and confidence intervals are provisional only.

## Section and volume budget

The 0914 body follows the 0820 submission's roughly 11-page scale after clean
compilation. The target allocation is: Introduction 1.0 page; Related Work 0.8;
Problem Setup 1.2; Execution-Calibrated HD-MCRA 3.0; Experiments 4.0; Limitations
0.5; and Conclusion/references 0.5--1.0. These are review targets, not permission
to pad the paper: space is reserved for definitions, calibration protocol,
controlled comparisons and failure analysis. RC-PPO derivations and low-level PPO
training details remain concise.
