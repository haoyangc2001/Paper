# HD-MCRA 0914 logic audit

This audit reviews the assumption-data draft as if it were read by a first-time
CoRL reviewer. The numerical placeholders are intentionally not treated as evidence.

## Causal chain

1. MCRA/RC-PPO supplies trajectory-level goal, safety and budget semantics.
2. A real quadruped realizes high-level velocity commands through a learned closed
   loop, creating cost overrun and path deviation.
3. Freezing the velocity tracker makes the high-level transition repeatable enough
   to collect paired nominal/executed samples.
4. Quantiles of paired cost and path errors map to separate corrections `alpha` and
   `delta`.
5. The corrected model is used by the inherited RC-PPO learning and budget search.
6. Controlled experiments and Go2 deployment test each link.

## Findings

### Critical

- None in the assumption-data draft after the first pass.

### Major

- Exact application cost, simulator configuration, calibration source, baseline
  implementation and confidence-interval inputs are still assumed or TBD. This is
  acceptable for structure review only; these values must be replaced before a
  scientific submission.
- The draft uses a 90th-percentile marginal calibration for `alpha` and `delta`.
  This does not account for temporal dependence or joint tail events. The text
  correctly limits the claim to empirical calibration-distribution robustness, but
  the final data section should report the sample count and uncertainty of each
  quantile.
- The Go2 failure categories are provisional. They must not be retained as factual
  analysis without the trial log.

### Minor

- The title emphasizes engineering deployment; the abstract and conclusion should
  retain that wording after real numbers are inserted.
- Bibliography entries for risk-aware planning are present and should be retained
  only when the related-work comparison remains accurate.
- Expected font substitution warnings come from the supplied CoRL style and are not
  content errors.

## OpenReview coverage matrix

| Concern | 0914 response |
| --- | --- |
| Undefined `C` | Defined in Problem Setup and attributed to RC-PPO |
| RC-PPO/CPPO unclear | RC-PPO inherited steps are summarized; CPPO variants are operationally defined in the protocol |
| Source of `alpha`/`delta` unclear | Paired rollout dataset and quantile procedure are described; source values remain assumed/TBD |
| `beta_L`/`beta_H` unclear | Defined as provisional low/high safety-weight baseline variants |
| “w/o Hierarchy” unclear | Defined as joint-space 12-actuator-target variant |
| Missing simulator/trials/parameters | Draft protocol provides assumed placeholders and an explicit TBD list |
| Why low-level randomness matters | Frozen interface and mismatch rationale are stated |
| Need risk-aware/stochastic comparison | STEP and Schmittle are discussed in Related Work; no unsupported numerical baseline is added |
| Margins seem arbitrary | Quantile calibration and sensitivity sweep are included, with empirical-claim limitation |
| No confidence intervals | Explicitly listed as required replacement data, not fabricated |
| Real-world evidence limited | Go2 study is framed as an engineering deployment study and limitations state its scale |
| Strict stochastic guarantee overstated | Main text repeatedly limits robustness to the calibration distribution |

## Resolution rule

Before submission, replace every `assumed data` marker using traceable logs, add
confidence intervals and exact protocol values, then rerun this audit. No placeholder
failure category or metric may survive into the final results section.

## Second-pass consistency review

The title, abstract, Introduction, method, experiments and conclusion now describe
the same contribution: an execution-calibrated deployment layer around RC-PPO/MCRA.
The low-level tracker is consistently presented as a frozen interface dependency;
the method sequence is interface -> paired mismatch data -> `alpha`/`delta` ->
calibrated RC-PPO -> deployment. RQ1--RQ5 map to that sequence without introducing
an unrelated algorithmic claim. The assumed-data warning appears in the abstract,
tables, figure captions, limitations and README. No critical or major logical issue
remains for the structure-review draft.

Remaining pre-submission work is factual replacement, not a change to the argument:
restore the exact cost and training configuration, replace all assumed values,
provide trial-level confidence intervals, verify baseline semantics, and replace the
provisional Go2 failure categories with logged outcomes.
