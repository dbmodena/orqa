# Skill: TabPFN Causal / Counterfactual Estimation (Foundation Model for Tabular Data)

# Generation

## Overview
TabPFN is a pretrained transformer foundation model for **tabular data**. It delivers strong
predictions on small-to-medium tables with **no hyperparameter tuning and no iterative
training loop** — you `fit` once (the model conditions on your data in a single forward pass)
and then `predict`. TabPFN has no dedicated causal-inference API — every technique below is a
standard causal-inference method (S-learner, T-learner, quantile-based uncertainty, subgroup
breakdown) layered on top of the plain `TabPFNRegressor`/`TabPFNClassifier`.

## When to Use
- Estimating the effect of an intervention/treatment (e.g. "estimate the effect of a treatment
  on recovery") on modest data (roughly ≤ 10,000 training rows, ≤ 500 effective features).
- Specifically when a plain correlation/`.corr()` between two columns would be misleading
  or incomplete — e.g. a confounding variable plausibly drives both the "treatment" and the
  outcome (correlation would overstate the effect), the relationship could run in either
  direction (correlation can't tell you which), or the question is genuinely about "what
  would happen if we changed X" rather than "do X and Y move together." If the question is
  satisfied by a plain association check, use `derive`/`.corr()` instead — reach for this
  skill only when the business question needs the effect isolated from confounding
  covariates, not just the raw association.
- The outcome can be continuous (regression S-/T-learner) OR binary/categorical (classification
  S-learner) — pick the estimator that matches the outcome's dtype.
- You want a strong baseline fast, without model selection or tuning.

## When NOT to Use
- The question is really just "what's associated with X?" with no named alternative
  explanation to disentangle — that's `derive`/`.corr()`, not causal estimation. A causal
  step needs a stated confound or intervention whose effect must be isolated, not a generic
  association.
- Very large datasets (well beyond ~10k rows) — sample down first or use a scalable library.
- Hard interpretability requirements — TabPFN is a black-box model.
- Settings with strong unmeasured confounding — this approach only adjusts for observed
  covariates.

## Import Requirements
```python
# Hosted API client (requires TABPFN_API_KEY in the environment).
from tabpfn_client import TabPFNRegressor, TabPFNClassifier
import numpy as np
import pandas as pd
```

## Data Preparation
TabPFN expects **numeric** feature matrices.
```python
# Encode categorical features to numeric via one-hot (drop_first avoids collinearity).
# "<CATEGORICAL_COL_1>", "<CATEGORICAL_COL_2>" are PLACEHOLDERS — replace them with the
# REAL categorical column names from the table's schema/sample given to you. Never copy
# these placeholder names into your code verbatim; a column that doesn't exist in the
# table will fail at execution with a KeyError.
df_encoded = pd.get_dummies(df, columns=["<CATEGORICAL_COL_1>", "<CATEGORICAL_COL_2>"], drop_first=True)
```
**Never encode or include identifier-like columns as covariates.** IDs, codes (school/agency
codes, DBNs, license or permit numbers), names, phone numbers, addresses, and zip/postal codes
are labels, not measurements — exclude them explicitly from the `columns=[...]` passed to
`get_dummies` and from `covariates`. One-hot-encoding an identifier turns each record's own
identity into a covariate, letting the model fit each row's individual history instead of a
real treatment effect.

Keep an eye on two limits: training **rows** and **effective feature count** (after
`get_dummies`).

## Choosing an Approach
- Continuous outcome, straightforward case → **S-learner (regressor)** — the default, below.
- Binary/categorical outcome (e.g. "did X happen: yes/no") → **S-learner (classifier)**.
- Suspect the treatment's effect is being under-weighted among many covariates, or want a
  robustness check on an S-learner result → **T-learner**.
- The question needs a confidence range, not just a single number → add **quantile
  uncertainty** to the regressor calls.
- The question asks *where* the effect is strongest → **subgroup breakdown** (Output Format).
Start from the simplest approach that answers the question; only add T-learner / quantiles /
subgroup breakdown when the question specifically calls for it.

## Causal / Counterfactual Estimation

### S-learner — continuous outcome (default)
Fit one model on covariates + treatment, then predict with the treatment flipped.
```python
outcome, treatment = "recovered", "treated"
covariates = [c for c in df_encoded.columns if c not in (outcome, treatment)]

reg = TabPFNRegressor()
reg.fit(df_encoded[covariates + [treatment]].to_numpy(), df_encoded[outcome].to_numpy())

# Counterfactual outcomes with everyone treated vs. no one treated.
X1 = df_encoded[covariates + [treatment]].copy(); X1[treatment] = 1
X0 = df_encoded[covariates + [treatment]].copy(); X0[treatment] = 0
effect = reg.predict(X1.to_numpy()) - reg.predict(X0.to_numpy())

average_treatment_effect = float(np.mean(effect))
```

### S-learner — binary/categorical outcome
Same idea with `TabPFNClassifier.predict_proba`: the effect is the change in the predicted
probability of the outcome (a risk difference), not a change in a continuous value.
```python
outcome, treatment = "recovered", "treated"  # outcome is 0/1 (or boolean) here
covariates = [c for c in df_encoded.columns if c not in (outcome, treatment)]

clf = TabPFNClassifier()
clf.fit(df_encoded[covariates + [treatment]].to_numpy(), df_encoded[outcome].to_numpy())

X1 = df_encoded[covariates + [treatment]].copy(); X1[treatment] = 1
X0 = df_encoded[covariates + [treatment]].copy(); X0[treatment] = 0
# predict_proba's columns follow clf.classes_ order, not the raw label values —
# always locate the outcome-of-interest class explicitly rather than assuming column 1.
pos_idx = list(clf.classes_).index(1)
p1 = clf.predict_proba(X1.to_numpy())[:, pos_idx]
p0 = clf.predict_proba(X0.to_numpy())[:, pos_idx]
effect = p1 - p0  # risk difference: change in P(outcome) caused by treatment

average_treatment_effect = float(np.mean(effect))
```

### T-learner — separate models per treatment arm (more robust)
Fits two independent models instead of one shared model with treatment as a feature — avoids
the S-learner's known failure mode of a black-box model under-weighting a single treatment
indicator among many covariates. Works the same way for `TabPFNClassifier` (swap in
`predict_proba` + the class-index lookup above) when the outcome is binary/categorical.
```python
outcome, treatment = "recovered", "treated"
covariates = [c for c in df_encoded.columns if c not in (outcome, treatment)]

treated_df = df_encoded[df_encoded[treatment] == 1]
control_df = df_encoded[df_encoded[treatment] == 0]

reg_treated = TabPFNRegressor().fit(treated_df[covariates].to_numpy(), treated_df[outcome].to_numpy())
reg_control = TabPFNRegressor().fit(control_df[covariates].to_numpy(), control_df[outcome].to_numpy())

X_all = df_encoded[covariates].to_numpy()
# Predicted outcome under treatment minus under control, for EVERY row (not just its own arm).
effect = reg_treated.predict(X_all) - reg_control.predict(X_all)

average_treatment_effect = float(np.mean(effect))
```
Needs enough rows in EACH arm (roughly ≥ 30–50) to fit two separate models reliably — check
`df_encoded[treatment].value_counts()` first and fall back to the S-learner when one arm is
too small.

### Uncertainty via quantiles (regression outcomes)
`TabPFNRegressor.predict(..., output_type="quantiles", quantiles=[...])` returns a **dict**
keyed by quantile (not a single array), letting you bound the effect instead of reporting only
its mean. The exact key string isn't guaranteed, so sort by the numeric suffix rather than
hardcoding a key:
```python
quantiles = [0.1, 0.5, 0.9]
q1 = reg.predict(X1.to_numpy(), output_type="quantiles", quantiles=quantiles)
q0 = reg.predict(X0.to_numpy(), output_type="quantiles", quantiles=quantiles)
q1_sorted = [q1[k] for k in sorted(q1, key=lambda k: float(k.split("_")[-1]))]
q0_sorted = [q0[k] for k in sorted(q0, key=lambda k: float(k.split("_")[-1]))]
effect_low  = np.array(q1_sorted[0])  - np.array(q0_sorted[-1])   # conservative lower bound
effect_high = np.array(q1_sorted[-1]) - np.array(q0_sorted[0])   # conservative upper bound
effect_median = np.array(q1_sorted[len(quantiles) // 2]) - np.array(q0_sorted[len(quantiles) // 2])
```
Do NOT use `output_type="full"` — it requires the optional `tabpfn`/`torch` packages, which
are not available in this environment; `"quantiles"` is the supported way to get uncertainty.

## Common Pitfalls
❌ Passing categorical/string covariates directly — encode them with `pd.get_dummies` first.
❌ Omitting the treatment column from the fit features — the model must see it to learn its
  effect.
❌ Exceeding the row / feature limits — sample rows or reduce features first.
❌ Assuming `predict_proba` column 1 is the positive class — index by
  `list(clf.classes_).index(<value>)` instead.
❌ Using `output_type="full"` — requires optional dependencies not installed here; use
  `"quantiles"` for uncertainty instead.
❌ Fitting a T-learner when one treatment arm has too few rows — check group sizes first and
  fall back to the S-learner.

## Output Format
Return the per-row counterfactual effect plus the average treatment effect.
```python
result = df.copy()
result["treatment_effect"] = effect
```
### Subgroup / heterogeneous effects
When the question asks where the effect is strongest, break the per-row effect down by a
covariate instead of reporting a single aggregate number:
```python
result.groupby("<SEGMENT_COL>")["treatment_effect"].mean().sort_values(ascending=False)
```

## Notes
- Pretrained foundation model: no tuning, no training loop — `fit` conditions on your data.
- No dedicated TabPFN causal estimator exists — S-learner, T-learner, quantile uncertainty and
  subgroup breakdown are all standard causal-inference techniques built on top of the plain
  `TabPFNRegressor`/`TabPFNClassifier`.
- Default to the S-learner; reach for the T-learner, quantile uncertainty, or a subgroup
  breakdown only when the question specifically calls for robustness, a confidence range, or
  heterogeneity — don't pile on every technique for a simple question.

# Plan Judge Check

This plan proposes a `causal` step. This section is additive to Check 4 above — it does not re-litigate whether a skill belongs here at all, only whether THIS specific skill's use is sound.

- **A genuine confound, not a proxy for plain association.** The question must explicitly contrast the suspected driver against a real alternative explanation named FROM THE DATA (a second column that could plausibly explain the same outcome) — "what's associated with X?" or "what predicts X?" is not a causal question, even when the plan labels the step `causal`; `derive`/`aggregate` already answers those, and a `causal` step there is unjustified.
- **Both sides must be real columns.** The treatment/driver and the alternative explanation must both be columns genuinely present in the table(s) the plan uses — an invented or vague "other factors" is not a valid confound.
- **The motivation must be stated, not assumed.** The plan/question should make clear WHY a simple association is insufficient here (a suspected confound, or an intervention whose effect needs isolating) — a `causal` step bolted onto a question that a stratified aggregation would answer just as well is unjustified.

# Code Judge Check

The plan approved a `causal` step — that choice itself is NOT yours to re-judge. Your job is whether the CODE implements it correctly. Trigger `skill_misuse` for any of the following.

- **Both counterfactual arms predicted.** A causal effect requires predicting the outcome with the treatment flipped BOTH ways (treatment=1 and treatment=0) and taking the difference. Code that fits a model and reports only one prediction, or reports a plain association/coefficient instead of this counterfactual difference, has not actually estimated an effect.
- **Categorical covariates encoded.** Raw string/categorical covariates passed directly into `TabPFNRegressor`/`TabPFNClassifier.fit` (no `pd.get_dummies` or equivalent) is a defect.
- **No identifier one-hot-encoded as a covariate.** If `get_dummies` was applied to an ID/code/name/DBN/zip/phone/address column and the resulting dummies ended up among `covariates`, that's `skill_misuse`: each dummy is really that record's own identity, not a genuine covariate, and it can absorb the treatment's real effect into per-entity noise.
- **Positive-class column located explicitly.** For a binary/categorical outcome, indexing `predict_proba`'s output by an assumed column position instead of `list(clf.classes_).index(<value>)` is a defect — it can silently report the wrong class's probability.
- **T-learner only with adequate arm sizes.** A T-learner (separate models per treatment arm) fit when one arm has very few rows (well under ~30) without falling back to an S-learner is unreliable — flag it.
- **The result is the treatment effect, not a correlation.** The reported output must be the counterfactual effect (per-row and/or averaged) — a `.corr()` or a raw coefficient presented as "the causal effect" is not what a `causal` step is supposed to produce.
