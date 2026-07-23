# Skill: TabPFN Regression (Foundation Model for Tabular Data)

# Generation

## Overview
TabPFN is a pretrained transformer foundation model for **tabular data**. It delivers strong
predictions on small-to-medium tables with **no hyperparameter tuning and no iterative
training loop** — you `fit` once (the model conditions on your data in a single forward pass)
and then `predict`. 

## When to Use
- The target's value is genuinely UNCERTAIN given the inputs, not already recoverable by
  arithmetic. This is the core test — if it doesn't hold, no amount of tuning makes
  regression the right tool.
- Continuous-value prediction (e.g. "predict a city's CO2 concentration") on modest data
  (roughly ≤ 10,000 training rows, ≤ 500 effective features).
- You want a strong baseline fast, without model selection or tuning.

## When NOT to Use
- The target is an ALGEBRAIC function of the given inputs — e.g. a total that is literally
  the sum of the other columns (`total_enrollment` = sum of grade-level counts), or a rate
  that is literally a ratio of two others. That's `derive`, not regression: TabPFN would
  just be re-deriving arithmetic with sampling noise on top. (The sandbox checks this at
  fit time and rejects it — see the target-determinism check.)
- Very large datasets (well beyond ~10k rows) — sample down first or use a scalable library.
- Text/image/free-form inputs — TabPFN expects numeric tabular features.
- Hard interpretability requirements — TabPFN is a black-box model.
- Streaming / online incremental learning.

## Import Requirements
```python
# Hosted API client (requires TABPFN_API_KEY in the environment).
from tabpfn_client import TabPFNRegressor
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

# Feature/target split. Features must be numeric; keep the effective (post-encoding)
# feature count within the model's limits.
feature_cols = [c for c in df_encoded.columns if c != "target"]
X = df_encoded[feature_cols].to_numpy()
```
**Never encode or include identifier-like columns as features.** IDs, codes (school/agency
codes, DBNs, license or permit numbers), names, phone numbers, addresses, and zip/postal codes
are labels, not measurements — even when "encode categorical columns" or "everything except the
target" above would otherwise sweep them in, EXCLUDE identifier-like columns explicitly from
both the `columns=[...]` passed to `get_dummies` and the final `feature_cols`. One-hot-encoding
a column like `school_dbn` or `school_name` turns each record's own identity into a feature,
letting the model fit each row's individual history instead of a real relationship — a
hypothetical-scenario prediction then silently collapses to whichever category `drop_first`
happened to drop, since the scenario's actual entity has no dummy column of its own.

Keep an eye on two limits: training **rows** and **effective feature count** (after
`get_dummies`).

## Regression
Predict a continuous numeric target.
```python
y = df["co2_concentration"].to_numpy()       # continuous target, no encoding needed

reg = TabPFNRegressor()
reg.fit(X, y)

result = df.copy()
result["predicted_value"] = reg.predict(X)
```

## Framing the Prediction: Hypothetical Scenario vs. Held-Out Evaluation
The plan's question tells you which of these two framings to use — check its
`description` for which one was chosen. Both `fit` on the real data; they differ in
what they `predict` on.

**Hypothetical / what-if scenario** — the question poses a constructed situation
("if a school had 40 teachers, how many parents would be expected to attend?"), not a
question about specific existing rows. Fit on the real data, then predict on a
purpose-built input row representing the scenario — never re-score rows already in
the table and call that "hypothetical."
```python
reg = TabPFNRegressor()
reg.fit(X, y)

# Construct the scenario as its own row, using the same feature columns as X.
scenario = pd.DataFrame([{"num_teachers": 40}])
scenario_encoded = scenario.reindex(columns=feature_cols, fill_value=0)  # align to X's columns
predicted_value = reg.predict(scenario_encoded.to_numpy())[0]
```

**Held-out evaluation** — the question asks how well an outcome can be predicted from
other columns ("how accurately can we predict CO2 concentration from weather
readings?"), i.e. it's about the model's predictive power, not a specific scenario.
Split by index (not by array position) so you can trace predictions back to the
original rows that were genuinely unseen during `fit`. Use `numpy` for the shuffle —
`sklearn` is not in the sandbox's allowed imports.
```python
rng = np.random.RandomState(0)
shuffled_idx = df_encoded.index.to_numpy().copy()
rng.shuffle(shuffled_idx)
split = int(len(shuffled_idx) * 0.8)
train_idx, test_idx = shuffled_idx[:split], shuffled_idx[split:]

reg = TabPFNRegressor()
reg.fit(
    df_encoded.loc[train_idx, feature_cols].to_numpy(),
    df_encoded.loc[train_idx, "co2_concentration"].to_numpy(),
)
predicted_values = reg.predict(df_encoded.loc[test_idx, feature_cols].to_numpy())
```

## Common Pitfalls
❌ Passing categorical/string columns directly — encode them with `pd.get_dummies` first.
❌ Exceeding the row / feature limits — sample rows or reduce features first.
❌ Reaching for `sklearn` (e.g. `LinearRegression`) instead of `TabPFNRegressor` — this skill
  is only satisfied by the TabPFN estimators; plain `sklearn` models are not part of the
  import allowlist for this task type.
❌ Calling a prediction "hypothetical" while actually just re-running `predict` on the
  same rows used for `fit` — that's the plain pattern above, not a what-if scenario.
❌ Held-out evaluation only tells you about the split you happened to draw — don't
  present it as a hypothetical scenario, and don't present the plain fit-predict-on-X
  pattern as an out-of-sample evaluation.

## Output Format
The shape of `result` depends on which framing was used:
- **Plain / hypothetical**: return the original rows plus the predicted value column
  (plain pattern), or a small result describing the scenario and its predicted value
  (hypothetical pattern).
```python
result = df.copy()
result["predicted_value"] = reg.predict(X)
```
- **Held-out evaluation**: return the held-out rows plus their predicted values, so the
  result shows predictions against genuinely unseen data.
```python
result = df.loc[test_idx].copy()
result["predicted_value"] = predicted_values
```

## Notes
- Pretrained foundation model: no tuning, no training loop — `fit` conditions on your data.
- Choose `TabPFNRegressor` for continuous targets.

# Plan Judge Check

This plan proposes a `regression` step. This section is additive to Check 4 above — it does not re-litigate whether a skill belongs here at all, only whether THIS specific skill's use is sound.

- **Genuinely continuous target.** The target (`columns_role.target`) must be a real measured quantity — never an identifier (zip/postal code, ID, record code, phone number, coordinate) and never a value already fixed/known for the specific rows the question describes.
- **No filtered-then-reused feature.** Same zero-variance failure as classification: if an earlier step already filters a column down to one value, reusing that column as a regression feature gives the model nothing left to learn from in it — the output is really just the target's mean among the already-filtered rows. Reject.
- **Hypothetical framing is concrete.** A what-if question ("What grant amount would a project in district 2 likely receive?") must commit to concrete predictor values in the question text itself; a question that only names predictor columns with no committed values should instead use a plain-language reliability framing, not model-evaluation vocabulary.

# Code Judge Check

The plan approved a `regression` step — that choice itself is NOT yours to re-judge. Your job is whether the CODE implements it correctly. Trigger `skill_misuse` for any of the following; use `meaningless_prediction_target` instead when the issue is specifically the choice of target column (already covered by Check 3), not the technique.

- **Categorical features must be encoded.** Raw string/categorical columns passed directly to `TabPFNRegressor.fit` (no `pd.get_dummies` or equivalent) is a defect.
- **Hypothetical framing predicts on a constructed scenario row.** When the question poses a what-if scenario, the code must build a new row for that scenario (aligned to the same feature columns) and predict on THAT — re-scoring existing rows and presenting it as the scenario's answer is wrong.
- **Held-out framing predicts on genuinely unseen rows.** When the question is about general reliability, the reported prediction(s) must come from a split the model did not train on, not the training rows.
- **No identifier one-hot-encoded as a feature.** If `get_dummies` (or any encoding) was applied to an ID/code/name/DBN/zip/phone/address column and the resulting dummy columns ended up in the feature set, that's `skill_misuse`: each dummy is really that record's own identity, not a genuine predictor, and it lets the model fit per-entity history instead of a real relationship — check the `columns=[...]` passed to the encoder and the final feature list, not just the raw column names visible in the question.
- **Row/feature limits respected.** Well beyond ~10,000 training rows or ~500 effective post-encoding features without any sampling/reduction is a defect, not just a performance nit — TabPFN's documented operating range.
- **Output matches the framing.** Hypothetical: the scenario's predicted value. Held-out: the held-out rows' predicted values. A result that doesn't match its own framing is incomplete.
