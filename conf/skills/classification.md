# Skill: TabPFN Classification (Foundation Model for Tabular Data)

# Generation

## Overview
TabPFN is a pretrained transformer foundation model for **tabular data**. It delivers strong
predictions on small-to-medium tables with **no hyperparameter tuning and no iterative
training loop** — you `fit` once (the model conditions on your data in a single forward pass)
and then `predict`.

## When to Use
- The mapping from inputs to the label is genuinely PROBABILISTIC: real rows with similar
  inputs still land on different real labels. This is the core test — if it doesn't hold,
  no amount of tuning makes classification the right tool.
- Discrete-label prediction (e.g. "classify transactions as fraud") on modest data (roughly
  ≤ 10,000 training rows, ≤ 500 effective features, ≤ 10 target classes).
- You want a strong baseline fast, without model selection or tuning.

## When NOT to Use
- The label is basically recoverable by a LOOKUP from the inputs — e.g. predicting a
  borough from a zip code, street name, or lat/long (near-fixed geographic mappings), or
  any case where "group by the inputs and take the most common label" would give
  essentially the same answer. That's `groupby`/mode, not classification, regardless of how
  the question is phrased.
- Very large datasets (well beyond ~10k rows) — sample down first or use a scalable library.
- Text/image/free-form inputs — TabPFN expects numeric tabular features.
- Hard interpretability requirements — TabPFN is a black-box model.
- Streaming / online incremental learning.

## Import Requirements
```python
# Hosted API client (requires TABPFN_API_KEY in the environment).
from tabpfn_client import TabPFNClassifier
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

Keep an eye on three limits: training **rows**, **effective feature count** (after
`get_dummies`), and the number of **target classes**. The hosted API **hard-rejects
targets with more than 160 classes** — check the target's cardinality and collapse
rare labels BEFORE fitting:
```python
# Keep the most frequent labels and fold the long tail into "other".
top_classes = df["target"].value_counts().nlargest(10).index
df["target"] = df["target"].where(df["target"].isin(top_classes), "other")
```

## Classification
Predict a discrete label plus a confidence.
```python
# Encode a string/categorical target to integer class codes: 0, 1, 2, ...
y = df["target"].astype("category").cat.codes.to_numpy()

clf = TabPFNClassifier()
clf.fit(X, y)

predictions = clf.predict(X)                 # integer class codes
probabilities = clf.predict_proba(X)         # shape (n_rows, n_classes)

result = df.copy()
result["predicted_class"] = predictions
result["confidence"] = probabilities.max(axis=1)
result = result.sort_values("confidence", ascending=False)
```

## Framing the Prediction: Hypothetical Scenario vs. Held-Out Evaluation
The plan's question tells you which of these two framings to use — check its
`description` for which one was chosen. Both `fit` on the real data; they differ in
what they `predict` on.

**Hypothetical / what-if scenario** — the question poses a constructed situation
("would a $5,000 transaction from a new merchant be classified as fraud?"), not a
question about specific existing rows. Fit on the real data, then predict on a
purpose-built input row representing the scenario — never re-score rows already in
the table and call that "hypothetical."
```python
clf = TabPFNClassifier()
clf.fit(X, y)

# Construct the scenario as its own row, using the same feature columns as X.
scenario = pd.DataFrame([{"amount": 5000}])
scenario_encoded = scenario.reindex(columns=feature_cols, fill_value=0)  # align to X's columns
predicted_class = clf.predict(scenario_encoded.to_numpy())[0]
confidence = clf.predict_proba(scenario_encoded.to_numpy()).max(axis=1)[0]
```

**Held-out evaluation** — the question asks how well a label can be predicted from
other columns ("how reliably can we flag fraud from transaction details?"), i.e. it's
about the model's predictive power, not a specific scenario. Split by index (not by
array position) so predictions can be traced back to rows genuinely unseen during
`fit`. Use `numpy` for the shuffle — `sklearn` is not in the sandbox's allowed imports.
```python
rng = np.random.RandomState(0)
shuffled_idx = df_encoded.index.to_numpy().copy()
rng.shuffle(shuffled_idx)
split = int(len(shuffled_idx) * 0.8)
train_idx, test_idx = shuffled_idx[:split], shuffled_idx[split:]

clf = TabPFNClassifier()
clf.fit(
    df_encoded.loc[train_idx, feature_cols].to_numpy(),
    df_encoded.loc[train_idx, "target"].astype("category").cat.codes.to_numpy(),
)
predictions = clf.predict(df_encoded.loc[test_idx, feature_cols].to_numpy())
probabilities = clf.predict_proba(df_encoded.loc[test_idx, feature_cols].to_numpy())
```

## Common Pitfalls
❌ Passing categorical/string columns directly — encode them with `pd.get_dummies` first.
❌ Non-integer classification targets — map labels to integer codes with `.cat.codes`.
❌ Exceeding the row / feature / class limits — sample rows or reduce features first.
❌ Fitting on a raw high-cardinality target (e.g. hundreds of distinct strings) — the
  API rejects >160 classes; collapse to top-N + "other" first (see Data Preparation).
❌ Calling a prediction "hypothetical" while actually just re-running `predict` on the
  same rows used for `fit` — that's the plain pattern above, not a what-if scenario.
❌ Held-out evaluation only tells you about the split you happened to draw — don't
  present it as a hypothetical scenario, and don't present the plain fit-predict-on-X
  pattern as an out-of-sample evaluation.

## Output Format
The shape of `result` depends on which framing was used:
- **Plain / hypothetical**: return the original rows plus the predicted class and
  confidence column (plain pattern), or a small result describing the scenario and its
  predicted class/confidence (hypothetical pattern), sorted by confidence when there
  are multiple rows.
```python
result = df.copy()
result["predicted_class"] = predictions
result["confidence"] = probabilities.max(axis=1)
result = result.sort_values("confidence", ascending=False)
```
- **Held-out evaluation**: return the held-out rows plus their predicted class and
  confidence, so the result shows predictions against genuinely unseen data.
```python
result = df.loc[test_idx].copy()
result["predicted_class"] = predictions
result["confidence"] = probabilities.max(axis=1)
result = result.sort_values("confidence", ascending=False)
```

## Notes
- Pretrained foundation model: no tuning, no training loop — `fit` conditions on your data.
- Choose `TabPFNClassifier` for discrete targets.

# Plan Judge Check

This plan proposes a `classification` step. This section is additive to Check 4 above — it does not re-litigate whether a skill belongs here at all (that's Check 4's job), only whether THIS specific skill's use is sound.

- **Real category, not a lookup.** The target (`columns_role.target`) must be a genuine class label with real uncertainty across the population — never a value that is already deterministically knowable from the given inputs (e.g. zip code → borough is a near-fixed mapping) or already recorded for every row the question's own filter already narrows down to. If the "prediction" is really a `groupby`/mode over rows the plan has already filtered to one slice, this is a plain aggregate wearing a classification step's clothing — reject it, and say so in `skill_check`.
- **No filtered-then-reused feature.** If an earlier step in the plan already filters a column down to one specific value (e.g. "filter where street equals 'Broadway'"), and the classification step then uses that SAME column as a feature, the model has zero variance left to learn from in it — its output collapses to the target's mode among the already-filtered rows. Reject.
- **Target is not identifier-like.** Zip/postal codes, IDs, record codes, phone numbers, URLs, addresses, names, and coordinates are labels, not genuine categories — "predicting" one is meaningless even when the column happens to be numeric.
- **Hypothetical framing is concrete.** If the question poses a what-if scenario, it must name concrete values for the predictors IN THE QUESTION TEXT itself (e.g. "grade 5 and program type STEM"), not just the predictor column names with no committed values — the latter is an evaluation question wearing hypothetical phrasing, and should instead read as a plain-language reliability question ("how well does X tell us Y?"), never in model-evaluation vocabulary ("accuracy", "held-out").

# Code Judge Check

The plan approved a `classification` step — that choice itself is NOT yours to re-judge (the plan panel already did). Your job is whether the CODE implements it correctly. Trigger `skill_misuse` for any of the following; use `meaningless_prediction_target` instead when the issue is specifically the choice of target column (already covered by Check 3), not the technique.

- **Categorical features must be encoded.** Raw string/categorical columns passed directly to `TabPFNClassifier.fit` (no `pd.get_dummies` or equivalent) is a defect, not a style choice — the model expects numeric input.
- **Target must be integer-coded.** Fitting on raw string/category labels instead of integer class codes (e.g. `.cat.codes`) is broken.
- **Target cardinality respected.** If the target has more than ~160 distinct values and the code does not collapse it (top-N + "other") before fitting, the call is expected to hard-fail — that's a code defect, not an edge case to excuse.
- **Hypothetical framing predicts on a constructed scenario row.** When the question poses a what-if scenario, the code must build a new row for that scenario (aligned to the same feature columns via e.g. `reindex`) and predict on THAT — re-scoring rows already in the table and presenting it as the scenario's answer is wrong.
- **Held-out framing predicts on genuinely unseen rows.** When the question is about general reliability, the split must produce a test set the model did not see during `fit`, and the reported prediction(s) must come from that held-out set — not the training rows.
- **No identifier one-hot-encoded as a feature.** If `get_dummies` (or any encoding) was applied to an ID/code/name/DBN/zip/phone/address column and the resulting dummy columns ended up in the feature set, that's `skill_misuse`: each dummy is really that record's own identity, not a genuine predictor, and it lets the model fit per-entity history instead of a real relationship — check the `columns=[...]` passed to the encoder and the final feature list, not just the raw column names visible in the question.
- **Output matches the framing.** Hypothetical: the scenario's predicted class + confidence. Held-out: the held-out rows' predicted class + confidence. A result that doesn't match its own framing is incomplete.
