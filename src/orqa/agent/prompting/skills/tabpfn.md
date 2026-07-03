---
name: tabpfn
version: 1
provider: api
applies_to: [PANDAS]
task_types:
  - classification
  - regression
  - timeseries
  - imputation
  - causal
data_constraints:
  min_rows: 1
  max_rows: 10000
  max_features: 500
  max_classes: 10
  context_budget: 100000
requires: [tabpfn_client]
keywords: [predict, forecast, classify, impute, causal, tabular]
---

# Skill: TabPFN (Foundation Model for Tabular Data)

## Overview
TabPFN is a pretrained transformer foundation model for **tabular data**. It delivers strong
predictions on small-to-medium tables with **no hyperparameter tuning and no iterative
training loop** — you `fit` once (the model conditions on your data in a single forward pass)
and then `predict`. Use it via the **hosted API** through the `tabpfn_client` package, which
requires a `TABPFN_API_KEY`.

TabPFN is versatile. Beyond plain classification it also supports **regression**,
**time-series prediction**, **missing-value imputation**, and **causal / counterfactual**
estimation. Pick the estimator and framing that matches the task.

## When to Use
- Tabular prediction on modest data (roughly ≤ 10,000 training rows, ≤ 500 effective features).
- You want a strong baseline fast, without model selection or tuning.
- Classification (e.g. "classify transactions as fraud"), regression (e.g. "predict a city's
  CO2 concentration"), forecasting a numeric series from its history, filling in missing cells,
  or estimating the effect of an intervention.

## When NOT to Use
- Very large datasets (well beyond ~10k rows) — sample down first or use a scalable library.
- Text/image/free-form inputs — TabPFN expects numeric tabular features.
- Hard interpretability requirements — TabPFN is a black-box model.
- Streaming / online incremental learning.

## Import Requirements
```python
# Hosted API client (requires TABPFN_API_KEY in the environment).
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
import numpy as np
import pandas as pd
```

## Data Preparation (applies to every task type)
TabPFN expects **numeric** feature matrices.
```python
# Encode categorical features to numeric via one-hot (drop_first avoids collinearity).
df_encoded = pd.get_dummies(df, columns=["city", "category"], drop_first=True)

# Feature/target split. Features must be numeric; keep the effective (post-encoding)
# feature count within the model's limits.
feature_cols = [c for c in df_encoded.columns if c != "target"]
X = df_encoded[feature_cols].to_numpy()
```
Keep an eye on three limits: training **rows**, **effective feature count** (after
`get_dummies`), and — for classification — the number of **target classes**.

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

## Regression
Predict a continuous numeric target.
```python
y = df["co2_concentration"].to_numpy()       # continuous target, no encoding needed

reg = TabPFNRegressor()
reg.fit(X, y)

result = df.copy()
result["predicted_value"] = reg.predict(X)
```

## Time-Series Prediction
Frame forecasting as **supervised regression on lag/time features** derived from the series,
then predict with `TabPFNRegressor`. Example: predict a city's CO2 concentration today from
its historical timeseries.
```python
series = df.sort_values("date").reset_index(drop=True)

# Build lag features and simple calendar features from the ordered series.
for lag in (1, 2, 3, 7):
    series[f"lag_{lag}"] = series["co2"].shift(lag)
series["dayofyear"] = pd.to_datetime(series["date"]).dt.dayofyear

feat = series.dropna().reset_index(drop=True)
lag_cols = [f"lag_{lag}" for lag in (1, 2, 3, 7)] + ["dayofyear"]

# Train on all-but-last, predict the most recent point (or forecast forward).
train, target = feat.iloc[:-1], feat.iloc[[-1]]
reg = TabPFNRegressor()
reg.fit(train[lag_cols].to_numpy(), train["co2"].to_numpy())
forecast = reg.predict(target[lag_cols].to_numpy())
```

## Missing-Value Imputation
Impute a column with missing entries by treating the **complete rows as training data** and
the incomplete rows as the prediction set. Use a regressor for numeric columns and a
classifier for categorical ones.
```python
target_col = "income"
predictor_cols = [c for c in df_encoded.columns if c != target_col]

known = df_encoded[df_encoded[target_col].notna()]
unknown = df_encoded[df_encoded[target_col].isna()]

reg = TabPFNRegressor()
reg.fit(known[predictor_cols].to_numpy(), known[target_col].to_numpy())

filled = df.copy()
filled.loc[df[target_col].isna(), target_col] = reg.predict(unknown[predictor_cols].to_numpy())
```

## Causal / Counterfactual Estimation
Estimate an intervention's effect by predicting the outcome under both treatment values and
comparing. Fit on observed data including the treatment column, then predict with the
treatment flipped.
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

## Common Pitfalls
❌ Passing categorical/string columns directly — encode them with `pd.get_dummies` first.
❌ Non-integer classification targets — map labels to integer codes with `.cat.codes`.
❌ Exceeding the row / feature / class limits — sample rows or reduce features first.
❌ Forgetting `TABPFN_API_KEY` — the hosted client cannot authenticate without it.

## Output Format
Return the original rows plus the prediction, and (for classification) a confidence column.
Sort by confidence so the most certain predictions come first.
```python
result = df.copy()
result["predicted_class"] = predictions
result["confidence"] = probabilities.max(axis=1)
result = result.sort_values("confidence", ascending=False)
```

## Notes
- Pretrained foundation model: no tuning, no training loop — `fit` conditions on your data.
- Choose `TabPFNClassifier` for discrete targets and `TabPFNRegressor` for continuous targets.
- Regression powers the timeseries, imputation, and causal framings shown above.
