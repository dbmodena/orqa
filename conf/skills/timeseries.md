# Skill: TabPFN Time-Series Prediction (Foundation Model for Tabular Data)

# Generation

## Overview
TabPFN is a pretrained transformer foundation model for **tabular data**. It delivers strong
predictions on small-to-medium tables with **no hyperparameter tuning and no iterative
training loop** — you `fit` once (the model conditions on your data in a single forward pass)
and then `predict`. 

TabPFN has no native sequence model, so time-series forecasting is framed as **supervised
regression on lag/time features** derived from the series.

## When to Use
- Determining the **tendency** of a numeric series from its own history (e.g. "is a city's
  CO2 concentration expected to rise, fall, or hold steady?") on modest data (roughly
  ≤ 10,000 training rows, ≤ 500 effective features after lag-feature construction). A raw
  point forecast is rarely the actual business question — the direction/rate of change
  usually is, so treat the forecast as an intermediate value used to derive it (see
  Time-Series Prediction below).
- You want a strong baseline fast, without model selection or tuning.

## When NOT to Use
- The question only needs a comparison between two KNOWN historical points (`rate_2012` vs
  `rate_2013`), or a lookup of one specific historical value — that's `derive`/`filter`, not
  a forecast: nothing about the future or an unobserved point is genuinely uncertain there.
- Very large datasets (well beyond ~10k rows) — sample down first or use a scalable library.
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
Build lag features and simple calendar features from the ordered series. Generated code
is flattened to one semicolon-separated line, so use an `.assign(**{...})` with a dict
comprehension for the lag columns — never a `for` statement — since a `for`/`if`/`while`
block requires an indented suite that a `;`-joined single line cannot express.
```python
series = df.sort_values("date").reset_index(drop=True)
series = series.assign(**{f"lag_{lag}": series["co2"].shift(lag) for lag in (1, 2, 3, 7)})
series["dayofyear"] = pd.to_datetime(series["date"]).dt.dayofyear

feat = series.dropna().reset_index(drop=True)
lag_cols = [f"lag_{lag}" for lag in (1, 2, 3, 7)] + ["dayofyear"]
```
Keep an eye on two limits: training **rows** and **effective feature count** (lag/calendar
columns after dropping NaNs from the shift).

## Time-Series Prediction
Train on all-but-last, forecast the most recent point (or forward), then classify the
**tendency** relative to the last known value — that comparison, not the raw forecast
number, is the headline result. Use a chained conditional EXPRESSION for the
classification, never an `if`/`elif`/`else` statement — same one-line-flattening
constraint as the lag loop above.
```python
train, target = feat.iloc[:-1], feat.iloc[[-1]]
reg = TabPFNRegressor()
reg.fit(train[lag_cols].to_numpy(), train["co2"].to_numpy())
forecast = reg.predict(target[lag_cols].to_numpy())[0]

# The forecast alone is an intermediate value. Compare it to the last known point to
# derive the tendency, which is what the business question is actually asking.
last_known = train["co2"].iloc[-1]
relative_change = (forecast - last_known) / abs(last_known) if last_known != 0 else 0.0
STABLE_THRESHOLD = 0.02  # +/-2% treated as "stable" rather than forecast noise
tendency = (
    "increasing" if relative_change > STABLE_THRESHOLD
    else "decreasing" if relative_change < -STABLE_THRESHOLD
    else "stable"
)
```

## Common Pitfalls
❌ Passing raw dates/strings instead of derived lag/calendar features.
❌ Forgetting to sort by time before building lag features — lag features on unsorted data are
  meaningless.
❌ Exceeding the row / feature limits — sample rows or reduce lag depth first.
❌ Returning only the raw forecast value as if it answered the question — a lone number
  doesn't say whether that's a rise, fall, or no real change from where the series
  already was; derive and report the tendency.
❌ Classifying tendency directly off `relative_change` without a stability band — small
  forecast noise around zero change will otherwise flip-flop between "increasing" and
  "decreasing" for what's actually a flat series.
❌ Writing a `for` loop or an `if`/`elif`/`else` statement anywhere in this code — generated
  code is flattened to one semicolon-separated line, and a `;` cannot substitute for the
  indented block these statements require. Use `.assign(**{... for ...})` instead of a
  `for` loop, and a chained conditional expression (`a if cond1 else b if cond2 else c`)
  instead of `if`/`elif`/`else` — exactly as shown above.

## Output Format
Return the tendency classification as the headline result, with the forecast and last
known value kept as supporting context (not the primary answer).
```python
result = target[["date"]].copy()
result["last_known_value"] = last_known
result["forecast"] = forecast
result["tendency"] = tendency
```

## Notes
- Pretrained foundation model: no tuning, no training loop — `fit` conditions on your data.
- `TabPFNRegressor` powers this framing — there is no dedicated TabPFN time-series estimator.

# Plan Judge Check

This plan proposes a `timeseries` step. This section is additive to Check 4 above — it does not re-litigate whether a skill belongs here at all, only whether THIS specific skill's use is sound.

- **Genuine ordered history, not a two-point comparison.** The table must carry a real date/datetime/period dimension with roughly 8+ observed points to learn a trend from. Comparing or differencing a handful of wide-format year columns (e.g. `rate_2012` vs `rate_2013`) is not a timeseries operation — that's `derive`/`aggregate`, and a `timeseries` step proposed for it is unjustified.
- **As-of anchor matches the data, not the calendar.** The question must state its as-of anchor as the LAST period actually observed in the table (visible in the column statistics/sample) — never a real-world "today" or "next year" when the data ends earlier. A plan anchored to 2024 when the table's data ends in 2020 is asking for a forecast the data cannot support — reject.
- **Projection scope matches the question's own scope.** If the question only asks about the trend within an observed historical window (e.g. "did it go up or down between 2000 and 2003?"), the plan must not extrapolate a forecast beyond that stated window — a step that forecasts further out than the question asks is scope creep, not the same task, even though both are labeled `timeseries`.
- **Direction, not an exact figure.** The question should ask whether the series is heading up, down, or flat — not demand a precise future value that a forecasting model over this data cannot honestly promise.

# Code Judge Check

The plan approved a `timeseries` step — that choice itself is NOT yours to re-judge. Your job is whether the CODE implements it correctly. Trigger `skill_misuse` for any of the following.

- **Sorted before lag features.** Lag/shift features built without first sorting by the time column are meaningless — this is a correctness defect, not a style nit.
- **Tendency derived, not just a raw forecast.** A trend/direction question ("is X expected to rise, fall, or hold steady?") answered with only a bare forecast number — no comparison back to the last known value — is incomplete: the forecast is an intermediate value, the tendency comparison is the actual answer.
- **Stability band on the tendency classification.** Classifying "increasing"/"decreasing" directly off the raw relative change with no threshold band lets pure forecast noise around zero flip the label for what is really a flat series — flag this as `skill_misuse` when present.
- **No `for`/`if`/`elif`/`else` statements.** Generated code executes as one semicolon-flattened line; a `for` loop or an `if`/`elif`/`else` block cannot execute in that form. Code using `.assign(**{...})`/chained conditional expressions is correct; code with an actual loop or multi-branch statement is a defect, and likely one that already failed at execution rather than merely being poor style.
