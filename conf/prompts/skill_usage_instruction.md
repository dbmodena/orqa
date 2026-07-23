The plan above contains a step whose op matches this skill's task type
(classification/regression/timeseries/causal). You MUST implement
that step with this skill's documented pattern — including its documented
imports (e.g. `from tabpfn_client import TabPFNRegressor`) — NOT with plain
pandas, sklearn, statsmodels, or any other substitute. Plain pandas is
acceptable only for the surrounding non-ML steps (filtering, joining,
feature preparation). A query whose plan calls for this skill but whose code
never uses the skill's documented model is WRONG even if it runs: e.g. a
`timeseries` plan answered by merely subtracting one period's column from
another has skipped the required forecasting step.

The skill's code examples use a generic `df` placeholder — this is NOT a
pre-bound variable. You MUST either replace every `df` in the pattern with
the real DataFrame alias given above (e.g. `Table_0`), or add `df = Table_0`
(using the real alias) as your first line before following the pattern.
Every DataFrame alias your code operates on must appear literally, verbatim,
somewhere in your code — this is checked mechanically.
