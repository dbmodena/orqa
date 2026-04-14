import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .LLMClientStructured import LLMClientStructured
from .validators.SQLValidator import SQLValidator
from .validators.PandasValidator import PandasValidator

# ---------------------------------------------------------------------------
# Keyword sets used for post-acceptance annotation
# ---------------------------------------------------------------------------

SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING",
    "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "OUTER JOIN",
    "FULL JOIN", "CROSS JOIN",
    "UNION", "UNION ALL", "INTERSECT", "EXCEPT",
    "SUM", "AVG", "MIN", "MAX", "COUNT", "RANK", "ROW_NUMBER",
    "DENSE_RANK", "CORR",
    "CAST", "COALESCE", "NULLIF", "CASE", "WHEN", "THEN", "ELSE",
    "DISTINCT", "LIMIT", "OFFSET", "WITH", "AS", "ON",
    "AND", "OR", "NOT", "IN", "EXISTS",
}

PANDAS_KEYWORDS = {
    "merge", "join", "concat", "groupby", "agg", "aggregate", "corr",
    "sum", "mean", "avg", "min", "max", "count", "nunique", "rank",
    "sort_values", "sort_index", "drop_duplicates", "fillna", "dropna",
    "apply", "map", "filter", "query", "where", "assign", "pivot", "melt",
    "stack", "unstack", "explode", "resample", "rolling", "expanding",
}


class LLMClientStatementGenerator(LLMClientStructured):
    """
    Specialised structured client for generating executable SQL / Pandas statements.

    Responsibilities beyond the parent:
    - Statement-specific retry messaging (validation errors from DuckDB / Pandas)
    - Keyword counting for accepted queries
    - Incremental ID assignment on acceptance
    - Sanitisation of malformed query objects before they reach the validator
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        dataframes,
        table_names,
        typology: str = "SQL",
        involved_cols=None,
        feedback: list | None = None,
    ) -> tuple[dict, dict, list[str], str]:
        """
        Generate, validate, and return SQL/Pandas queries.

        Parameters
        ----------
        prompt : str
            The base prompt describing the generation task.
        dataframes : list
            DataFrames to validate queries against.
        table_names : dict
            Mapping of table aliases to table names.
        typology : str
            Query language – ``"SQL"`` or ``"PANDAS"``.
        involved_cols : optional
            Column information used for suffix constraints.
        feedback : list | None
            Optional list of feedback messages from the judge loop.
            When provided, the conversation starts with the base prompt
            followed by the feedback messages instead of a fresh prompt.

        Returns
        -------
        ({"queries": [...]}, usage_total, all_errors, model_name)
        """
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        initial_message = self.reform_prompt_constraint(prompt)

        if feedback is not None:
            # Judge iteration: start with the base prompt, then append feedback
            messages = [
                {"role": "system", "content": "You are an expert Data Engineer"},
                {"role": "user", "content": initial_message},
            ]
            messages.extend(feedback)
        else:
            messages = [
                {"role": "system", "content": "You are an expert Data Engineer"},
                {"role": "user", "content": initial_message},
            ]

        last_content = ""
        last_error = None
        # Keyed by incremental int so duplicate LLM ids don't stomp each other
        good_queries: dict[int, dict] = {}
        all_errors: list[str] = []

        for attempt in range(self.max_retries):
            try:
                completion_args = {
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": self.temperature,
                }

                response = self.router.completion(**completion_args)
                usage = response["usage"]
                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                content = response["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError(
                        "LLM returned None content – the model may have emitted a "
                        "tool-call response instead of a text completion."
                    )
                last_content = content

                cleaned_content = self._clean_json_response(content)

                try:
                    # Full repair pipeline inherited from LLMClientStructured
                    json_data = self._repair_json(cleaned_content)
                    json_data = self._normalize_response(json_data, root_key="queries")
                    # Sanitise individual query objects before Pydantic sees them
                    json_data = self._sanitize_query_objects(json_data)

                    result = self.response_model.model_validate(json_data)
                    result = result.model_dump()

                    outcome, errors, accepted_queries, error_messages = self.validate_queries(
                        dataframes, result, table_names, typology
                    )

                    # Accept good queries and assign monotonically increasing IDs
                    for accepted_query in accepted_queries.values():
                        new_id = len(good_queries) + 1
                        accepted_query["id"] = new_id
                        accepted_query["keywords"] = self.count_keywords(
                            accepted_query["code"], typology
                        )
                        good_queries[new_id] = accepted_query

                    if not outcome:
                        all_errors.extend(error_messages)
                        messages = [
                            {"role": "system", "content": "You are an expert Data Engineer."},
                            {"role": "user", "content": prompt},
                        ]
                        messages.extend(errors)
                        messages.append({"role": "user", "content": self.reform_prompt_constraint("")})
                        continue

                    return (
                        {"queries": list(good_queries.values())},
                        usage_total,
                        [str(e) for e in all_errors],
                        self.config["model"],
                    )

                except json.JSONDecodeError as e:
                    last_error = e
                    all_errors.append(f"JSONDecodeError: {e}")
                    if attempt < self.max_retries - 1:
                        error_msg = self._format_json_error(cleaned_content, e)
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert Data Engineer. Below are the instructions "
                                    "for generating and correcting queries."
                                ),
                            },
                            {"role": "user", "content": initial_message},
                            {"role": "assistant", "content": last_content},
                            {
                                "role": "user",
                                "content": (
                                    f"You generated badly-formatted output:\n{error_msg}\n"
                                    + self.reform_prompt_constraint("")
                                ),
                            },
                        ]
                        time.sleep(self.retry_delay)
                        continue

                except ValidationError as e:
                    last_error = e
                    all_errors.append(f"ValidationError: {e}")
                    if attempt < self.max_retries - 1:
                        error_msg = self._format_validation_error(e)
                        messages = [
                            {"role": "system", "content": "You are an expert Data Engineer."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": last_content},
                            {
                                "role": "user",
                                "content": (
                                    f"Generated queries are not valid:\n{error_msg}\n"
                                    + self.reform_prompt_constraint("")
                                ),
                            },
                        ]
                        time.sleep(self.retry_delay)
                        continue

            except Exception as e:
                last_error = e
                all_errors.append(f"Exception: {e}")
                if attempt < self.max_retries - 1:
                    schema_str = json.dumps(
                        self.response_model.model_json_schema(), indent=2
                    )
                    messages = [
                        {"role": "system", "content": "You are an expert Data Engineer"},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": (
                                f"Generated queries that are not valid, error:\n{e}\n"
                                f"Return a unique JSON matching this Pydantic schema:\n{schema_str}"
                            ),
                        },
                    ]
                    time.sleep(self.retry_delay)

        # Exhausted – return whatever was accepted so far
        return (
            {"queries": list(good_queries.values())},
            usage_total,
            [str(e) for e in all_errors],
            self.config["model"],
        )


    # ------------------------------------------------------------------
    # Statement-specific helpers
    # ------------------------------------------------------------------

    def _sanitize_query_objects(self, json_data: dict) -> dict:
        """
        Filter out structurally malformed query dicts before Pydantic validation.

        A query is considered malformed and will be dropped when:
        - It is not a dict
        - ``code`` is missing, None, or an empty/whitespace-only string
        - ``question`` is missing or None
        - ``tables`` is present but is not a list

        Dropped items are silently skipped; Pydantic will still catch any
        remaining field-level issues in the survivors.
        """
        raw_queries = json_data.get("queries", [])
        if not isinstance(raw_queries, list):
            # LLM wrapped queries in an unexpected structure – reset to empty
            json_data["queries"] = []
            return json_data

        clean: list[dict] = []
        for item in raw_queries:
            if not isinstance(item, dict):
                continue
            code = item.get("code")
            question = item.get("question")
            tables = item.get("tables")

            # Must have non-empty code and a question
            if not code or not isinstance(code, str) or not code.strip():
                continue
            if not question or not isinstance(question, str):
                continue
            # Tables, if present, must be a list (may be empty)
            if tables is not None and not isinstance(tables, list):
                item["tables"] = []

            # Sanitise each table entry
            if isinstance(item.get("tables"), list):
                item["tables"] = [
                    t for t in item["tables"]
                    if isinstance(t, dict) and t.get("name") and t.get("reason")
                ]

            clean.append(item)

        json_data["queries"] = clean
        return json_data

    def validate_queries(
        self, dataframes, result: dict, table_names, typology: str
    ) -> tuple[bool, list, dict, list]:
        if typology == "SQL":
            return self._validate_sql_queries(dataframes, result, table_names)
        return self._validate_dataframe_queries(dataframes, result, table_names)

    def _validate_dataframe_queries(self, dataframes, result, table_names):
        validator = PandasValidator(dataframes, list(table_names.keys()), table_names)
        return validator.validate_queries(result)

    def _validate_sql_queries(self, dataframes, result, table_names):
        validator = SQLValidator(dataframes, list(table_names.keys()), table_names)
        return validator.validate_queries(result)

    def count_keywords(self, query: str, kind: str = "SQL") -> dict[str, int]:
        """Count occurrences of domain-specific keywords in a query string."""
        keywords = SQL_KEYWORDS if kind == "SQL" else PANDAS_KEYWORDS
        query_search = query.upper() if kind == "SQL" else query
        counts: dict[str, int] = {}
        for kw in sorted(keywords):
            kw_pattern = kw.upper() if kind == "SQL" else kw
            matches = re.findall(
                rf"\b{re.escape(kw_pattern)}\b",
                query_search,
                flags=re.IGNORECASE,
            )
            if matches:
                counts[kw] = len(matches)
        return counts

    def add_suffix_constraint(self, prompt: str, dataframes, table_names, involved_cols) -> str:
        """Append merge-suffix information to the prompt when joins are involved."""
        from itertools import combinations

        common_cols_info = []
        for (name_a, df_a), (name_b, df_b) in combinations(
            zip(table_names, dataframes), 2
        ):
            common = set(df_a.columns) & set(df_b.columns)
            common_cols_info.append(
                f"  {name_a} ∩ {name_b}: "
                f"{sorted(common) if common else '(no common columns)'}"
            )

        constraint = (
            "### Suffix Constraint Information\n"
            "Columns suffixed after merge (ONLY these get a suffix):\n"
            + "\n".join(common_cols_info)
            + "\n\n"
            + (
                f"Join key columns (NEVER suffixed – kept as-is after merge):\n"
                f"  {involved_cols}\n\n"
                if involved_cols
                else ""
            )
            + "All other columns keep their original name – do not add any suffix to them.\n"
        )
        return f"{prompt}\n{constraint}"
