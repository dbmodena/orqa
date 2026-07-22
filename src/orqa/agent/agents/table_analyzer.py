"""Batched table analysis.

The legacy pipeline analysed one table per LLM call (see
``LLMClientStatementGenerator._run_table_analysis`` in ``StatementClient.py``),
so analysis cost scaled with the number of tables. :class:`TableAnalyzer`
replaces that per-table loop with a single batched call: one prompt carries the
columns, sample rows, and metadata for *every* table and the model returns a
``TableAnalyses`` payload (one entry per alias).

The analyzer reuses the existing structured-output infrastructure: it delegates
the actual completion to a client exposing ``_complete_with_model(prompt,
"table_analyzer")`` (the ``LLMClientStatementGenerator``), which loads the
``table_analyzer`` Pydantic response model and roots the response at ``tables``.

Contract (design §2a):
    Preconditions:  ``len(dfs) == len(aliases)``; each df has >= 1 column.
    Postconditions: returns exactly one analysis per alias, in alias order;
                    never reads only ``tables[0]``.
"""

import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class TableAnalyzer:
    """Produces table analyses for all aliases in one batched LLM call.

    Args:
        client: An object exposing ``_complete_with_model(prompt, model_name)``
            returning ``(result_dict, usage)`` — typically an
            ``LLMClientStatementGenerator``. The batched call uses the
            ``table_analyzer`` response model, whose payload is rooted at
            ``tables`` (a list of per-table analyses).
    """

    #: Response-model name registered on the client for table analysis.
    RESPONSE_MODEL = "table_analyzer"

    def __init__(self, client: Any):
        self._client = client
        # Token usage from the most recent batched call, so callers (e.g. the
        # BudgetGuard) can account for it without changing the return type.
        self.last_usage: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alias_names(aliases: Any) -> List[str]:
        """Return alias names in order from a dict (keys) or a sequence."""
        if isinstance(aliases, dict):
            return list(aliases.keys())
        return list(aliases)

    @staticmethod
    def _normalize_metadata(metadata: Any, alias_names: List[str]) -> List[dict]:
        """Coerce metadata into a per-alias list aligned with ``alias_names``.

        Accepts a list (already per-table), a dict keyed by alias, or ``None``.
        Missing entries default to an empty dict.
        """
        if metadata is None:
            return [{} for _ in alias_names]
        if isinstance(metadata, dict):
            return [metadata.get(alias, {}) for alias in alias_names]
        metadata_list = list(metadata)
        # Pad short lists so indexing stays safe.
        if len(metadata_list) < len(alias_names):
            metadata_list = metadata_list + [
                {} for _ in range(len(alias_names) - len(metadata_list))
            ]
        return metadata_list

    @staticmethod
    def _normalize_languages(languages: Any) -> List[str]:
        """Coerce ``languages`` into a list of language names."""
        if languages is None:
            return []
        if isinstance(languages, str):
            return [languages]
        return list(languages)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_batch_prompt(
        self,
        alias_names: List[str],
        columns_per_table: List[List[str]],
        samples_per_table: List[List[dict]],
        metadata_list: List[dict],
        languages: List[str],
    ) -> str:
        """Build a single prompt carrying every table's columns/samples/metadata."""
        table_blocks = []
        for idx, alias in enumerate(alias_names):
            block = (
                f"Alias: {alias}"
                f"\nColumns:\n{json.dumps(columns_per_table[idx], indent=2, ensure_ascii=False)}"
                f"\nMetadata:\n{json.dumps(metadata_list[idx], indent=2, ensure_ascii=False, default=str)}"
                f"\nSample rows:\n{json.dumps(samples_per_table[idx], indent=2, ensure_ascii=False, default=str)}"
            )
            table_blocks.append(block)

        joined_blocks = "\n\n".join(table_blocks)
        return (
            "Analyze ALL of the following tables and return a JSON object matching "
            "the provided schema. Return exactly one analysis entry per table, in "
            "the same order the tables are listed, echoing each table's alias "
            "unchanged. Do not include any explanatory text outside the JSON.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Produce one analysis object per table under the 'tables' key.\n"
            "- For each table, extract up to 10 keywords (max) that best capture "
            "the essential and domain concepts in that table.\n"
            "- Keywords should be meaningful terms a domain expert would use to "
            "describe what the table contains, and help non-expert users "
            "understand the data.\n"
            "- Analyze every table independently; do not merge or skip tables.\n"
            f"\n\nAliases (in order): {json.dumps(alias_names, ensure_ascii=False)}"
            f"\nDetected languages: {json.dumps(languages, ensure_ascii=False)}"
            f"\n\nTables:\n{joined_blocks}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        dfs: List[Any],
        aliases: Any,
        metadata: Any = None,
        languages: Any = None,
    ) -> List[dict]:
        """Analyse every table in one batched LLM call.

        Args:
            dfs: The prepared DataFrames, one per alias, in alias order.
            aliases: Alias names as a sequence, or a dict whose keys are aliases.
            metadata: Per-table metadata as a list (aligned with aliases) or a
                dict keyed by alias. Missing entries default to ``{}``.
            languages: Detected languages as a list or a single string.

        Returns:
            ``TableAnalyses.tables``: a list with exactly one analysis dict per
            alias, in alias order. Any alias the model omits is filled with a
            default empty analysis so the one-per-alias contract always holds.

        Raises:
            ValueError: If ``len(dfs) != len(aliases)`` (Requirement 4.4), or if
                any DataFrame has no columns (design precondition).
        """
        alias_names = self._alias_names(aliases)

        # --- Preconditions -------------------------------------------------
        if len(dfs) != len(alias_names):
            raise ValueError(
                "TableAnalyzer.analyze_batch: number of data frames "
                f"({len(dfs)}) must equal the number of aliases "
                f"({len(alias_names)})."
            )
        for idx, df in enumerate(dfs):
            if len(df.columns) < 1:
                raise ValueError(
                    "TableAnalyzer.analyze_batch: table "
                    f"{alias_names[idx]!r} has no columns; each data frame must "
                    "have at least one column."
                )

        metadata_list = self._normalize_metadata(metadata, alias_names)
        detected_languages = self._normalize_languages(languages)

        columns_per_table = [
            [f"{col} ({df[col].dtype})" for col in df.columns] for df in dfs
        ]
        samples_per_table = [df.head(3).to_dict(orient="records") for df in dfs]

        prompt = self._build_batch_prompt(
            alias_names,
            columns_per_table,
            samples_per_table,
            metadata_list,
            detected_languages,
        )

        # --- Single batched LLM call (Requirement 4.1) ---------------------
        result, usage = self._client._complete_with_model(prompt, self.RESPONSE_MODEL)
        if isinstance(usage, dict):
            self.last_usage = usage

        tables = []
        if isinstance(result, dict):
            tables = result.get("tables", []) or []

        # --- Reconcile to exactly one analysis per alias, in alias order ---
        # (Requirements 4.2, 4.3.) Map returned analyses by alias so an
        # out-of-order or partial response is still projected onto the alias
        # sequence; missing aliases get a default empty analysis.
        analyses_by_alias: dict = {}
        for entry in tables:
            if not isinstance(entry, dict):
                continue
            key = entry.get("alias")
            if key is not None and key not in analyses_by_alias:
                analyses_by_alias[key] = entry

        # Positional fallback for entries whose alias the model failed to echo.
        unlabeled = [
            entry
            for entry in tables
            if isinstance(entry, dict) and entry.get("alias") is None
        ]

        ordered: List[dict] = []
        for idx, alias in enumerate(alias_names):
            entry = analyses_by_alias.get(alias)
            if entry is None and idx < len(unlabeled):
                entry = {**unlabeled[idx], "alias": alias}
            if entry is None:
                logger.warning(
                    "TableAnalyzer.analyze_batch: no analysis returned for alias "
                    "%r; using an empty default.",
                    alias,
                )
                entry = {
                    "alias": alias,
                    "table_description": "",
                    "table_keywords": [],
                }
            ordered.append(entry)

        return ordered
