"""Batched table analysis.

Analysing one table per LLM call would scale analysis cost with the number of
tables. :class:`TableAnalyzer` instead replaces that per-table loop with a
single batched call: one prompt carries the columns, sample rows, and
metadata for *every* table and the model returns a ``TableAnalyses`` payload
(one entry per alias).

The analyzer owns a dedicated :class:`TableAnalyzerLLMClient` rather than
sharing the generation pipeline's ``LLMClientStatementGenerator`` instance.
That shared client's ``_complete_with_model`` swaps its ``response_model``
attribute in place (save/restore around each call) to serve several response
models from one config load — safe only because today's pipeline calls it
sequentially, never concurrently. A dedicated client with a ``response_model``
fixed once at construction has no such shared mutable state, at the cost of
one extra client/config load.

Contract (design §2a):
    Preconditions:  ``len(dfs) == len(aliases)``; each df has >= 1 column.
    Postconditions: returns exactly one analysis per alias, in alias order;
                    never reads only ``tables[0]``.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..prompting.prompts import TableAnalyzerBatchPrompt
from ...utils import shield_dataframe_for_prompt

logger = logging.getLogger(__name__)


class TableAnalyzerLLMClient(LLMClientStructured):
    """Dedicated LLM client for batched table analysis.

    Fixes ``response_model`` to the ``table_analyzer`` YAML config section
    (``TableAnalyses``) once at construction — unlike
    ``LLMClientStatementGenerator._complete_with_model``, nothing here mutates
    shared state per call, so an instance is safe to use however its owner
    (:class:`TableAnalyzer`) is used.
    """

    def __init__(self, config_path: Path):
        super().__init__(config_path, "table_analyzer")

    def complete(self, prompt: str, **kwargs) -> Any:
        """Complete against the fixed ``table_analyzer`` model, rooted at ``tables``."""
        return super().complete(prompt, root_key="tables", **kwargs)


class TableAnalyzer:
    """Produces table analyses for all aliases in one batched LLM call.

    Args:
        config_path: Path to the LLM YAML config, used to construct the
            dedicated :class:`TableAnalyzerLLMClient` when ``client`` is not
            injected.
        client: Optional pre-built client exposing ``complete(prompt)`` ->
            ``(result_dict, usage)``, rooted at ``tables``. Injected for
            testing; defaults to a real :class:`TableAnalyzerLLMClient`.
        cache_path: Optional JSON file (conventionally
            ``<candidates_discovery>/table_analysis_cache.json``) caching
            per-table analyses as ``table_id -> model -> {description,
            keywords}``. A table already analysed by the configured model is
            served from this cache instead of a new LLM call; only the
            never-seen tables of a batch go to the model. ``None`` disables
            caching entirely.
    """

    def __init__(
        self,
        config_path: Path,
        client: Optional[Any] = None,
        cache_path: Optional[Path] = None,
    ):
        self._client = client or TableAnalyzerLLMClient(config_path)
        self._cache_path = Path(cache_path) if cache_path is not None else None
        # Token usage from the most recent batched call, so callers can
        # accumulate it into the run's total token usage without changing
        # the return type.
        self.last_usage: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Analysis cache (table id -> model -> {description, keywords})
    # ------------------------------------------------------------------

    @property
    def _cache_model(self) -> str:
        """The model id new cache entries are keyed under.

        The configured PRIMARY model: the router may occasionally serve a call
        from a fallback, but which model actually answered is not reported
        back, so the primary is the honest stable key for a run's entries.
        """
        config = getattr(self._client, "config", None) or {}
        return str(config.get("model", "unknown"))

    def is_cached(self, table_id: str) -> bool:
        """True when ``table_id`` already has a cached analysis for the
        configured model — callers can skip loading the table entirely."""
        if self._cache_path is None:
            return False
        return self._cache_lookup(self._load_cache(), str(table_id), "") is not None

    def _load_cache(self) -> dict:
        """Read the cache JSON; a missing or corrupt file is an empty cache."""
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            return cache if isinstance(cache, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Table-analysis cache %s is unreadable (%s); ignoring it.",
                self._cache_path, exc,
            )
            return {}

    def _save_cache(self, cache: dict) -> None:
        """Atomically persist the cache (tmp file + rename, never half-written)."""
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=self._cache_path.parent, prefix=self._cache_path.name, suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self._cache_path)
        except OSError as exc:
            logger.warning(
                "Could not persist table-analysis cache %s: %s",
                self._cache_path, exc,
            )

    def _cache_lookup(self, cache: dict, table_id: str, alias: str) -> Optional[dict]:
        """Return the cached analysis for (table_id, configured model), if any."""
        entry = (cache.get(table_id) or {}).get(self._cache_model)
        if not isinstance(entry, dict):
            return None
        description = entry.get("description", "")
        keywords = entry.get("keywords", [])
        if not description and not keywords:
            return None
        return {
            "alias": alias,
            "table_description": description,
            "table_keywords": list(keywords),
        }

    def _cache_store(self, cache: dict, table_id: str, analysis: dict) -> bool:
        """Record a fresh analysis; empty (defaulted) analyses never poison the cache."""
        description = analysis.get("table_description", "") or ""
        keywords = list(analysis.get("table_keywords", []) or [])
        if not description and not keywords:
            return False
        cache.setdefault(table_id, {})[self._cache_model] = {
            "description": description,
            "keywords": keywords,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return True

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
            return [metadata.get(alias) or {} for alias in alias_names]
        metadata_list = list(metadata)
        # Pad short lists so indexing stays safe.
        if len(metadata_list) < len(alias_names):
            metadata_list = metadata_list + [
                {} for _ in range(len(alias_names) - len(metadata_list))
            ]
        # Coerce None entries (datasets with no metadata) to {} so the prompt
        # never renders a bare "Metadata: null" block.
        return [m or {} for m in metadata_list]

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
        return TableAnalyzerBatchPrompt().update(
            aliases=json.dumps(alias_names, ensure_ascii=False),
            languages=json.dumps(languages, ensure_ascii=False),
            tables=joined_blocks,
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

        # Table ids for the cache: the dataset NAME each alias points at when
        # ``aliases`` is a mapping (alias names like Table_0 are per-run and
        # would never hit twice), otherwise the alias itself.
        if isinstance(aliases, dict):
            table_ids = [str(aliases[alias]) for alias in alias_names]
        else:
            table_ids = [str(alias) for alias in alias_names]

        # --- Cache partition: already-seen tables skip the LLM entirely ----
        cache = self._load_cache() if self._cache_path is not None else {}
        cached_by_alias: dict = {}
        miss_indices: List[int] = []
        for idx, alias in enumerate(alias_names):
            hit = (
                self._cache_lookup(cache, table_ids[idx], alias)
                if self._cache_path is not None
                else None
            )
            if hit is not None:
                cached_by_alias[alias] = hit
            else:
                miss_indices.append(idx)
        if cached_by_alias:
            logger.info(
                "TableAnalyzer: %d/%d table(s) served from the analysis cache "
                "(%s), %d to analyse.",
                len(cached_by_alias), len(alias_names),
                self._cache_path, len(miss_indices),
            )

        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # --- Single batched LLM call over the MISSES only (Requirement 4.1)
        fresh_by_alias: dict = {}
        if miss_indices:
            miss_aliases = [alias_names[i] for i in miss_indices]
            columns_per_table = [
                [f"{col} ({dfs[i][col].dtype})" for col in dfs[i].columns]
                for i in miss_indices
            ]
            samples_per_table = [
                shield_dataframe_for_prompt(dfs[i].head(3)).to_dict(orient="records")
                for i in miss_indices
            ]
            miss_metadata = [metadata_list[i] for i in miss_indices]

            prompt = self._build_batch_prompt(
                miss_aliases,
                columns_per_table,
                samples_per_table,
                miss_metadata,
                detected_languages,
            )

            result, usage = self._client.complete(prompt)
            if isinstance(usage, dict):
                self.last_usage = usage

            tables = []
            if isinstance(result, dict):
                tables = result.get("tables", []) or []

            fresh = self._reconcile(tables, miss_aliases)
            fresh_by_alias = {entry["alias"]: entry for entry in fresh}

            # Persist the new analyses under table_id -> model -> {...};
            # defaulted-empty entries are never cached so a failed call can't
            # poison future runs.
            if self._cache_path is not None:
                stored = False
                for idx, alias in zip(miss_indices, miss_aliases):
                    stored |= self._cache_store(
                        cache, table_ids[idx], fresh_by_alias[alias]
                    )
                if stored:
                    self._save_cache(cache)

        # --- Merge: exactly one analysis per alias, in alias order ---------
        return [
            cached_by_alias.get(alias) or fresh_by_alias[alias]
            for alias in alias_names
        ]

    @staticmethod
    def _reconcile(tables: List[Any], alias_names: List[str]) -> List[dict]:
        """Project raw LLM analyses onto ``alias_names``, one entry per alias.

        (Requirements 4.2, 4.3.) Maps returned analyses by alias so an
        out-of-order or partial response is still projected onto the alias
        sequence; entries whose alias the model failed to echo are matched
        positionally, and any alias still missing gets an empty default.
        """
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
