"""Batched table analysis.

Analysing one table per LLM call would scale analysis cost with the number of
tables. :class:`TableAnalyzer` instead replaces that per-table loop with a
single batched call: one prompt carries the columns, scope and breakdowns (see
:func:`render_table_facts`), a random sample of rows, and metadata for *every*
table and the model returns a ``TableAnalyses`` payload (one entry per alias).

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

import pandas as pd
from pandas.api import types as ptypes

from ..retrieval.text import tokenize

from ..agent.llm_client.LLMClientStructured import LLMClientStructured
from ..agent.prompting.prompts import TableAnalyzerBatchPrompt
from . import shield_dataframe_for_prompt, summarize_large_value

logger = logging.getLogger(__name__)

# Bump when the analysis prompt, or the metadata it is shown, changes in a way
# that should regenerate cached descriptions/keywords: entries stored under
# another version are cache misses.
ANALYSIS_VERSION = 3

# The only metadata fields shown to the analyzer: what the table is, who is
# responsible for it and which period it covers. ``title`` and ``description``
# are shared by every file of a dataset, while ``resource_name`` and
# ``resource_description`` describe this file and are often the only place its
# snapshot date appears ("2021-12-31 Organogram (Junior)"). Ids, URLs, format
# and publication dates add noise, and a publication date is easily mistaken
# for the period the data covers.
DISTINGUISHING_METADATA_FIELDS = (
    "title",
    "resource_name",
    "resource_description",
    "description",
    "publisher",
    "responsible_entity",
    "temporal_coverage",
    "tags",
)

# Rows shown per table, drawn at random: the first rows of a sorted file share
# one unit, region or year, which then reads as the scope of the whole table.
SAMPLE_ROWS = 5
# A varying column with at most this many distinct values is a breakdown.
BREAKDOWN_MAX_VALUES = 100
# A breakdown with at most this many values has them all listed. A numeric
# column with more values is a measure rather than a breakdown.
BREAKDOWN_LISTED_VALUES = 10


def distinguishing_metadata(metadata: dict) -> dict:
    """Project normalized metadata onto the fields that tell a table apart."""
    return {
        key: metadata[key]
        for key in DISTINGUISHING_METADATA_FIELDS
        if metadata.get(key) not in (None, "", [], "N/A")
    }


# The metadata shown next to each table's analysis to the planner and the plan
# judges: the portal's own name, publisher and period for the file, so a
# question's vintage and program come from the source rather than from the
# analysis summary. The dataset description is left out: it is long, shared by
# every file of the dataset, and can state a scope the table does not have.
PORTAL_METADATA_FIELDS = (
    "title",
    "resource_name",
    "resource_description",
    "publisher",
    "responsible_entity",
    "temporal_coverage",
)


def portal_metadata(metadata: dict) -> dict:
    """Project normalized metadata onto the short fields naming a table's source."""
    return {
        key: summarize_large_value(metadata[key])
        for key in PORTAL_METADATA_FIELDS
        if metadata.get(key) not in (None, "", [], "N/A")
    }


# The metadata shown wherever a whole portal record would otherwise be dumped
# into a prompt — statement generation and the discovery prompts, all of which
# render it through ``DatasetDescription``. Same reasoning as
# DISTINGUISHING_METADATA_FIELDS above, kept as its own list because that one
# is tied to ANALYSIS_VERSION: changing what the analyzer sees invalidates
# cached descriptions, which has nothing to do with what a generation prompt
# should show.
#
# `columns` is deliberately absent: every template that renders this also
# prints the table's real schema right below it (DatasetDescription's own
# "Column Details", the light schema block), so the record's column list is
# redundant where it exists at all — and on CKAN portals it is empty in every
# single record. Ids, URLs, `format` and the publication timestamps are left
# out as noise; `created_at`/`modified_at` are worse than noise, being easy to
# read as the period the data covers.
PROMPT_METADATA_FIELDS = (
    "title",
    "resource_name",
    "resource_description",
    "description",
    "publisher",
    "responsible_entity",
    "temporal_coverage",
    "tags",
)


def prompt_metadata(metadata: dict) -> dict:
    """Project a portal record onto the fields worth showing a prompt.

    Empty fields are dropped rather than rendered: a portal that publishes
    none of them would otherwise spend the prompt's attention on
    ``responsible_entity: None, tags: [], temporal_coverage: None`` — on the
    UK CKAN corpus those are empty in 100%, 67% and 97% of records.
    """
    return {
        key: summarize_large_value(metadata[key])
        for key in PROMPT_METADATA_FIELDS
        if metadata.get(key) not in (None, "", [], "N/A")
    }


def scope_values(df: Any) -> dict:
    """``{column: value}`` for every column holding exactly ONE distinct
    (non-null) value across the whole table — the same single-value test
    ``render_table_facts`` reports as "Scope", factored out so a caller that
    only needs the raw values (see ``orqa.retrieval.gate
    .build_contract``'s data-scope facet fallback) doesn't have to re-derive
    them by parsing that prose back out.
    """
    result: dict = {}
    for column in df.columns:
        values = df[column].dropna()
        if int(values.nunique()) == 1:
            result[column] = summarize_large_value(str(values.iloc[0]))
    return result


def render_table_facts(df: Any) -> str:
    """Render what every row of a table shares and what the rows span.

    Computed over all rows, so the analyzer can tell the table's scope from
    values that merely appear in its sample rows: a column with a single value
    is scope ("Organisation = Home Office in all 5,473 rows"), a column with
    few values is a breakdown the table covers in full ("Unit: 56 values").
    """
    num_rows = len(df)
    scope: List[str] = []
    breakdowns: List[str] = []
    scope_column_values = scope_values(df)
    for column in df.columns:
        values = df[column].dropna()
        distinct = int(values.nunique())
        if column in scope_column_values:
            value = scope_column_values[column]
            if len(values) == num_rows:
                rows = f"all {num_rows:,} rows"
            else:
                rows = f"{len(values):,} of {num_rows:,} rows, the others empty"
            scope.append(f"- {column} = {value} in {rows}")
        elif distinct <= BREAKDOWN_LISTED_VALUES:
            if distinct:
                listed = ", ".join(_sorted_values(values))
                breakdowns.append(f"- {column}: {distinct} values ({listed})")
        elif distinct <= BREAKDOWN_MAX_VALUES and not (
            ptypes.is_numeric_dtype(values) and not ptypes.is_bool_dtype(values)
        ):
            breakdowns.append(f"- {column}: {distinct} values")

    return "\n".join(
        [
            f"Rows: {num_rows:,}",
            "Scope (columns with a single value):",
            *(scope or ["- none"]),
            "Breakdowns (columns with few values; the table covers all of them):",
            *(breakdowns or ["- none"]),
        ]
    )


# ---------------------------------------------------------------------------
# Table identity: what tells one table from its siblings. On a family of
# monthly/quarterly files of one publisher the analyses, the columns and the
# scope facts are all alike; what differs is the PERIOD the rows cover and a
# word or two of the file name. The question writer needs both to name each
# table's period in the question (topic linkage) and to make every table
# necessary (its period must be in the question's scope); the question judge
# tests exactly that. Measured in experiments/structured_stability round 2
# (2026-11-03): topic-linkage and table-necessity rejections (50 of 121) were
# mostly a period the writer got wrong or left out.
_YEAR_COLUMN_HINTS = ("year", "anno", "año", "annee", "année", "jahr", "fiscal", "ejercicio")
_DATE_SAMPLE = 400
_DATE_PARSE_RATIO = 0.9
_PLAIN_NUMBER_RE = r"\s*-?\d+(\.\d+)?\s*"


def _parse_dates(values: Any) -> Any:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=True)


def table_period(df: Any) -> Optional[dict]:
    """The span of dates the table's rows cover, or ``None``.

    ``{"column", "start", "end", "kind"}`` — ``kind`` ``"date"`` from the
    text/datetime column whose values parse best as dates (detected on a
    sample, measured over every row), else ``"year"`` from an integer column
    whose name says year and whose values are years.
    """
    if df is None or len(getattr(df, "columns", [])) == 0:
        return None
    best: Optional[tuple] = None
    for column in df.columns:
        series = df[column]
        try:
            if ptypes.is_datetime64_any_dtype(series):
                values = series.dropna()
                if values.empty:
                    continue
                candidate = (1.0, column, values.min(), values.max())
            elif ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series):
                values = series.dropna().astype(str)
                if values.empty:
                    continue
                sample = values.sample(n=min(_DATE_SAMPLE, len(values)), random_state=0)
                if sample.str.fullmatch(_PLAIN_NUMBER_RE).mean() > 0.5:
                    continue
                parsed = _parse_dates(sample)
                ratio = float(parsed.notna().mean())
                if ratio < _DATE_PARSE_RATIO:
                    continue
                full = _parse_dates(values).dropna()
                if full.empty:
                    continue
                candidate = (ratio, column, full.min(), full.max())
            else:
                continue
        except Exception:  # noqa: BLE001 — a column that cannot be parsed is simply not the date column
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is not None:
        _, column, start, end = best
        return {"column": str(column), "start": start.date().isoformat(), "end": end.date().isoformat(), "kind": "date"}
    for column in df.columns:
        name = str(column).lower()
        if not any(hint in name for hint in _YEAR_COLUMN_HINTS):
            continue
        series = df[column]
        if not ptypes.is_numeric_dtype(series) or ptypes.is_bool_dtype(series):
            continue
        values = series.dropna()
        if values.empty or values.between(1900, 2100).mean() < 0.95:
            continue
        return {"column": str(column), "start": str(int(values.min())), "end": str(int(values.max())), "kind": "year"}
    return None


def table_identity(alias: str, df: Any, metadata: Optional[dict], siblings: Any = ()) -> dict:
    """What tells ``alias`` apart: the period its rows cover, its file name and
    publisher, and the title/file-name words no other table of the group has.

    ``metadata`` is the table's portal metadata (``portal_metadata``);
    ``siblings`` the other tables' metadata dicts.
    """
    md = dict(metadata or {})
    name_text = f"{md.get('title') or ''} {md.get('resource_name') or ''}"
    own_tokens = tokenize(name_text)
    other_tokens: set = set()
    for sib in siblings or ():
        sib = sib or {}
        other_tokens |= set(tokenize(f"{sib.get('title') or ''} {sib.get('resource_name') or ''}"))
    distinct = list(dict.fromkeys(t for t in own_tokens if t not in other_tokens))
    return {
        "alias": alias,
        "period": table_period(df),
        "temporal_coverage": md.get("temporal_coverage") or None,
        "resource_name": md.get("resource_name") or None,
        "title": md.get("title") or None,
        "publisher": md.get("publisher") or md.get("responsible_entity") or None,
        "distinct_words": distinct[:8],
    }


def table_identities(aliases: Any, dfs: Any, table_metadata: Optional[dict]) -> List[dict]:
    """``table_identity`` for every alias of a group, siblings included."""
    alias_list = list(aliases)
    metadata = table_metadata or {}
    frames = list(dfs or [])
    out: List[dict] = []
    for idx, alias in enumerate(alias_list):
        df = frames[idx] if idx < len(frames) else None
        siblings = [metadata.get(other) or {} for other in alias_list if other != alias]
        out.append(table_identity(alias, df, metadata.get(alias) or {}, siblings))
    return out


def render_identity_line(identity: dict, group_size: int = 1) -> str:
    parts: List[str] = []
    period = identity.get("period")
    if period and period.get("kind") == "date":
        parts.append(f"rows dated {period['start']} to {period['end']} (column {period['column']})")
    elif period:
        parts.append(f"years {period['start']} to {period['end']} (column {period['column']})")
    elif identity.get("temporal_coverage"):
        parts.append(f"declared coverage {identity['temporal_coverage']}")
    else:
        parts.append("no dated rows found — a fixed period only if the file name states one")
    if identity.get("resource_name"):
        parts.append(f"file \"{identity['resource_name']}\"")
    if identity.get("publisher"):
        parts.append(f"publisher {identity['publisher']}")
    words = identity.get("distinct_words") or []
    if words:
        parts.append("words only this table's title/file name has: " + ", ".join(words))
    elif group_size > 1:
        parts.append("its title/file name shares every word with another table of the group")
    return "; ".join(parts)


def render_table_identities(identities: Any) -> str:
    """The "### TABLE IDENTITY" block: one line per table."""
    identities = list(identities or [])
    if not identities:
        return "(none)"
    return "\n".join(
        f"- {i['alias']}: {render_identity_line(i, len(identities))}" for i in identities
    )


def _sorted_values(values: Any) -> List[str]:
    """Distinct values of a series, in natural order when they compare."""
    unique = list(values.unique())
    try:
        unique.sort()
    except TypeError:
        unique.sort(key=str)
    return [summarize_large_value(str(value)) for value in unique]


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
        seed: Seed for the random sample of rows shown per table, so reruns
            show the same rows.
    """

    def __init__(
        self,
        config_path: Path,
        client: Optional[Any] = None,
        cache_path: Optional[Path] = None,
        seed: int = 0,
    ):
        self._client = client or TableAnalyzerLLMClient(config_path)
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._seed = seed
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
        if not isinstance(entry, dict) or entry.get("version") != ANALYSIS_VERSION:
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
            "version": ANALYSIS_VERSION,
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
        facts_per_table: List[str],
        samples_per_table: List[List[dict]],
        metadata_list: List[dict],
        languages: List[str],
    ) -> str:
        """Build a single prompt carrying every table's columns/facts/samples/metadata."""
        table_blocks = []
        for idx, alias in enumerate(alias_names):
            block = (
                f"Alias: {alias}"
                f"\nColumns:\n{json.dumps(columns_per_table[idx], indent=2, ensure_ascii=False)}"
                f"\nMetadata:\n{json.dumps(distinguishing_metadata(metadata_list[idx]), indent=2, ensure_ascii=False, default=str)}"
                f"\n{facts_per_table[idx]}"
                f"\nRandom sample rows:\n{json.dumps(samples_per_table[idx], indent=2, ensure_ascii=False, default=str)}"
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
            facts_per_table = [render_table_facts(dfs[i]) for i in miss_indices]
            samples_per_table = [
                shield_dataframe_for_prompt(
                    dfs[i].sample(n=min(SAMPLE_ROWS, len(dfs[i])), random_state=self._seed)
                ).to_dict(orient="records")
                for i in miss_indices
            ]
            miss_metadata = [metadata_list[i] for i in miss_indices]

            prompt = self._build_batch_prompt(
                miss_aliases,
                columns_per_table,
                facts_per_table,
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
