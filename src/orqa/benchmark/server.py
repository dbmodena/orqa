"""
MCP server exposing one city/source's benchmark: its dataset reverse
index plus the todo list of natural language questions to answer.

Started as a workflow step from main.py, which selects the city and its
workflow yaml (where the port, statement kind and portal language are
configured):

    uv run src/main.py --country usa --city nyc --steps mcp_search --mode stdio
    uv run src/main.py --country france --city paris --steps mcp_search --mode port

In "stdio" mode the client spawns the command above directly; in "port"
mode the server listens with the streamable HTTP transport on the port
set in the workflow yaml (tasks.mcp_search.port) and clients connect to
http://<host>:<port>/mcp.

At startup the materialized index under <data_path>/index/ is loaded if
it is up to date with the normalized metadata, and (re)built otherwise.
Use --mode build to only materialize the index and exit. When the
generate-statements output exists (generated_queries.json or
generated_queries_semantic.json, whichever
tasks.candidates_discovery.method selects), its questions are loaded
fresh (organized by id, phrased in the portal's target language) and
synced into the list under <data_path>/benchmark/<kind>/.

The tools drive an agent through the retrieval benchmark loop: fetch a
question (get_question hands out each one, once), search_datasets ranks
its candidate tables and relates them to each other on the fly (backed by
get_table_info/get_dataset_metadata/preview_dataset/compare_tables for
closer inspection), and evaluate_tables checks that selection — tables
AND, per table, the columns used — against the hidden ground truth. This
is retrieval-only for now — there is no code-writing, execution or result
reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP

from orqa.benchmark.index import DatasetIndex
from orqa.benchmark.questions import BenchmarkTodo, evaluate_table_retrieval, load_questions

# Row cap applied when a tool needs the actual data (preview, valentine
# comparison): enough rows to be representative, bounded read time.
_ROW_CAP = 10_000

# search_datasets computes relationships for every pair among its ranked
# results, on the fly, inline in the response: a much smaller cap keeps
# that bounded even at the default top_k=10 (up to 45 pairs). It's a
# quick screen, not a precise measurement — call compare_tables (full
# _ROW_CAP) for the reliable figure on a promising pair.
_RELATIONSHIP_ROW_CAP = 500

# Valentine's per-pair cost (even schema-only matchers) scales with the
# column-count product, not row count: a table with a few hundred
# columns (some Socrata datasets are this wide) takes 10s-plus per pair
# regardless of its partner's width, and can single-handedly stall
# search_datasets. Datasets over this width are excluded from the
# automatic relationships (still returned in `results`); compare_tables
# still works on them directly, at that cost, if deliberately called.
_RELATIONSHIP_MAX_COLUMNS = 60

# Read options shared by every tool that opens a dataset CSV.
_READ_OPTS = dict(
    sep=None,
    engine="python",
    encoding_errors="ignore",
    on_bad_lines="skip",
)


def _log(message: str) -> None:
    # Never write diagnostics to stdout: in stdio mode it carries the
    # MCP JSON-RPC stream.
    print(message, file=sys.stderr)


def _jsonable(value):
    """Coerce a pandas/numpy scalar (from .min()/.max()/etc.) to a plain
    Python type the MCP JSON-RPC transport can serialize."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        value = round(value, 4)
    return value


def _column_info(series: pd.Series) -> dict:
    """
    One column's descriptive stats: dtype, null/distinct counts, and
    either min/max (numeric or datetime-like columns) or the most frequent
    values (everything else) — enough to judge what a column's values look
    like, and whether it's the kind of thing a question would filter or
    aggregate on, without reading raw rows.
    """
    n_null = int(series.isna().sum())
    non_null = series.dropna()
    info = {
        "dtype": str(series.dtype),
        "n_null": n_null,
        "n_distinct": int(non_null.nunique()),
    }
    if non_null.empty:
        return info

    if pd.api.types.is_numeric_dtype(series):
        info["min"] = _jsonable(non_null.min())
        info["max"] = _jsonable(non_null.max())
        return info

    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    if parsed.notna().mean() > 0.9:
        info["min"] = str(parsed.min())
        info["max"] = str(parsed.max())
        return info

    top = non_null.astype(str).value_counts().head(5)
    info["top_values"] = [{"value": v, "count": int(c)} for v, c in top.items()]
    return info


def _valentine_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, matcher: str, top_k: int) -> dict:
    """
    Valentine schema matching between two already-loaded frames: ranked
    column matches, the best-key join and its row count, the union column
    ratio, and Pearson correlations between matched numeric columns.

    Shared by compare_tables (one deliberate pair, full detail) and
    search_datasets (every pair among the ranked results, on the fly).
    """
    from valentine import valentine_match

    from orqa.schema_matching.valentine_matcher import THRESHOLD, instantiate_matcher

    # NB: valentine indexes the second character of the table name for its
    # column guids, so names must be at least two characters long.
    raw = valentine_match(df_a, df_b, instantiate_matcher(matcher), "table_a", "table_b")
    matches = sorted(
        (
            {"column_a": col_a, "column_b": col_b, "score": round(score, 4)}
            for ((_, col_a), (_, col_b)), score in raw.items()
        ),
        key=lambda m: m["score"],
        reverse=True,
    )

    join_info = None
    correlations = []
    if matches:
        key_a, key_b = matches[0]["column_a"], matches[0]["column_b"]
        try:
            merged = df_a.merge(
                df_b, left_on=key_a, right_on=key_b, suffixes=("_a", "_b")
            )
        except (ValueError, TypeError) as exc:
            join_info = {"left_on": key_a, "right_on": key_b, "error": str(exc)}
        else:
            join_info = {
                "left_on": key_a,
                "right_on": key_b,
                "score": matches[0]["score"],
                "joined_rows": int(len(merged)),
            }
            for m in matches[1:]:
                col_a = m["column_a"] if m["column_a"] in merged.columns else f"{m['column_a']}_a"
                col_b = m["column_b"] if m["column_b"] in merged.columns else f"{m['column_b']}_b"
                if col_a not in merged.columns or col_b not in merged.columns:
                    continue
                if not (
                    pd.api.types.is_numeric_dtype(merged[col_a])
                    and pd.api.types.is_numeric_dtype(merged[col_b])
                ):
                    continue
                pearson = merged[col_a].corr(merged[col_b])
                if pd.notna(pearson):
                    correlations.append(
                        {
                            "column_a": m["column_a"],
                            "column_b": m["column_b"],
                            "pearson": round(float(pearson), 4),
                        }
                    )

    confident_a = {m["column_a"] for m in matches if m["score"] >= THRESHOLD}
    union_ratio = round(len(confident_a) / len(df_a.columns), 4) if len(df_a.columns) else 0.0

    return {
        "column_matches": matches[:top_k],
        "join": join_info,
        "union_column_ratio": union_ratio,
        "correlations": correlations,
    }


def _read_dataset(
    index,
    resource_id: str,
    seed: int,
    n_rows: Optional[int] = None,
    row_cap: int = _ROW_CAP,
    limit_to_n_columns: Optional[int] = None,
):
    """
    Load a dataset's CSV (capped at `row_cap` rows at parse time — the
    cheap way to bound read cost, unlike sampling down after a full
    parse), optionally capped to its first `limit_to_n_columns` columns
    (file order — the same column-count knob the rest of the pipeline
    uses, so this stays consistent with what other stages see for the
    same table), and optionally take a seeded sample of `n_rows` from
    that.
    """
    record = index.get(resource_id)
    if record is None:
        raise ValueError(f"No dataset found with resource_id {resource_id!r}")

    filepath = index.dataset_filepath(resource_id)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found on disk: {filepath}")

    df = pd.read_csv(filepath, nrows=row_cap, **_READ_OPTS)
    if limit_to_n_columns is not None:
        df = df[df.columns[:limit_to_n_columns]]
    if n_rows is not None and n_rows < len(df):
        # Seeded sampling keeps the selected rows identical across calls,
        # sessions and sibling workflows sharing the same yaml seed, so
        # benchmark runs stay comparable. sort_index restores file order.
        df = df.sample(n=n_rows, random_state=seed).sort_index()
    return df, filepath


def create_server(
    index,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    language: str = "English",
    kind: str = "PANDAS",
    seed: int = 42,
    limit_to_n_columns: int = 20,
    todo: Optional[BenchmarkTodo] = None,
) -> FastMCP:
    """
    Build the MCP benchmark server for one city's dataset index.

    `index` is a DatasetIndex or an es_index.ESDatasetIndex (they share
    the search/get/dataset_filepath/len interface). `language` is the
    portal's target language (tasks.query_generation.languages in the
    workflow yaml), `kind` the statement language the questions were
    generated for (PANDAS or SQL, informational only — this server is
    retrieval-only for now), `seed` the workflow seed used for
    deterministic row selection, and `limit_to_n_columns` the same
    tasks.candidates_discovery.limit_to_n_columns knob the rest of the
    pipeline (task proposer, dataset descriptions, statement generation)
    caps its column view at, so previews/stats here stay consistent with
    what every other stage ever sees for a given table. Tools are defined
    as closures over these, so the server is fully configured by the
    selected workflow with no module-level state.
    """
    source = index.source or "unknown"
    kind = kind.upper()

    question_flow = ""
    if todo is not None:
        question_flow = (
            " A list of benchmark questions is available, read fresh from "
            "the generated queries: call get_question (no arguments) to "
            "fetch the next one not yet fetched — just its id and question "
            "text, nothing else. list_questions shows every id + question "
            "if an overview is needed."
        )

    mcp = FastMCP(
        "orqa-benchmark",
        instructions=(
            f"Retrieval benchmark over the {source} open data collection "
            f"({len(index)} datasets). The portal's metadata and search "
            f"keywords are in {language}; the questions were generated for "
            f"{kind} (informational only — this server is retrieval-only "
            "for now: no code-writing, execution or result reporting)."
            f"{question_flow} "
            "The loop for each question: (1) fetch it with get_question; "
            "(2) extract its keywords (entities, topics, measures, place "
            f"and time expressions) IN {language} — translate them first "
            "when reasoning in another language, since the reverse index "
            f"only matches {language} metadata — and call search_datasets "
            "with them, which ranks the top matching datasets AND, on the "
            "fly, their pairwise join/union/correlation relationships "
            "(compare_tables gives the same detail for one deliberate "
            "pair, or a table outside the ranking); (3) call "
            "evaluate_tables with the tables you settled on — resource_id "
            "-> the columns from it you'd actually use — to check that "
            "selection against the question's hidden ground truth, table "
            "AND column level. Use get_dataset_metadata, preview_dataset "
            "and get_table_info (per-column dtype/null/distinct counts "
            "and min/max or top values, to judge which columns matter "
            "without reading raw rows) alongside search_datasets to "
            "inspect candidates."
        ),
        host=host,
        port=port,
    )

    def search_datasets(
        keywords: list[str],
        top_k: int = 10,
        only_available: bool = False,
        with_relationships: bool = True,
        matcher: str = "similarity_flooding",
    ) -> dict:
        results = index.search(keywords, top_k, only_available)

        relationships = []
        skipped_too_wide = []
        if with_relationships:
            # Relationships need real data, so only among the ranked
            # results whose CSV is actually on disk — read each once and
            # reuse it across every pair it appears in.
            available = [r for r in results if r.csv_exists]
            dfs: dict[str, pd.DataFrame] = {}
            for r in available:
                try:
                    df, _ = _read_dataset(
                        index,
                        r.resource_id,
                        seed,
                        row_cap=_RELATIONSHIP_ROW_CAP,
                        limit_to_n_columns=limit_to_n_columns,
                    )
                except Exception:
                    continue
                if df.shape[1] > _RELATIONSHIP_MAX_COLUMNS:
                    # A single very wide table can single-handedly stall
                    # every pair it's part of (cost scales with column
                    # count, not rows) — exclude it, not just this pair.
                    skipped_too_wide.append(
                        {"resource_id": r.resource_id, "n_columns": int(df.shape[1])}
                    )
                    continue
                dfs[r.resource_id] = df

            for i, a in enumerate(available):
                df_a = dfs.get(a.resource_id)
                if df_a is None:
                    continue
                for b in available[i + 1 :]:
                    df_b = dfs.get(b.resource_id)
                    if df_b is None:
                        continue
                    pair = {"resource_id_a": a.resource_id, "resource_id_b": b.resource_id}
                    try:
                        cmp = _valentine_compare(df_a, df_b, matcher, top_k=1)
                        pair["join"] = cmp["join"]
                        pair["union_column_ratio"] = cmp["union_column_ratio"]
                        pair["top_correlation"] = (
                            cmp["correlations"][0] if cmp["correlations"] else None
                        )
                    except Exception as exc:
                        pair["error"] = str(exc)
                    relationships.append(pair)

        return {
            "results": [r.to_dict() for r in results],
            "relationships": relationships,
            "relationships_skipped_too_wide": skipped_too_wide,
        }

    search_datasets.__doc__ = f"""
        Find the datasets (CSV files) most relevant to a question, and
        (by default) how they relate to each other.

        Args:
            keywords: Keywords extracted from the question, IN {language}
                (the portal's language): entities, topics, measures, place
                and time expressions. E.g. for a question about public
                lighting maintenance pass the {language} words for
                ["street lighting", "maintenance", "year"]. Translate
                keywords into {language} before searching — the reverse
                index only contains {language} metadata. Multi-word phrases
                are fine; matching is case- and accent-insensitive.
            top_k: Maximum number of results to return.
            only_available: When true, only return datasets whose CSV file
                is actually present on disk.
            with_relationships: When true (default), also compute, on the
                fly, the Valentine join/union/correlation relationship
                (see compare_tables) between every pair of ranked results
                whose CSV is on disk — read with a much smaller row cap
                ({_RELATIONSHIP_ROW_CAP}) than compare_tables and matched
                with a schema-only (not instance-based) matcher, so this
                stays fast even at top_k=10 (45 pairs); treat it as a
                quick screen and confirm a promising pair with
                compare_tables's default "jaccard_distance" (slower,
                instance-based, full sample) before relying on it.
                Each table read for this pass is capped to its first
                limit_to_n_columns columns (same knob as preview_dataset/
                get_table_info/compare_tables), which also means datasets
                wider than {_RELATIONSHIP_MAX_COLUMNS} columns are
                excluded from this automatic pass (see
                relationships_skipped_too_wide below) since Valentine's
                cost scales with column count and a single very wide
                table can stall every pair it's part of; compare_tables
                still works on them directly, at that cost. Set false to
                skip this and get results faster when you only need the
                ranking.
            matcher: Valentine matcher used for the relationships:
                "similarity_flooding" (default, schema-based, fast) or
                "coma" (schema-based); avoid "jaccard_distance",
                "cupid" and "distribution_based" here — their
                instance-based comparisons are far too slow to run
                automatically over many pairs (use compare_tables for
                those on one pair at a time instead).

        Returns:
            results: ranked matches with resource_id, title, tags,
                relevance score, the keywords that matched, and the local
                csv_path to load.
            relationships: one entry per pair of results (resource_id_a,
                resource_id_b, best-key join info, union_column_ratio,
                top_correlation — see compare_tables for how to read
                them); empty when with_relationships is false or fewer
                than two results (excluding wide ones) have their CSV on
                disk. Call compare_tables directly for the full
                column_matches list on a specific pair, a different
                matcher, or a table outside this ranking.
            relationships_skipped_too_wide: resource_id + n_columns for
                any ranked result excluded from `relationships` for being
                too wide (see with_relationships above); always empty
                when with_relationships is false.
        """
    mcp.tool()(search_datasets)

    @mcp.tool()
    def get_dataset_metadata(resource_id: str) -> dict:
        """
        Return the full normalized metadata of a dataset (title,
        description, publisher, tags, column names/types, URLs) plus its
        local CSV path.

        Args:
            resource_id: The dataset's resource id as returned by
                search_datasets.
        """
        record = index.get(resource_id)
        if record is None:
            raise ValueError(f"No dataset found with resource_id {resource_id!r}")
        return record

    @mcp.tool()
    def preview_dataset(
        resource_id: str,
        n_rows: int = 5,
        how: Literal["sample", "head"] = "sample",
    ) -> dict:
        """
        Read rows of a dataset's CSV file to inspect its actual content
        and column headers.

        Row selection is deterministic: "sample" (default) returns a
        seed-fixed random sample — the same rows on every call and in
        every sibling workflow, so runs stay comparable — while "head"
        returns the first rows of the file. Only the first
        limit_to_n_columns columns (file order — the same knob the rest
        of the pipeline uses) are ever shown; a wider table's remaining
        columns are invisible here just as they are to every other stage.

        Args:
            resource_id: The dataset's resource id as returned by
                search_datasets.
            n_rows: Number of rows to return.
            how: "sample" for the deterministic seeded sample, "head" for
                the first rows.
        """
        df, filepath = _read_dataset(
            index,
            resource_id,
            seed,
            n_rows=n_rows if how == "sample" else None,
            limit_to_n_columns=limit_to_n_columns,
        )
        if how == "head":
            df = df.head(n_rows)
        return {
            "resource_id": resource_id,
            "csv_path": str(filepath),
            "row_selection": how,
            "seed": seed if how == "sample" else None,
            "limit_to_n_columns": limit_to_n_columns,
            "columns": [str(c) for c in df.columns],
            "rows": df.astype(str).to_dict(orient="records"),
        }

    @mcp.tool()
    def get_table_info(resource_id: str, columns: Optional[list[str]] = None) -> dict:
        """
        Per-column statistics for a dataset: dtype, null count, distinct
        count, and either min/max (numeric or datetime-like columns) or
        the top 5 most frequent values (everything else).

        Computed on the same deterministic seeded read as preview_dataset
        and compare_tables, so figures are stable across calls. Use this
        to judge which columns a question would actually filter/aggregate
        on — and so which to declare in evaluate_tables's `tables` — when
        a quick distribution check is enough and reading raw rows via
        preview_dataset isn't necessary.

        Only the first limit_to_n_columns columns (file order — same
        knob as preview_dataset) are ever described; requesting a column
        beyond that window raises just like an unknown column would.

        Args:
            resource_id: The dataset's resource id as returned by
                search_datasets.
            columns: Optional subset of column names to describe; omit to
                describe every column.
        """
        df, filepath = _read_dataset(
            index, resource_id, seed, limit_to_n_columns=limit_to_n_columns
        )
        if columns is not None:
            unknown = [c for c in columns if c not in df.columns]
            if unknown:
                raise ValueError(f"Unknown column(s) for {resource_id!r}: {unknown}")
            df = df[columns]

        return {
            "resource_id": resource_id,
            "csv_path": str(filepath),
            "n_rows_sampled": int(len(df)),
            "seed": seed,
            "limit_to_n_columns": limit_to_n_columns,
            "columns": {str(c): _column_info(df[c]) for c in df.columns},
        }

    @mcp.tool()
    def compare_tables(
        resource_id_a: str,
        resource_id_b: str,
        matcher: str = "jaccard_distance",
        top_k: int = 10,
    ) -> dict:
        """
        Check how two datasets relate — join, union and correlation —
        using Valentine schema matching.

        Interpreting the result in pandas terms:
        - join: the top column match is the best key candidate for
          df_a.merge(df_b, left_on=..., right_on=...); "joined_rows" is
          how many rows that merge yields on the sampled data.
        - union: "union_column_ratio" is the fraction of columns of A
          with a confident counterpart in B — high values mean the tables
          are concatenable with pd.concat after renaming matched columns.
        - correlation: Pearson corr() between matched numeric column
          pairs, computed on the frames joined via the best key.

        Rows are read with the same deterministic seeded cap as
        preview_dataset, so results are reproducible. Each table is also
        capped to its first limit_to_n_columns columns (file order, same
        knob as preview_dataset/get_table_info), so matches never
        reference a column outside what every other stage of the
        pipeline sees.

        Args:
            resource_id_a: First dataset's resource id.
            resource_id_b: Second dataset's resource id.
            matcher: Valentine matcher to use: "jaccard_distance"
                (default, instance-based), "coma", "cupid",
                "similarity_flooding" or "distribution_based".
            top_k: Maximum number of column matches to return.
        """
        df_a, _ = _read_dataset(
            index, resource_id_a, seed, limit_to_n_columns=limit_to_n_columns
        )
        df_b, _ = _read_dataset(
            index, resource_id_b, seed, limit_to_n_columns=limit_to_n_columns
        )
        return {
            "matcher": matcher,
            "seed": seed,
            "limit_to_n_columns": limit_to_n_columns,
            **_valentine_compare(df_a, df_b, matcher, top_k),
        }

    @mcp.tool()
    def index_info() -> dict:
        """
        Describe the collection served by this benchmark: source, target
        language of questions/keywords/metadata, statement kind, number
        of datasets, and where the CSV files live.
        """
        info = {
            "source": source,
            "language": language,
            "kind": kind,
            "seed": seed,
            "limit_to_n_columns": limit_to_n_columns,
            "n_datasets": len(index),
            "datasets_path": str(index.datasets_path),
            "datasets_format": index.datasets_format,
        }
        if todo is not None:
            info["questions"] = todo.counts()
        return info

    def _require_todo() -> BenchmarkTodo:
        if todo is None:
            raise ValueError(
                "No benchmark questions available: the generate-statements "
                "output was not found when the server started."
            )
        return todo

    def list_questions(fetched: Optional[bool] = None) -> list[dict]:
        return _require_todo().list(fetched)

    list_questions.__doc__ = f"""
        The benchmark questions, each as just its id and question text
        (in {language}, the portal's language) — nothing else.

        Args:
            fetched: Optional filter: true for questions already handed
                out by get_question, false for questions still to fetch.
        """
    mcp.tool()(list_questions)

    def get_question(question_id: Optional[str] = None) -> dict:
        todo_list = _require_todo()
        if question_id is None:
            entry = todo_list.next_unfetched()
            if entry is None:
                raise ValueError("All questions have already been fetched.")
        else:
            entry = todo_list.get(question_id)
            if entry is None:
                raise ValueError(f"No question with id {question_id!r}")
        todo_list.mark_fetched(entry["id"])
        return {"id": entry["id"], "question": entry.get("question", "")}

    get_question.__doc__ = f"""
        The next benchmark question to answer — just its id and question
        text (in {language}, the portal's language), nothing else.

        Called without arguments (the normal flow) it hands out the first
        question not yet fetched and marks it fetched, so repeated
        no-argument calls walk through every question exactly once. Pass
        question_id to (re)fetch a specific one instead — e.g. one listed
        by list_questions.

        From here: extract keywords from the question yourself (in
        {language}) and call search_datasets to find and relate its
        candidate tables, then evaluate_tables to check your selection
        against the hidden ground truth.

        Args:
            question_id: Optional specific question id (as listed by
                list_questions); omit to get the first not-yet-fetched one.
        """
    mcp.tool()(get_question)

    def evaluate_tables(question_id: str, tables: dict[str, list[str]]) -> dict:
        entry = _require_todo().get(question_id)
        if entry is None:
            raise ValueError(f"No question with id {question_id!r}")
        reference_tables = (entry.get("reference") or {}).get("tables") or {}
        return evaluate_table_retrieval(reference_tables, tables)

    evaluate_tables.__doc__ = """
        Check whether the datasets found for a question — and the columns
        used from each — match what the reference answer actually used.

        Compares `tables` — resource_id -> the columns you'd use from it,
        the candidates and columns you settled on after search_datasets /
        compare_tables / get_table_info — against that question's hidden
        ground truth.

        Table-level: set precision/recall/F1 over resource_ids, plus which
        ones are "correct", "missing" (in the reference but not retrieved)
        or "extra" (retrieved but not referenced).

        Column-level: for every correctly retrieved table, precision/
        recall/F1 of its submitted columns against the reference's columns
        for that table (exact name match), plus a micro-averaged
        `columns_overall` across all correct tables. Get a column right
        even for a table you got wrong and it won't count here — only
        correctly retrieved tables are scored on columns.

        This is the final step of the retrieval benchmark loop for a
        question; call it again if you change your selection.

        Args:
            question_id: The question's id (as returned by get_question /
                list_questions).
            tables: Mapping from each retrieved dataset's resource_id to
                the list of column names from it you'd actually use to
                answer the question. Pass an empty list for a table if
                you're only asserting it's relevant, not which columns.
        """
    mcp.tool()(evaluate_tables)

    return mcp


def _load_index(cfg):
    """
    Prepare the reverse index for the configured backend, creating it
    from the normalized metadata when missing or stale.
    """
    if cfg.mcp_search.backend == "elasticsearch":
        import os

        from orqa.benchmark import es_index

        es_url = (
            os.environ.get("ELASTICSEARCH_URL", "").strip()
            or cfg.mcp_search.elasticsearch_url
        )
        es = es_index.connect(es_url)
        index, rebuilt = es_index.ESDatasetIndex.build_or_load(
            es,
            cfg.mcp_search.es_index_name,
            cfg.normalized_metadata_filepath,
            cfg.datasets_path,
            cfg.datasets_format,
            source=cfg.source,
        )
        location = f"Elasticsearch index {cfg.mcp_search.es_index_name!r} at {es_url}"
    else:
        index, rebuilt = DatasetIndex.build_or_load(
            cfg.normalized_metadata_filepath,
            cfg.mcp_search.index_filepath,
            cfg.datasets_path,
            cfg.datasets_format,
            source=cfg.source,
        )
        location = str(cfg.mcp_search.index_filepath)

    action = "Created" if rebuilt else "Reusing"
    _log(f"{action} {location} ({len(index)} datasets)")
    return index


def _load_todo(cfg) -> Optional[BenchmarkTodo]:
    """
    Load the questions produced by generate-statements (organized by id,
    in the workflow's target language) and sync them into the todo list
    under benchmark/<kind>/. Returns None when no questions exist yet.
    """
    queries_filepath = cfg.statement_generation.queries_path
    if not queries_filepath.exists():
        _log(
            f"No generated questions at {queries_filepath}; run the "
            "generate-statements step first. Serving dataset discovery only."
        )
        return None

    questions = load_questions(
        queries_filepath,
        language=cfg.statement_generation.target_language,
        kind=cfg.statement_generation.kind,
    )
    todo = BenchmarkTodo(cfg.benchmark_results_path)
    todo.sync(questions)
    counts = todo.counts()
    _log(
        f"Benchmark todo at {todo.todo_filepath}: {counts['total']} questions "
        f"({counts['fetched']} fetched, {counts['unfetched']} not yet fetched)"
    )
    return todo


def run_mcp_search(
    cfg,
    mode: Literal["stdio", "port", "build"] = "stdio",
) -> None:
    """
    Workflow step entry point (see main.py): prepare the reverse index
    and the questions todo list for the selected city, then serve them
    over MCP.
    """
    index = _load_index(cfg)

    if mode == "build":
        return

    server = create_server(
        index,
        cfg.mcp_search.host,
        cfg.mcp_search.port,
        language=cfg.statement_generation.target_language,
        kind=cfg.statement_generation.kind,
        seed=cfg.seed,
        limit_to_n_columns=cfg.candidates_discovery.limit_to_n_columns,
        todo=_load_todo(cfg),
    )
    if mode == "port":
        _log(
            "Serving MCP (streamable HTTP) on "
            f"http://{cfg.mcp_search.host}:{cfg.mcp_search.port}/mcp"
        )
        server.run(transport="streamable-http")
    else:
        _log("Serving MCP over stdio")
        server.run(transport="stdio")
