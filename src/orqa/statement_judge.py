import asyncio
import gc
import json
from pathlib import Path

import pandas as pd
from conf import OrQAConfig
from tqdm import tqdm

from .agent.agent import GenerateResponseAgent
from .queries.query_execution import QueryExecutor


class QueryResponsePipeline:
    """
    Processes pending queries and streams result records.

    Persistence strategy
    --------------------
    - queries_path  (.json)  — the original query file; never rewritten here.
    - responses_path (.jsonl) — append-only log; one JSON line per completed
                                query.  Written with a single `write()` call so
                                no full-file reload is ever needed.

    On resume, already-completed query IDs are read from the .jsonl log at
    startup (O(lines), not O(file-size)), so nothing is re-processed.
    """

    def __init__(self, cfg: OrQAConfig):
        self.cfg = cfg
        self.executor = QueryExecutor(
            cfg.datasets_path, cfg.statement_generation.bad_tokens
        )
        self.agent = GenerateResponseAgent(
            cfg.llm_config_path.joinpath("litellm.yaml"),
            max_tokens=cfg.statement_generation.max_response_tokens,
        )

        queries_path: Path        = cfg.statement_generation.queries_path
        self._responses_path: Path = queries_path.with_suffix(".responses.jsonl")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — batch
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, max_result_rows: int = 10):
        """Iterate over all pending queries, yield one record per query."""
        queries_path = self.cfg.statement_generation.queries_path
        done_ids     = self._load_done_ids()

        with open(queries_path, encoding="utf-8") as f:
            queries_data = json.load(f)

        for model_key, model_entries in queries_data.items():
            for kind_key, kind_entries in model_entries.items():
                for key, entry in kind_entries.items():
                    if kind_key != "PANDAS":
                        continue
                    if entry.get("status") != "success":
                        continue

                    queries = entry.get("data", {}).get("queries", [])
                    pending = [
                        q for q in queries
                        if self._response_key(key, q.get("id")) not in done_ids
                    ]

                    for query in tqdm(pending, desc=f"{key}", unit="query", leave=False):
                        if query.get("response") is not None:
                            continue
                        record = self._process_query(entry, query, key, kind_key, max_result_rows)
                        if record is not None:
                            self._append_response(record)
                            done_ids.add(self._response_key(key, record["query_id"]))
                            yield record

                    entry["data"] = {"__evicted__": True}
                    gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — single query (async, used by web_app.py)
    # ──────────────────────────────────────────────────────────────────────────

    async def run_single_async(
        self,
        entry: dict,
        query: dict,
        entry_key: str,
        language,
        max_result_rows: int = 10,
        generate_nl: bool = True,
    ) -> dict:
        """
        Execute a single PANDAS query and optionally generate a natural-language
        response.  Runs all blocking work in a thread executor so the FastAPI
        event loop is never blocked.

        Args:
            generate_nl: if True (default), calls the LLM to produce a
                         natural-language answer.  Set False to get only the
                         DataFrame result without an LLM call.

        Returns a dict with:
            entry_key       : str
            query_id        : int
            question        : str
            difficulty      : str
            query_code      : str             — pandas code from the query dict
            language        : str             — always "PANDAS" for now
            response        : str | None      — NL answer (None if generate_nl=False)
            df_columns      : list[str]       — column headers ([] if no result)
            df_rows         : list[list]      — row values     ([] if no result)
            df_total_rows   : int             — total rows before truncation
            execution_error : str | None      — set if execution failed
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_single_sync, entry, query, entry_key,
            language, max_result_rows, generate_nl,
        )
        # Persist only when we have something worth saving
        if generate_nl and (result.get("response") or result.get("execution_error")):
            await loop.run_in_executor(None, self._persist_single, result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_single_sync(
        self,
        entry: dict,
        query: dict,
        entry_key: str,
        language: str = "PANDAS",
        max_result_rows: int = 10,
        generate_nl: bool = True,
    ) -> dict:
        """
        Blocking implementation of run_single_async — intended to be called
        from run_in_executor only.  Unlike _process_query it also serialises
        the DataFrame into df_columns / df_rows / df_total_rows so the web
        layer can render it as a table.
        """
        question   = query.get("question", "")
        query_id   = query.get("id", -1)
        difficulty = query.get("difficulty", "")

        df_columns: list[str]   = []
        df_rows:    list[list]  = []
        df_total    = 0
        response    = None
        exec_error  = None
        result_df   = None

        try:
            # ── 1. Execute ────────────────────────────────────────────────
            try:
                result_df = self.executor.execute(entry, query, language)
            except Exception as exc:
                return {
                    "entry_key":       entry_key,
                    "query_id":        query_id,
                    "question":        question,
                    "difficulty":      difficulty,
                    "language":        language,
                    "query_code":      query.get("query") or query.get("code") or query.get("pandas_query", ""),
                    "query_tables":    query.get("tables", []),
                    "response":        None,
                    "df_columns":      [],
                    "df_rows":         [],
                    "df_total_rows":   0,
                    "execution_error": str(exc),
                }

            # ── 2. Serialise DataFrame ───────────────────────────────────
            # NOTE: columns_involved in query["tables"] are the *input* columns
            # used by the query (for joins, filters, etc.) — NOT the output columns.
            # The result DataFrame already contains exactly what the query selected,
            # so we show it as-is without any column filtering.
            if result_df is not None and not result_df.empty:
                df_total   = len(result_df)
                truncated  = result_df.head(max_result_rows)
                df_columns = list(truncated.columns)

                def _to_native(v):
                    """Convert numpy/pandas scalars to plain Python types for JSON."""
                    try:
                        if pd.isna(v):
                            return None
                    except (TypeError, ValueError):
                        pass
                    # numpy integers → int, numpy floats → float
                    if hasattr(v, 'item'):
                        return v.item()
                    return v

                df_rows    = [
                    [_to_native(v) for v in row]
                    for row in truncated.itertuples(index=False, name=None)
                ]
                data_str = truncated.to_string(index=False)
                if df_total > max_result_rows:
                    data_str += (
                        f"\n\n[… truncated to {max_result_rows} "
                        f"of {df_total} rows]"
                    )
            else:
                data_str = "The query returned no results."

            # ── 4. Generate NL response (optional) ───────────────────────
            if generate_nl:
                try:
                    agent_out = self.agent.generate_statements(
                        question=question, data=data_str
                    )
                    response = agent_out.get("result") if agent_out else None
                except Exception as exc:
                    print(
                        f"[QueryResponsePipeline] ⚠️  Response generation failed "
                        f"(entry={entry_key}, id={query_id}): {exc}"
                    )

        finally:
            del result_df
            gc.collect()

        return {
            "entry_key":       entry_key,
            "query_id":        query_id,
            "question":        question,
            "difficulty":      difficulty,
            "language":        language,
            "query_code":      query.get("query") or query.get("code") or query.get("pandas_query", ""),
            "query_tables":    query.get("tables", []),
            "response":        response,
            "df_columns":      df_columns,
            "df_rows":         df_rows,
            "df_total_rows":   df_total,
            "execution_error": exec_error,
        }

    def _persist_single(self, result: dict) -> None:
        """Append a single result to the .jsonl log and merge back into JSON."""
        self._append_response({
            "entry_key":       result["entry_key"],
            "query_id":        result["query_id"],
            "question":        result["question"],
            "difficulty":      result["difficulty"],
            "response":        result["response"],
            "produce_result":  "YES" if result["response"] else "NO",
            "execution_error": result.get("execution_error"),
            "token_usage":     None,
        })
        self.merge_responses_back()

    @staticmethod
    def _response_key(entry_key: str, query_id: int) -> str:
        return f"{entry_key}:{query_id}"

    def _load_done_ids(self) -> set[str]:
        done: set[str] = set()
        if not self._responses_path.exists():
            return done
        with open(self._responses_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(self._response_key(rec["entry_key"], rec["query_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        return done

    def _append_response(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self._responses_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _process_query(
        self,
        entry: dict,
        query: dict,
        entry_key: str,
        query_kind: str,
        max_result_rows: int,
    ) -> dict | None:
        """Used by the batch run() method. Does not return DataFrame columns/rows."""
        question       = query.get("question", "")
        query_id       = query.get("id", -1)
        difficulty     = query.get("difficulty", "")
        produce_result = "NO"
        result_df      = None

        try:
            try:
                result_df = self.executor.execute(entry, query, query_kind)
            except Exception as exc:
                print(
                    f"[QueryResponsePipeline] ⚠️  Execution failed "
                    f"(entry={entry_key}, id={query_id}): {exc}"
                )
                return {
                    "entry_key":       entry_key,
                    "query_id":        query_id,
                    "question":        question,
                    "difficulty":      difficulty,
                    "execution_error": str(exc),
                    "response":        None,
                    "produce_result":  produce_result,
                    "token_usage":     None,
                }

            if result_df is None or result_df.empty:
                data_str = "The query returned no results."
            else:
                truncated = result_df.head(max_result_rows)
                data_str  = truncated.to_string(index=False)
                if len(result_df) > max_result_rows:
                    data_str += (
                        f"\n\n[… truncated to {max_result_rows} "
                        f"of {len(result_df)} rows]"
                    )

            try:
                agent_output   = self.agent.generate_statements(question=question, data=data_str)
                response_text  = agent_output.get("result")  if agent_output else None
                token_usage    = agent_output.get("token_usage") if agent_output else None
                produce_result = "YES"
            except Exception as exc:
                print(
                    f"[QueryResponsePipeline] ⚠️  Response generation failed "
                    f"(entry={entry_key}, id={query_id}): {exc}"
                )
                response_text = None
                token_usage   = None

            return {
                "entry_key":     entry_key,
                "query_id":      query_id,
                "question":      question,
                "query_type":    query_kind,
                "difficulty":    difficulty,
                "response":      response_text,
                "produce_result":produce_result,
                "token_usage":   token_usage,
            }

        finally:
            del result_df
            gc.collect()

    def merge_responses_back(self) -> None:
        queries_path = self.cfg.statement_generation.queries_path

        responses: dict[str, dict] = {}
        if self._responses_path.exists():
            with open(self._responses_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = self._response_key(rec["entry_key"], rec["query_id"])
                        responses[key] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass

        with open(queries_path, encoding="utf-8") as f:
            queries_data = json.load(f)

        for model_key, model_entries in queries_data.items():
            for kind_key, kind_entries in model_entries.items():
                for entry_key, entry in kind_entries.items():
                    for query in entry.get("data", {}).get("queries", []):
                        rkey = self._response_key(entry_key, query.get("id"))
                        if rkey in responses:
                            rec = responses[rkey]
                            query["response"]       = rec.get("response")
                            query["produce_result"] = rec.get("produce_result")
                            query["token_usage"]    = rec.get("token_usage")
                            if "execution_error" in rec:
                                query["execution_error"] = rec["execution_error"]

        tmp_path = queries_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(queries_data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(queries_path)


def generate_response(cfg: OrQAConfig):
    pipeline = QueryResponsePipeline(cfg)
    records  = list(pipeline.run())
    pipeline.merge_responses_back()
    return records