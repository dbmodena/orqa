import gc
import json
from pathlib import Path

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

        queries_path: Path = cfg.statement_generation.queries_path
        self._responses_path: Path = queries_path.with_suffix(".responses.jsonl")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_result_rows: int = 10):
        queries_path = self.cfg.statement_generation.queries_path
        done_ids = self._load_done_ids()

        with open(queries_path, encoding="utf-8") as f:
            queries_data = json.load(f)

        for model_key, model_entries in queries_data.items():
            for kind_key, kind_entries in model_entries.items():
                for key, entry in kind_entries.items():
                    query_kind = kind_key
                    if query_kind != "PANDAS":
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
                        record = self._process_query(entry, query, key, query_kind,max_result_rows)
                        if record is not None:
                            self._append_response(record)
                            done_ids.add(self._response_key(key, record["query_id"]))
                            yield record

                    # Evict heavy payload; the JSON file is never written back
                    entry["data"] = {"__evicted__": True}
                    gc.collect()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _response_key(entry_key: str, query_id: int) -> str:
        return f"{entry_key}:{query_id}"

    def _load_done_ids(self) -> set[str]:
        """
        Read the response log and return the set of already-completed
        (entry_key, query_id) pairs encoded as "entry_key:query_id".
        Runs in O(lines) with negligible RAM — only the key strings are kept.
        """
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
                    pass  # corrupt line — skip, do not crash
        return done

    def _append_response(self, record: dict) -> None:
        """
        Append one record to the .jsonl log.
        Single write() call — no read, no full-file rewrite, ~0 RAM overhead.
        """
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self._responses_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _process_query(
        self,
        entry: dict,
        query: dict,
        entry_key: str,
        query_kind:str,
        max_result_rows: int,
    ) -> dict | None:
        question: str = query.get("question", "")
        query_id: int = query.get("id", -1)
        query_type: str = query_kind
        difficulty: str = query.get("difficulty", "")
        produce_result = "NO"
        result_df = None

        try:
            # ── 1. Execute ────────────────────────────────────────────────
            try:
                result_df = self.executor.execute(entry, query,query_type)
            except Exception as exc:
                print(
                    f"[QueryResponsePipeline] ⚠️  Execution failed "
                    f"(entry={entry_key}, id={query_id}): {exc}"
                )
                return {
                    "entry_key": entry_key,
                    "query_id": query_id,
                    "question": question,
                    #"query_type": query_type,
                    "difficulty": difficulty,
                    "execution_error": str(exc),
                    "response": None,
                    "produce_result": produce_result,
                    "token_usage": None,
                }

            # ── 2. Serialise — keep only the string, drop the DataFrame ──
            if result_df is None or result_df.empty:
                data_str = "The query returned no results."
                print(data_str)
            else:
                truncated = result_df.head(max_result_rows)
                data_str = truncated.to_string(index=False)
                if len(result_df) > max_result_rows:
                    data_str += (
                        f"\n\n[… truncated to {max_result_rows} "
                        f"of {len(result_df)} rows]"
                    )

            # ── 3. Generate response ──────────────────────────────────────
            try:
                agent_output = self.agent.generate_statements(
                    question=question, data=data_str
                )
                response_text = agent_output.get("result") if agent_output else None
                token_usage = agent_output.get("token_usage") if agent_output else None
                produce_result = "YES"
            except Exception as exc:
                print(
                    f"[QueryResponsePipeline] ⚠️  Response generation failed "
                    f"(entry={entry_key}, id={query_id}): {exc}"
                )
                response_text = None
                token_usage = None

            return {
                "entry_key": entry_key,
                "query_id": query_id,
                "question": question,
                "query_type": query_type,
                "difficulty": difficulty,
                "response": response_text,
                "produce_result": produce_result,
                "token_usage": token_usage,
            }

        finally:
            del result_df
            gc.collect()

    def merge_responses_back(self) -> None:
        """
        Read the .jsonl log and write the responses back into the original
        queries .json file, keyed by entry_key + query_id.
        """
        queries_path = self.cfg.statement_generation.queries_path

        # 1. Load all responses from .jsonl into a lookup dict
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

        # 2. Load the original queries JSON
        with open(queries_path, encoding="utf-8") as f:
            queries_data = json.load(f)

        # 3. Patch each matching query in-place
        for model_key, model_entries in queries_data.items():
            for kind_key, kind_entries in model_entries.items():
                for entry_key, entry in kind_entries.items():
                    queries = entry.get("data", {}).get("queries", [])
                    for query in queries:
                        rkey = self._response_key(entry_key, query.get("id"))
                        if rkey in responses:
                            rec = responses[rkey]
                            query["response"] = rec.get("response")
                            query["produce_result"] = rec.get("produce_result")
                            query["token_usage"] = rec.get("token_usage")
                            if "execution_error" in rec:
                                query["execution_error"] = rec["execution_error"]

        # 4. Write back atomically (temp file → rename)
        tmp_path = queries_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(queries_data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(queries_path)  # atomic on POSIX

def generate_response(cfg: OrQAConfig):
    pipeline = QueryResponsePipeline(cfg)
    records = list(pipeline.run())
    pipeline.merge_responses_back()
    return records