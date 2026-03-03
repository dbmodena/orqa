import json
from pathlib import Path
from conf import OrQAConfig
from .agent.agent import GenerateResponseAgent
from .queries.query_execution import QueryExecutor
from .utils import save_json,load_json

class QueryResponsePipeline:
    """
    Orchestrates the full response-generation flow:
      1. Reads generated_queries.json
      2. For every (entry, query) pair executes the query via QueryExecutor
      3. Passes the question + query result to GenerateResponseAgent
      4. Collects and returns all responses as a list of dicts:

         [
           {
             "entry_key":        "0",
             "query_id":         1,
             "question":         "...",
             "query_type":       "sql",
             "difficulty":       "simple",
             "execution_result": <DataFrame | None>,
             "response":         "...",   # natural-language answer from the LLM
             "token_usage":      {...},
           },
           ...
         ]
    """

    def __init__(self, cfg: OrQAConfig):
        self.cfg = cfg
        self.executor = QueryExecutor(cfg.datasets_path,cfg.statement_generation.bad_tokens)
        self.kind = cfg.statement_generation.kind
        self.agent = GenerateResponseAgent(cfg.llm_config_path.joinpath("litellm.yaml"), kind=self.kind,max_tokens=cfg.statement_generation.max_response_tokens)

    def run(
        self,
        entry_keys: list[str] | None = None,
        max_result_rows: int = 50,
    ) -> list[dict]:
        queries_data = load_json(self.cfg.statement_generation.queries_path)
        responses: list[dict] = []

        for kind_key, kind_entries in queries_data.items():
            if self.kind is not None and kind_key != self.kind:
                continue
            for key, entry in kind_entries.items():
                if entry_keys is not None and key not in entry_keys:
                    continue
                if entry.get("status") != "success":
                    continue
                for query in entry.get("data", {}).get("queries", []):
                    record = self._process_query(entry, query, key, max_result_rows)
                    if record is not None:
                        query["response"] = record.get("response") 
                        responses.append(record)

        save_json(queries_data, self.cfg.statement_generation.queries_path) 
        return responses


    def _process_query(
        self,
        entry: dict,
        query: dict,
        entry_key: str,
        max_result_rows: int,
    ) -> dict | None:
        question: str = query.get("question", "")
        query_id: int = query.get("id", -1)
        query_type: str = query.get("query_type", "sql")
        difficulty: str = query.get("difficulty", "")

        # 1. Execute the query
        try:
            result_df = self.executor.execute(entry, query)
        except Exception as exc:
            print(f"[QueryResponsePipeline] ⚠️  Execution failed (entry={entry_key}, id={query_id}): {exc}")
            return {
                "entry_key": entry_key,
                "query_id": query_id,
                "question": question,
                "query_type": query_type,
                "difficulty": difficulty,
                "execution_result": None,
                "execution_error": str(exc),
                "response": None,
                "token_usage": None,
            }

        # 2. Serialise the result for the LLM
        if result_df is None or result_df.empty:
            data_str = "The query returned no results."
        else:
            truncated = result_df.head(max_result_rows)
            data_str = truncated.to_string(index=False)
            if len(result_df) > max_result_rows:
                data_str += f"\n\n[… truncated to {max_result_rows} of {len(result_df)} rows]"

        # 3. Generate the natural-language response
        try:

            agent_output = self.agent.generate_statements(question=question, data=data_str)
            response_text = agent_output.get("result") if agent_output else None
            token_usage = agent_output.get("token_usage") if agent_output else None
            #print(f"Question:{question}")
            #print(f"Response :{response_text}")
            #print(f"Data\n:{data_str}")
        except Exception as exc:
            print(f"[QueryResponsePipeline] ⚠️  Response generation failed (entry={entry_key}, id={query_id}): {exc}")
            response_text = None
            token_usage = None

        return {
            "entry_key": entry_key,
            "query_id": query_id,
            "question": question,
            "query_type": query_type,
            "difficulty": difficulty,
            "execution_result": result_df,
            "response": response_text,
            "token_usage": token_usage,
        }


def generate_response(cfg: OrQAConfig) -> list[dict]:
    """
    Entry point. Runs the full pipeline and returns a list of response dicts.
    """
    pipeline = QueryResponsePipeline(cfg)
    return pipeline.run()