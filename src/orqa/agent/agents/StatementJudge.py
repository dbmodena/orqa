from pathlib import Path
from typing import Any

from ..llm_client.LLMClientStructured import LLMClientStructured


class LLMStatementJudge(LLMClientStructured):
    """
    Single primary-model code judge (the fallback when the LLM yaml configures
    no ``judge_profiles.code`` panel — see ``JudgePanel``).

    Uses a fixed 2-block message structure (system + query evaluation) that is
    rebuilt from scratch on each call — no retained history between calls.
    """

    # A verdict is a short output: native JSON-schema decoding is safe there
    # (the judge panels run it too).
    native_schema = True

    def __init__(self, config_path: Path):
        super().__init__(config_path, "statement_judge")

    def complete(self, prompt: str, data: Any = None, **kwargs) -> Any:
        """
        Make a structured completion request with fixed 2-block messages.

        ``prompt`` carries ONLY the static judge instructions; ``data`` carries
        the per-query evaluation payload. The instructions plus the (equally
        static) output-schema constraint form the system message, so the system
        message is byte-identical across every judge call in a run — the
        provider's prompt cache serves that whole prefix, and only the short
        user message (the payload) is uncached.

        The call is the shared budget-aware structured loop
        (``LLMClientStructured.complete``): on a parse error the user payload
        is rebuilt with the error feedback (the system message never changes),
        a verdict cut by the output budget is retried with a bigger one.

        Returns ``(result_dict, usage_total)`` on success; ``({}, usage_total)`` on
        exhausted retries.
        """
        # Never empty — providers reject zero-token messages (OCI 400:
        # "message must be at least 1 token long").
        query_payload = (
            f"Queries:\n{data}\n\n"
            "Evaluate the queries above following the instructions and "
            "return only the JSON verdict."
        )
        return LLMClientStructured.complete(
            self, query_payload, root_key="queries", system_prompt=prompt, **kwargs
        )
