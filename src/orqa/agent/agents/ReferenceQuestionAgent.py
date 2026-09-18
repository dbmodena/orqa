"""Per-table reference-question decomposition for an approved multi-table
query — see ``StatementOrchestrator._generate_reference_questions_for_query``
(agent.py) for the caller and ``orqa.agent.utility.structured_outputs.
ReferenceQuestionSet`` for why this call's schema is deliberately narrow.

One question per table in a SINGLE batched call, coupled to that table's own
plan ``reason`` (not an independently invented question — see
``prompting.models.Table.reason``), so it reads as the natural slice of the
main question that table alone answers. A rejected table (failed the
deterministic top-1 retrieval gate) gets a SMALL follow-up call scoped to
just that table, not a full regeneration of the group.
"""
from pathlib import Path
from typing import Optional

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..prompting import ReferenceQuestionsPrompt


class _ReferenceQuestionsClient(LLMClientStructured):
    def __init__(self, config_path: Path):
        super().__init__(config_path, response_model="reference_questions")


class ReferenceQuestionAgent:
    def __init__(self, config_path: Path):
        self._client = _ReferenceQuestionsClient(config_path)
        self._prompt = ReferenceQuestionsPrompt()

    def generate(self, main_question: str, tables: list[dict]) -> tuple[dict, dict]:
        """``tables``: ``[{"alias", "reason", "facets": [str, ...],
        "anchor": [str, ...], "feedback": str | None}, ...]`` — ``facets``/
        ``anchor``/``feedback`` are all optional per entry; ``feedback``
        (only set on a retry) is the deterministic gate's own miss
        diagnosis for that table's PREVIOUS attempt.

        Returns ``({alias: {"question", "question_keywords"}}, usage)`` —
        only the aliases the model actually returned are present; a missing
        alias is the caller's to detect and (optionally) retry.
        """
        prompt = self._prompt.update(
            main_question=main_question,
            tables_block=self._render_tables(tables),
        )
        result, usage = self._client.complete(prompt, root_key=None)
        return self._by_alias(result), usage

    @staticmethod
    def _render_tables(tables: list[dict]) -> str:
        lines: list[str] = []
        for t in tables:
            lines.append(f"- {t['alias']}: {t['reason']}")
            if t.get("facets"):
                lines.append(f"  must state: {'; '.join(t['facets'])}")
            if t.get("anchor"):
                lines.append(f"  retrieval anchor: {', '.join(t['anchor'])}")
            if t.get("feedback"):
                lines.append(f"  previous attempt rejected: {t['feedback']}")
        return "\n".join(lines)

    @staticmethod
    def _by_alias(result: Optional[dict]) -> dict:
        out: dict = {}
        for item in (result or {}).get("questions") or []:
            alias = item.get("table")
            if not alias:
                continue
            out[alias] = {
                "question": item.get("question", ""),
                "question_keywords": item.get("question_keywords") or [],
            }
        return out
