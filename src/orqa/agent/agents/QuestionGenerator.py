"""Stage-1 question generation — the agent that writes questions from the
verified retrieval anchor, before any plan exists.

The planner never writes a question: this agent does. It is handed the
vocabulary that a real search over the index already proved surfaces each
table — the anchor from ``orqa.agent.utility.keyword_suggestion`` — and builds
the question AROUND it, and states for every table why the question needs it
(``table_reasons``), which the question judge then tests. See
``orqa.agent.agents.QuestionStage`` for the loop that gates what it writes, and
``orqa.agent.utility.question_gate`` for the checks.

One batched call covers every slot in a round; a correction round re-sends
only the slots still failing, each with its previous question and the exact
reason it was rejected.
"""
from pathlib import Path
from typing import Any, Optional

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..prompting import QuestionGeneratorPrompt


class _QuestionDraftClient(LLMClientStructured):
    def __init__(self, config_path: Path):
        super().__init__(config_path, response_model="question_generator")


class QuestionGeneratorAgent:
    def __init__(self, config_path: Optional[Path] = None, client: Optional[Any] = None):
        """``client`` is injected for testing; otherwise one is built from
        ``config_path`` (loading config is cheap, nothing is requested until
        :meth:`generate` runs)."""
        if client is None:
            if config_path is None:
                raise ValueError("QuestionGeneratorAgent needs a config_path or a client")
            client = _QuestionDraftClient(config_path)
        self._client = client
        self._prompt = QuestionGeneratorPrompt()

    def generate(self, context: dict, slots: list[dict]) -> tuple[dict[int, dict], dict]:
        """Write one question per slot.

        ``context``: pre-rendered prompt blocks — ``{"languages": str,
        "time_context": str, "links_block": str, "tables_block": str}``
        (the caller owns rendering, so this agent stays free of the planner's
        internals).

        ``slots``: ``[{"slot": int, "tier": "easy"|"medium"|"hard",
        "anchors": {alias: [term, ...]}, "previous_question": str | None,
        "feedback": str | None}, ...]`` — everything but ``slot`` and
        ``tier`` is optional; the last two only appear on a correction
        round.

        Returns ``({slot: {"question", "question_keywords", "table_reasons"}},
        usage)`` — only the slots the model actually returned; a missing slot
        is the caller's to detect and retry. ``table_reasons`` is ``{alias:
        why the question needs that table}`` as the writer stated it (``{}``
        when it gave none): the question judge tests those claims, so it is
        handed them with the question.
        """
        prompt = self._prompt.update(
            n_slots=len(slots),
            languages=context.get("languages", "English"),
            time_context=context.get("time_context", ""),
            links_block=context.get("links_block") or "(single table — no relationships)",
            tables_block=context.get("tables_block", ""),
            slots_block=self._render_slots(slots),
        )
        result, usage = self._client.complete(prompt, root_key=None)
        return self._by_slot(result), usage

    @staticmethod
    def _render_slots(slots: list[dict]) -> str:
        blocks: list[str] = []
        for slot in slots:
            lines = [f"Slot {slot['slot']} — ambition: {slot.get('tier', 'medium')}"]
            anchors = slot.get("anchors") or {}
            for alias in sorted(anchors):
                if anchors[alias]:
                    lines.append(f"  {alias} ANCHOR TERMS: {', '.join(anchors[alias])}")
            if not any(anchors.values()):
                lines.append("  (no anchor terms for this slot — see the retrieval-anchor rules)")
            if slot.get("previous_question"):
                lines.append(f"  PREVIOUS ATTEMPT (rejected): {slot['previous_question']}")
            if slot.get("feedback"):
                lines.append(f"  WHY IT WAS REJECTED: {slot['feedback']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _table_reasons(raw: Any) -> dict[str, str]:
        """``{alias: reason}`` from what the model returned: the schema's list
        of ``{"alias", "reason"}`` entries, or a plain ``{alias: reason}``
        mapping. Entries missing an alias or a reason are dropped — the judge
        sees such a table as having no stated reason."""
        pairs = raw.items() if isinstance(raw, dict) else (
            (entry.get("alias"), entry.get("reason"))
            for entry in raw or []
            if isinstance(entry, dict)
        )
        reasons: dict[str, str] = {}
        for alias, reason in pairs:
            alias, reason = str(alias or "").strip(), str(reason or "").strip()
            if alias and reason:
                reasons[alias] = reason
        return reasons

    @staticmethod
    def _by_slot(result: Optional[dict]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for item in (result or {}).get("questions") or []:
            try:
                slot = int(item.get("slot"))
            except (TypeError, ValueError, AttributeError):
                continue
            out[slot] = {
                "question": (item.get("question") or "").strip(),
                "question_keywords": list(item.get("question_keywords") or []),
                "table_reasons": QuestionGeneratorAgent._table_reasons(item.get("table_reasons")),
            }
        return out
