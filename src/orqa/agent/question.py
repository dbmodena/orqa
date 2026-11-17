"""Stage 1 of the split pipeline: write, gate and judge QUESTIONS — no plan.

The old flow asked the planner for a question and its steps in one go, then
judged both together; a question that missed retrieval or topic linkage sent
the WHOLE plan back for regeneration, and rewording a question under steps
that already existed could break the plan it belonged to. Here a question is
settled first and only then handed to the planner (see ``QueryPlanner.
plan_batch``'s ``questions``), frozen.

Per slot, per round (cheapest check first, so a failure never pays for the
checks after it):

1. ``QuestionGeneratorAgent`` writes the question from the verified retrieval
   anchor (one batched call across every slot still open).
2. ``gate.gate_question`` — leak, then the retriever panel's
   majority vote (lexical gate, semantic, hybrid RRF).
3. The question judge panel — readability, topic linkage, grounding, table
   necessity, difficulty (does the question demand about as much analysis as
   its slot's tier) — only for a question the deterministic gate passed.

A rejected slot is re-sent, with its previous question and the exact reason,
up to ``max_corrections`` more times; a slot still failing is dropped. A
question that failed a gate never ships as a "least bad" fallback.

``question_keywords`` of an approved question are the anchor terms actually
present in its text (deterministic), never a list the model authored.

``rewrite`` takes questions the planner could not turn into an approved plan,
each with the reason, and runs the same gated loop on them — so a question
that cannot be planned is rewritten once more rather than dropped (see
``StatementOrchestrator._replan_unplannable``).
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from orqa.retrieval.text import tokenize
from orqa.utils.generation_trace import trace_event
from orqa.utils.table_analyzer import render_identity_line, render_table_facts, table_identities
from orqa.retrieval.gate import RetrievalContract, covered_terms, describe_votes, gate_question
from orqa.agent.llm_client.LLMClientStructured import LLMClientStructured
from orqa.agent.prompting import QuestionGeneratorPrompt


# ── Question generator ────────────────────────────────────────────────────────
# Stage-1 question generation — the agent that writes questions from the
# verified retrieval anchor, before any plan exists.
#
# The planner never writes a question: this agent does. It is handed the
# vocabulary that a real search over the index already proved surfaces each
# table — the anchor from ``orqa.retrieval.anchors`` — and builds
# the question AROUND it, and states for every table why the question needs it
# (``table_reasons``), which the question judge then tests. See
# ``orqa.agent.question`` for the loop that gates what it writes, and
# ``orqa.retrieval.gate`` for the checks.
#
# One batched call covers every slot in a round; a correction round re-sends
# only the slots still failing, each with its previous question and the exact
# reason it was rejected.

class _QuestionDraftClient(LLMClientStructured):
    # Questions are a short output: native JSON-schema decoding is safe here
    # (it looped only on long outputs — plans, code — in the experiments), and
    # a draft set with one unreadable slot keeps its valid slots: the stage
    # re-asks the missing ones on the next round.
    native_schema = True
    salvage_items = True

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
            codes = slot.get("code_terms") or {}
            for alias in sorted(anchors):
                code_like = [t for t in anchors[alias] if t in set(codes.get(alias) or [])]
                plain = [t for t in anchors[alias] if t not in code_like]
                if plain:
                    lines.append(f"  {alias} ANCHOR TERMS: {', '.join(plain)}")
                if code_like:
                    lines.append(
                        f"  {alias} CODE-LIKE TERMS (the search needs them; never paste them — say what "
                        f"they mean in words, see TABLE IDENTITY): {', '.join(code_like)}"
                    )
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


logger = logging.getLogger(__name__)

# 1 first attempt + this many correction rounds per slot. The workflow yaml's
# tasks.query_generation.max_question_corrections overrides it. 5 (was 3):
# on the 6-table group of experiments/structured_stability the two approved
# slots passed the question judge on their 5th and 6th attempt; 4 rounds
# ended with none, twice.
MAX_QUESTION_CORRECTIONS = 5

# How many times a question the planner could not turn into an approved plan
# is rewritten and planned again before it is dropped.
MAX_QUESTION_REPLANS = 1

TIERS = ("easy", "medium", "hard")


def tiers_for(n: int) -> list[str]:
    """Target difficulty per slot: easy -> medium -> hard, cycling, so three
    slots give one of each. The question judge holds each question to its
    slot's tier, and that tier is the SHIPPED difficulty label — it is pinned
    onto the plan, and nothing on the plan side estimates or checks it."""
    return [TIERS[i % len(TIERS)] for i in range(max(0, n))]


@dataclass
class QuestionStageResult:
    approved: list[dict] = field(default_factory=list)
    feedback: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: _zero_usage())
    # None when at least one question was approved; otherwise the run status:
    # "unretrievable_group" if a slot's final failure was retrieval-driven
    # (a retriever-panel miss), else "no_valid_question".
    abort_status: Optional[str] = None


@dataclass
class _Slot:
    slot: int
    tier: str
    # The anchor this slot is written against.
    anchors: dict[str, list[str]] = field(default_factory=dict)
    # Anchor terms a person would never type as written (a file-name token,
    # a column code): the search needs them, the writer must say what they
    # mean instead of pasting them (anchors.is_speakable_term).
    code_terms: dict[str, list[str]] = field(default_factory=dict)
    previous_question: Optional[str] = None
    feedback: Optional[str] = None
    last_failure: Optional[str] = None
    attempts: list[dict] = field(default_factory=list)
    approved: Optional[dict] = None


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_usage(total: dict, part: Optional[dict]) -> None:
    for key in total:
        total[key] += (part or {}).get(key, 0)


def build_question_tables_context(
    aliases: Any,
    analyses: list,
    table_metadata: Optional[dict],
    stats: list,
    dfs: Optional[list],
    identities: Optional[list] = None,
) -> dict:
    """The compact table context the question judge is shown (and the stage
    records): alias NAMES only (the values of ``aliases`` are the portal's
    ids — no information for a judge), the analyses, the portal metadata, the
    columns with dtypes, the scope facts, and each table's identity line (the
    dated span of its rows, its file name and publisher, the words only its
    title/file name has — see ``table_analyzer.table_identity``)."""
    alias_list = list(aliases)
    metadata = table_metadata or {}
    frames = list(dfs or [])
    if identities is None:
        identities = table_identities(alias_list, frames, metadata)
    return {
        "aliases": alias_list,
        "analyses": list(analyses),
        "metadata": metadata,
        "columns": {
            table.alias: [f"{c.column} ({c.dtype})" for c in table.columns]
            for table in stats
        },
        "facts": {
            alias: render_table_facts(df).splitlines()
            for alias, df in zip(alias_list, frames)
        },
        "identity": {
            identity["alias"]: render_identity_line(identity, len(identities))
            for identity in identities
        },
    }


def build_question_judge_payload(
    question: str,
    tier: str,
    tables_context: dict,
    anchors: dict[str, list[str]],
    table_reasons: Optional[dict[str, str]] = None,
) -> str:
    """The user message for one question judge: the question, its slot's
    target difficulty, the anchor it was built from, the writer's stated reason
    for each table (and which tables it gave none for), and the compact table
    context (analyses, portal metadata, columns, scope facts)."""
    reasons = dict(table_reasons or {})
    payload = {
        "question": question,
        "slot_ambition": tier,
        "retrieval_anchor": anchors,
        "table_reasons": reasons,
        "tables_without_a_reason": [
            alias for alias in (tables_context.get("aliases") or []) if alias not in reasons
        ],
        "tables": tables_context,
    }
    return (
        "Question to evaluate:\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        + "\n\nEvaluate the question above following the instructions and "
        "return only the JSON verdict."
    )


def _rank_summary(retrieval: Optional[dict]) -> dict:
    """Ranks per retriever — the numbers behind a retrieval verdict, without
    the full vote payload."""
    votes = (retrieval or {}).get("votes") or {}
    return {n: r["ranks"] for n, r in (votes.get("per_retriever") or {}).items()}


class QuestionStage:
    def __init__(
        self,
        generator: Any,
        judge: Optional[Any] = None,
        judge_instructions: str = "",
        panel: Optional[Any] = None,
        *,
        max_corrections: int = MAX_QUESTION_CORRECTIONS,
        per_table_top_k: Optional[int] = None,
        leak_check: Optional[Callable[[str], Optional[str]]] = None,
        judge_kwargs: Optional[dict] = None,
        log: Optional[Any] = None,
    ):
        """
        Args:
            generator: ``QuestionGeneratorAgent``-shaped: ``generate(context,
                slots) -> ({slot: {...}}, usage)``.
            judge: ``JudgePanel``-shaped (``is_configured``, ``evaluate``) or
                ``None`` to skip the LLM judge (deterministic gate only).
            panel: the ``RetrieverPanel`` (``None`` skips retrieval, as the
                gate itself does).
            per_table_top_k: retrieval window per table for a multi-table
                group, where the lexical retriever becomes one keyword check
                per table (see ``check_question_retrievability``); ``None``
                votes the whole group at once across the retriever panel.
            log: anything with ``info``/``warning`` (a ``PipelineLogger``).
        """
        self.generator = generator
        self.judge = judge
        self.judge_instructions = judge_instructions
        self.panel = panel
        self.max_corrections = max(0, int(max_corrections))
        self.per_table_top_k = per_table_top_k
        self.leak_check = leak_check
        self.judge_kwargs = dict(judge_kwargs or {})
        self._log = log

    # ------------------------------------------------------------------

    def _info(self, message: str) -> None:
        (self._log.info if self._log is not None else logger.info)(message)

    def _warn(self, message: str) -> None:
        (self._log.warning if self._log is not None else logger.warning)(message)

    def _report_votes(self, votes: dict, title: str) -> None:
        """Show the retriever panel's vote in the pipeline log (a plain
        ``logging`` logger has no such view and simply skips it)."""
        report = getattr(self._log, "retrieval_votes", None)
        if callable(report):
            report(votes, title)

    def run(
        self,
        *,
        tiers: list[str],
        context: dict,
        tables_context: dict,
        anchors: dict[str, list[str]],
        contract: Optional[RetrievalContract],
        code_terms: Optional[dict[str, list[str]]] = None,
    ) -> QuestionStageResult:
        """Produce up to ``len(tiers)`` approved questions.

        ``anchors``: ``{alias: [...]}`` — the verified vocabulary the
        questions are built from (per table for a multi-table group).
        ``code_terms``: ``{alias: [...]}`` — the subset of each anchor a
        person would never type as written (the writer is told to say what
        they mean, never to paste them). ``contract`` gates retrieval
        (``None`` skips it). ``context`` / ``tables_context``: see
        ``QuestionGeneratorAgent.generate`` and
        ``build_question_judge_payload``.
        """
        codes = {a: list(t) for a, t in (code_terms or {}).items() if t}
        slots = [
            _Slot(slot=i, tier=tier, anchors={a: list(t) for a, t in anchors.items()},
                  code_terms={a: list(t) for a, t in codes.items() if a in anchors})
            for i, tier in enumerate(tiers, start=1)
        ]
        return self._loop(slots, context, tables_context, contract, phase="initial")

    def rewrite(
        self,
        requests: list[dict],
        *,
        context: dict,
        tables_context: dict,
        contract: Optional[RetrievalContract],
    ) -> QuestionStageResult:
        """Write a new question for each request — same gated loop as
        :meth:`run`, started from a rejected question and the reason.

        ``requests``: ``[{"slot": int, "tier": str, "anchors": {alias:
        [...]}, "previous_question": str, "feedback": str}, ...]`` — the
        slot's tier and anchor carry over from the run that produced the
        question. ``feedback`` is why it could not be used
        downstream (e.g. the planner could not plan it).
        """
        slots = [
            _Slot(
                slot=r["slot"],
                tier=r.get("tier", "medium"),
                anchors={a: list(t) for a, t in (r.get("anchors") or {}).items()},
                code_terms={a: list(t) for a, t in (r.get("code_terms") or {}).items() if t},
                previous_question=r.get("previous_question"),
                feedback=r.get("feedback"),
            )
            for r in requests
        ]
        return self._loop(slots, context, tables_context, contract, phase="rewrite")

    def _loop(
        self,
        slots: list[_Slot],
        context: dict,
        tables_context: dict,
        contract: Optional[RetrievalContract],
        phase: str = "initial",
    ) -> QuestionStageResult:
        result = QuestionStageResult()
        trace_event(
            "question_stage_start",
            phase=phase,
            max_rounds=1 + self.max_corrections,
            contract=contract.to_dict() if hasattr(contract, "to_dict") else contract,
            slots=[
                {
                    "slot": s.slot,
                    "tier": s.tier,
                    "anchors": s.anchors,
                    "code_terms": s.code_terms,
                    "previous_question": s.previous_question,
                    "feedback": s.feedback,
                }
                for s in slots
            ],
            # What the writer and the judges are shown — identical for a
            # rewrite of the same group, so recorded with the first run only.
            writer_context=context if phase == "initial" else None,
            judge_context=tables_context if phase == "initial" else None,
        )

        for round_idx in range(1 + self.max_corrections):
            pending = [s for s in slots if s.approved is None]
            if not pending:
                break
            self._info(
                f"Question stage round {round_idx + 1}/{1 + self.max_corrections}: "
                f"{len(pending)} open slot(s)."
            )
            drafts, gen_usage = self.generator.generate(
                context,
                [
                    {
                        "slot": s.slot,
                        "tier": s.tier,
                        "anchors": s.anchors,
                        "code_terms": s.code_terms,
                        "previous_question": s.previous_question,
                        "feedback": s.feedback,
                    }
                    for s in pending
                ],
            )
            _add_usage(result.usage, gen_usage)

            for s in pending:
                self._settle(
                    s, drafts.get(s.slot), round_idx + 1,
                    contract, tables_context, result.usage, phase=phase,
                )

        for s in slots:
            result.feedback.append(
                {
                    "slot": s.slot,
                    "tier": s.tier,
                    "approved": s.approved is not None,
                    "question": (s.approved or {}).get("question") or s.previous_question or "",
                    "anchors": s.anchors,
                    "attempts": s.attempts,
                }
            )
        result.approved = [s.approved for s in slots if s.approved is not None]
        if not result.approved:
            retrieval_driven = any(s.last_failure == "retrieval" for s in slots)
            result.abort_status = "unretrievable_group" if retrieval_driven else "no_valid_question"
            self._warn(
                "Question stage produced no approved question "
                f"(status {result.abort_status})."
            )
        trace_event(
            "question_stage_end",
            phase=phase,
            abort_status=result.abort_status,
            usage=result.usage,
            approved=[
                {k: q.get(k) for k in ("slot", "tier", "question", "question_keywords")}
                for q in result.approved
            ],
            slots=[
                {
                    "slot": s.slot,
                    "tier": s.tier,
                    "approved": s.approved is not None,
                    "attempts": len(s.attempts),
                    "last_failure": s.last_failure,
                }
                for s in slots
            ],
        )
        return result

    # ------------------------------------------------------------------

    def _settle(
        self,
        s: _Slot,
        draft: Optional[dict],
        attempt_no: int,
        contract: Optional[RetrievalContract],
        tables_context: dict,
        usage: dict,
        phase: str = "initial",
    ) -> None:
        """Gate (then judge) one slot's draft; approve it or record why not —
        and record the attempt in the generation trace either way."""
        try:
            self._settle_draft(s, draft, attempt_no, contract, tables_context, usage)
        except Exception as exc:
            self._trace_attempt(s, attempt_no, phase, error=exc)
            raise
        self._trace_attempt(s, attempt_no, phase)

    @staticmethod
    def _trace_attempt(
        s: _Slot, attempt_no: int, phase: str, error: Optional[Exception] = None
    ) -> None:
        """One ``question_attempt`` event: the attempt as recorded in the
        slot's feedback (question, gate verdict and votes, judge panel), its
        outcome, and the feedback the writer gets for its next round."""
        attempt = s.attempts[-1] if s.attempts else {"attempt": attempt_no}
        if error is not None:
            outcome = "error"
        elif s.approved is not None:
            outcome = "approved"
        else:
            outcome = s.last_failure or "rejected"
        trace_event("question_attempt", **{
            "phase": phase,
            "slot": s.slot,
            "tier": s.tier,
            "outcome": outcome,
            **attempt,
            "feedback_to_writer": None if s.approved is not None else s.feedback,
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
        })

    def _settle_draft(
        self,
        s: _Slot,
        draft: Optional[dict],
        attempt_no: int,
        contract: Optional[RetrievalContract],
        tables_context: dict,
        usage: dict,
    ) -> None:
        anchors = s.anchors
        attempt: dict = {"attempt": attempt_no}
        s.attempts.append(attempt)

        question = (draft or {}).get("question") or ""
        if not question:
            attempt.update(question="", failed_at="empty")
            s.last_failure = "empty"
            s.feedback = "No question was returned for this slot."
            return

        table_reasons = dict((draft or {}).get("table_reasons") or {})
        attempt.update(question=question, table_reasons=table_reasons)

        gate = gate_question(
            question,
            anchors=anchors,
            contract=contract,
            panel=self.panel,
            per_table_top_k=self.per_table_top_k,
            leak_check=self.leak_check,
        )
        pop_usage = getattr(self.panel, "pop_usage", None)
        if callable(pop_usage):
            _add_usage(usage, pop_usage())
        attempt["gate"] = {
            "approved": gate.approved,
            "failed_at": gate.failed_at,
            "feedback": gate.feedback,
            "missing_tables": (gate.retrieval or {}).get("missing_tables", []),
            "ranks": _rank_summary(gate.retrieval),
        }
        votes = describe_votes((gate.retrieval or {}).get("votes"), contract)
        if votes:
            attempt["gate"]["votes"] = votes
            self._report_votes(votes, f"slot {s.slot} · attempt {attempt_no}")
        if not gate.approved:
            s.last_failure = gate.failed_at
            s.previous_question = question
            s.feedback = gate.feedback
            self._info(f"Slot {s.slot} attempt {attempt_no} failed the {gate.failed_at} gate.")
            return

        if self.judge is not None and getattr(self.judge, "is_configured", False):
            judgment, judge_usage = self.judge.evaluate(
                self.judge_instructions,
                build_question_judge_payload(
                    question, s.tier, tables_context, anchors, table_reasons
                ),
                **self.judge_kwargs,
            )
            _add_usage(usage, judge_usage)
            attempt["judge"] = {
                "approved": bool(judgment.get("approved", False)),
                "readability_approval": judgment.get("readability_approval"),
                "topic_linkage_approval": judgment.get("topic_linkage_approval"),
                "grounding_approval": judgment.get("grounding_approval"),
                "table_necessity_approval": judgment.get("table_necessity_approval"),
                "difficulty_approval": judgment.get("difficulty_approval"),
                "feedback": judgment.get("feedback", ""),
                "suggestions": judgment.get("suggestions", ""),
                "panel": judgment.get("panel", {}),
            }
            if not judgment.get("approved", False):
                s.last_failure = "judge"
                s.previous_question = question
                s.feedback = "\n".join(
                    part for part in (
                        judgment.get("feedback", ""),
                        f"Suggestions: {judgment['suggestions']}"
                        if judgment.get("suggestions") else "",
                    ) if part
                )
                self._info(f"Slot {s.slot} attempt {attempt_no} was rejected by the question judges.")
                return

        keywords = covered_terms(question, anchors)
        if not keywords:
            question_tokens = set(tokenize(question))
            keywords = [
                k for k in ((draft or {}).get("question_keywords") or [])
                if tokenize(k) and all(t in question_tokens for t in tokenize(k))
            ][:6]
        s.approved = {
            "slot": s.slot,
            "tier": s.tier,
            "question": question,
            "question_keywords": keywords,
            # why the writer says the question needs each table (what the
            # judge was shown); not passed on to the planner
            "table_reasons": table_reasons,
            # the anchor this question was written against — a later rewrite
            # starts from it
            "anchors": {a: list(t) for a, t in anchors.items()},
            "code_terms": {a: list(t) for a, t in s.code_terms.items()},
        }
        s.last_failure = None
        self._info(f"Slot {s.slot} approved on attempt {attempt_no}: {question}")


@dataclass
class QuestionStageSession:
    """A :class:`QuestionStage` together with the inputs of one run, so the
    caller can come back to it later (``rewrite``) without re-deriving them."""

    stage: QuestionStage
    context: dict
    tables_context: dict
    anchors: dict[str, list[str]]
    contract: Optional[RetrievalContract]
    code_terms: dict = field(default_factory=dict)

    def run(self, tiers: list[str]) -> QuestionStageResult:
        return self.stage.run(
            tiers=tiers,
            context=self.context,
            tables_context=self.tables_context,
            anchors=self.anchors,
            contract=self.contract,
            code_terms=self.code_terms,
        )

    def rewrite(self, requests: list[dict]) -> QuestionStageResult:
        return self.stage.rewrite(
            requests,
            context=self.context,
            tables_context=self.tables_context,
            contract=self.contract,
        )
