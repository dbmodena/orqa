import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orqa.agent.agent import MAX_REFERENCE_QUESTION_CORRECTIONS, MULTI, StatementOrchestrator
from orqa.agent.agents.ReferenceQuestionAgent import ReferenceQuestionAgent
from orqa.agent.utility.retrievability_gate import RetrievalContract, TableContract
from orqa.benchmark.families import FamilyIndex
from orqa.benchmark.retrieval_panel import RetrieverPanel
from orqa.utils.pipeline_logger import PipelineLogger


class TestReferenceQuestionAgentRendering(unittest.TestCase):
    def test_render_tables_includes_reason_facets_anchor_feedback(self):
        block = ReferenceQuestionAgent._render_tables([
            {
                "alias": "Table_0",
                "reason": "Provides January parking ticket counts.",
                "facets": ["period: January 2023"],
                "anchor": ["parking", "tickets"],
                "feedback": "missed the gate",
            }
        ])
        self.assertIn("Table_0: Provides January parking ticket counts.", block)
        self.assertIn("must state: period: January 2023", block)
        self.assertIn("retrieval anchor: parking, tickets", block)
        self.assertIn("previous attempt rejected: missed the gate", block)

    def test_render_tables_omits_optional_lines_when_absent(self):
        block = ReferenceQuestionAgent._render_tables([
            {"alias": "Table_0", "reason": "Provides January counts."}
        ])
        self.assertEqual(block, "- Table_0: Provides January counts.")

    def test_by_alias_keys_on_table_field(self):
        out = ReferenceQuestionAgent._by_alias({
            "questions": [
                {"table": "Table_0", "question": "Q0?", "question_keywords": ["a", "a", "b"]},
                {"table": "", "question": "ignored, no alias"},
            ]
        })
        self.assertEqual(set(out), {"Table_0"})
        self.assertEqual(out["Table_0"]["question"], "Q0?")

    def test_by_alias_empty_on_none(self):
        self.assertEqual(ReferenceQuestionAgent._by_alias(None), {})


@dataclass
class FakeResult:
    resource_id: str


class FakeLexicalIndex:
    """A search index whose ranking is keyed by which question text is asked
    — lets a test control exactly which alias's question "finds" its table."""

    def __init__(self, ranking_by_marker: dict[str, list[str]], default: list[str] | None = None):
        self._ranking_by_marker = ranking_by_marker
        self._default = default or []
        self._records = {}

    def search(self, keywords, top_k: int = 10, only_available: bool = False):
        text = keywords if isinstance(keywords, str) else " ".join(keywords)
        for marker, ranking in self._ranking_by_marker.items():
            if marker in text:
                return [FakeResult(rid) for rid in ranking[:top_k]]
        return [FakeResult(rid) for rid in self._default[:top_k]]

    def get(self, resource_id: str):
        return self._records.get(resource_id)


class FakeReferenceQuestionAgent:
    """Records every call and replays scripted responses — same return
    contract as the real ``ReferenceQuestionAgent.generate``:
    ``({alias: {"question", "question_keywords"}}, usage)``, i.e. already
    shaped by ``_by_alias``, not the raw ``{"questions": [...]}`` client
    payload."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def generate(self, main_question: str, tables: list[dict]):
        self.calls.append(tables)
        response = self._responses.pop(0) if self._responses else {}
        return response, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _bare_orchestrator(
    reference_agent, panel, generate_reference_questions=True, single_table_top_k=1
) -> StatementOrchestrator:
    orch = object.__new__(StatementOrchestrator)
    orch._generate_reference_questions = generate_reference_questions
    orch._retrieval_contract_single_table_top_k = single_table_top_k
    orch._retrieval_panel = panel
    orch._reference_question_agent = reference_agent
    orch._log = PipelineLogger()
    return orch


def _contract(alias="Table_0", resource_id="jan") -> RetrievalContract:
    return RetrievalContract(
        top_k=1,
        min_agreement=1,
        tables=[TableContract(alias, resource_id, resource_id, 1, facets=[], residual_siblings=[])],
    )


class FakeQueryPlan:
    def __init__(self, tables):
        self.tables = tables


class FakeTable:
    def __init__(self, name, reason):
        self.name = name
        self.reason = reason


class TestGenerateReferenceQuestionsForQuery(unittest.TestCase):
    def test_noop_outside_multi_mode(self):
        orch = _bare_orchestrator(FakeReferenceQuestionAgent([]), panel=None)
        out = orch._generate_reference_questions_for_query(
            "single", "main q", FakeQueryPlan([]), {"Table_0": "jan"}, _contract(), {}, {}
        )
        self.assertEqual(out, {})

    def test_noop_when_flag_disabled(self):
        orch = _bare_orchestrator(
            FakeReferenceQuestionAgent([]), panel=None, generate_reference_questions=False
        )
        out = orch._generate_reference_questions_for_query(
            MULTI, "main q", FakeQueryPlan([]), {"Table_0": "jan"}, _contract(), {}, {}
        )
        self.assertEqual(out, {})

    def test_couples_reason_into_the_prompt_payload_and_passes_first_try(self):
        family_index = FamilyIndex(
            [{"dataset_id": "d0", "resource_id": "jan"}], Path("/nonexistent")
        )
        lexical = FakeLexicalIndex({"january": ["jan"]})
        panel = RetrieverPanel(lexical, family_index=family_index)
        fake_agent = FakeReferenceQuestionAgent([
            {"Table_0": {"question": "What is the total of january parking tickets?", "question_keywords": ["january"]}}
        ])
        orch = _bare_orchestrator(fake_agent, panel)
        plan = FakeQueryPlan([FakeTable("Table_0", "Provides January parking ticket counts.")])

        all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        out = orch._generate_reference_questions_for_query(
            MULTI, "total parking tickets Jan+Feb", plan, {"Table_0": "jan"}, _contract(), {}, all_tokens
        )

        self.assertEqual(len(fake_agent.calls), 1)
        self.assertEqual(fake_agent.calls[0][0]["reason"], "Provides January parking ticket counts.")
        self.assertEqual(out["Table_0"]["status"], "success")
        self.assertEqual(
            out["Table_0"]["question"], "What is the total of january parking tickets?"
        )
        # Usage from the (fake) LLM call is folded into the run's token total.
        self.assertEqual(all_tokens["total_tokens"], 2)

    def test_retries_failing_table_with_feedback_then_succeeds(self):
        family_index = FamilyIndex(
            [{"dataset_id": "d0", "resource_id": "jan"}], Path("/nonexistent")
        )
        lexical = FakeLexicalIndex({"january": ["jan"]}, default=["unrelated"])
        panel = RetrieverPanel(lexical, family_index=family_index)
        fake_agent = FakeReferenceQuestionAgent([
            {"Table_0": {"question": "How many tickets in total?", "question_keywords": []}},
            {"Table_0": {"question": "How many january parking tickets?", "question_keywords": ["january"]}},
        ])
        orch = _bare_orchestrator(fake_agent, panel)
        plan = FakeQueryPlan([FakeTable("Table_0", "Provides January parking ticket counts.")])

        all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        out = orch._generate_reference_questions_for_query(
            MULTI, "main q", plan, {"Table_0": "jan"}, _contract(), {}, all_tokens
        )

        self.assertEqual(len(fake_agent.calls), 2)
        # Second call is scoped to the still-failing table and carries the
        # gate's own feedback from the first attempt.
        self.assertIn("feedback", fake_agent.calls[1][0])
        self.assertEqual(out["Table_0"]["status"], "success")

    def test_gives_up_after_max_corrections(self):
        family_index = FamilyIndex(
            [{"dataset_id": "d0", "resource_id": "jan"}], Path("/nonexistent")
        )
        lexical = FakeLexicalIndex({}, default=["unrelated"])
        panel = RetrieverPanel(lexical, family_index=family_index)
        fake_agent = FakeReferenceQuestionAgent([
            {"Table_0": {"question": "How many tickets in total?", "question_keywords": []}}
            for _ in range(1 + MAX_REFERENCE_QUESTION_CORRECTIONS)
        ])
        orch = _bare_orchestrator(fake_agent, panel)
        plan = FakeQueryPlan([FakeTable("Table_0", "Provides January parking ticket counts.")])

        all_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        out = orch._generate_reference_questions_for_query(
            MULTI, "main q", plan, {"Table_0": "jan"}, _contract(), {}, all_tokens
        )

        self.assertEqual(len(fake_agent.calls), 1 + MAX_REFERENCE_QUESTION_CORRECTIONS)
        self.assertEqual(out["Table_0"]["status"], "unretrievable")


if __name__ == "__main__":
    unittest.main()
