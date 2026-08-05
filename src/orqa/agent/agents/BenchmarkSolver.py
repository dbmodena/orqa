"""Benchmark solver agent — the LLM decision points for orqa.benchmark.solve.

Three single-shot calls answering ONE benchmark question in the round-trip
solver loop (see orqa.benchmark.solve.solve_one_question): given a question
ALONE (never the hidden ground truth), decide search keywords, which
retrieved table(s) actually answer it, and the code that answers it. No
phase retries itself — see structured_outputs.SearchKeywords/TableSelection/
SolverCode's module docstring for why: a benchmark solver measures raw
capability against retrieved, not-guaranteed-relevant context, so looping a
phase until it succeeds would measure the loop, not the question's actual
difficulty.

The looping/retrieval/Valentine-matching/execution/scoring machinery that
CALLS this class lives in orqa.benchmark.solve — "the benchmarking module
contains the logic," this module only owns the three LLM calls.
"""
from pathlib import Path
from typing import Any

from ..llm_client.LLMClientStructured import LLMClientStructured
from ..prompting import (
    BenchmarkSearchKeywordsPrompt,
    BenchmarkSolverCodePrompt,
    BenchmarkTableSelectionPrompt,
)

_PANDAS_CODE_RULES = """### Pandas rules
- Each selected table is pre-loaded as a DataFrame under its given alias — start operations directly on it, never `pd.read_csv()`/`pd.DataFrame()`/reassign the alias.
- Column access (mandatory): always use bracket/getitem indexing — `df["column name"]` — never dot/attribute access and never a bare column name inside `.query()`/`.eval()`. Columns can contain spaces, accented characters, punctuation, or be purely numeric text (e.g. `"2023"`), all of which break attribute access or eval-string parsing.
- Prefer method chaining. Correct Python/Pandas only.
- End the code in an assignment or a bare expression whose value IS the answer (e.g. `result = ...` or a trailing `df.groupby(...).sum()`) — the last statement's value is what gets captured."""

_SQL_CODE_RULES = """### SQL rules
- Only reference the selected table aliases — DuckDB/ANSI SQL syntax.
- Column-name quoting (mandatory): always wrap every column reference in double quotes — `"column name"` — even for names that look like ordinary identifiers; a purely-numeric name left unquoted parses as a numeric literal instead of a column reference (wrong result, no error).
- A single query. If a column name itself contains a `"`, double it: `col "a" b` -> `"col ""a"" b"`."""


class _SearchKeywordsClient(LLMClientStructured):
    def __init__(self, config_path: Path):
        super().__init__(config_path, response_model="benchmark_search_keywords")


class _TableSelectionClient(LLMClientStructured):
    def __init__(self, config_path: Path):
        super().__init__(config_path, response_model="benchmark_table_selection")


class _SolverCodeClient(LLMClientStructured):
    def __init__(self, config_path: Path):
        super().__init__(config_path, response_model="benchmark_solver_code")


class BenchmarkSolverAgent:
    """The three-phase decision-making agent for one benchmark question.

    Owns three independent LLMClientStructured clients (mirrors how
    StatementOrchestrator owns _planner/_validator/etc.) — each pinned to
    its own response_model config key in litellm.yaml
    (benchmark_search_keywords / benchmark_table_selection /
    benchmark_solver_code).
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._keywords_client = _SearchKeywordsClient(config_path)
        self._table_client = _TableSelectionClient(config_path)
        self._code_client = _SolverCodeClient(config_path)
        self._keywords_prompt = BenchmarkSearchKeywordsPrompt()
        self._table_prompt = BenchmarkTableSelectionPrompt()
        self._code_prompt = BenchmarkSolverCodePrompt()

    def generate_keywords(self, question: str) -> tuple[dict, dict]:
        """Phase 1: question -> {detected_language, keywords}. (result, usage)."""
        prompt = self._keywords_prompt.update(question=question)
        result, usage = self._keywords_client.complete(prompt, root_key=None)
        return result or {}, usage

    def select_tables(
        self, question: str, candidates: str, relationships: str
    ) -> tuple[dict, dict]:
        """Phase 2: question + retrieved candidates + Valentine relationships
        -> TableSelection dict (tables/expected_result_type/reasoning/
        no_viable_selection). (result, usage)."""
        prompt = self._table_prompt.update(
            question=question, candidates=candidates, relationships=relationships
        )
        result, usage = self._table_client.complete(prompt, root_key=None)
        return result or {}, usage

    def write_code(
        self,
        question: str,
        kind: str,
        expected_result_type: str,
        selected_tables: str,
    ) -> tuple[dict, dict]:
        """Phase 3: question + Phase 2's selection -> {code}. (result, usage)."""
        kind_rules = _PANDAS_CODE_RULES if kind.upper() == "PANDAS" else _SQL_CODE_RULES
        prompt = self._code_prompt.update(
            question=question,
            kind=kind.upper(),
            kind_rules=kind_rules,
            expected_result_type=expected_result_type,
            selected_tables=selected_tables,
        )
        result, usage = self._code_client.complete(prompt, root_key=None)
        return result or {}, usage
