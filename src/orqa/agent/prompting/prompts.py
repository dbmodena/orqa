import json
import logging
import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
import numbers
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, get_args

import numpy as np

from orqa.utils.undecodable_names import escaped_literal, is_undecodable
from .models import ColumnStat, PandasQueryPlan, SQLQueryPlan, TableStats
from .text_profile import CUT_KEEP, CUT_LONG

logger = logging.getLogger(__name__)

T = TypeVar("T")

QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]

# NOTE: This module now lives inside the `prompting` package
# (`src/orqa/agent/prompting/prompts.py`), one directory deeper than the
# original `src/orqa/agent/prompting.py`. An extra `.parent` keeps the resolved
# path pointing at `<repo>/conf/prompts`.
PROMPT_PATH = Path(__file__).parent.parent.parent.parent.parent.joinpath("conf", "prompts")


def _load_prompt(prompt_path: Path, section: Optional[str] = None, **kwargs) -> str:
    """
    Load prompt content from a markdown file.
    Supports variable substitution using {variable_name} syntax.
    Can extract specific sections by header name.

    Args:
        filepath: Path to the markdown file
        section: Optional section header to extract (without ##)
        **kwargs: Variables to inject into the prompt

    Returns:
        Full content or specific section content with variables injected

    Examples:
        # Load entire file
        load_prompt(Path("prompt.md"), name="John")

        # Load specific section
        load_prompt(Path("prompt.md"), section="Analysis Instructions", name="John")
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract specific section if requested
    if section:
        content = _extract_section(content, section)

    # Inject variables
    if kwargs:
        content = content.format(**kwargs)

    return content


def _extract_section(content: str, section_name: str) -> str:
    """
    Extract a specific section from markdown content by header name.
    Supports ## headers at any level.

    Lines inside a fenced code block (```...```) are never treated as
    headers, fence delimiters included — a Python comment like
    `# encode categorical features` inside an embedded code example would
    otherwise be misparsed as a level-1 markdown header and prematurely
    close whatever section it appears in.

    Args:
        content: Full markdown content
        section_name: Header name to find (without ## prefix)

    Returns:
        Content of the section (excluding the header itself)

    Raises:
        ValueError: If section is not found
    """
    lines = content.split("\n")
    section_lines = []
    in_section = False
    section_level = None
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            if in_section:
                section_lines.append(line)
            continue

        # Check if this is a header line (never inside a fenced code block).
        if not in_code_fence and stripped.startswith("#"):
            # Parse header level and title
            header_match = stripped.lstrip("#")
            header_level = len(stripped) - len(header_match)
            header_title = header_match.strip()

            # Check if this is our target section
            if header_title.lower() == section_name.lower():
                in_section = True
                section_level = header_level
                continue  # Skip the header itself

            # If we're in a section and hit a same/higher level header, we're done
            elif in_section and header_level <= section_level:
                break

        # Collect lines if we're in the target section
        if in_section:
            section_lines.append(line)

    if not section_lines:
        raise ValueError(f"Section '{section_name}' not found in markdown file")

    return "\n".join(section_lines).strip()


class Prompt(Generic[T]):
    _prompt_path: Path

    def __init__(self):
        self._current_prompt: str = "Main prompt not formatted yet"

        with open(self._prompt_path, "r") as file:
            self._prompt = file.read()

    def _update(self, **opts) -> str:
        """
        Format the original-prompt with the given formatting updates
        """
        self._current_prompt = self._prompt.format_map(opts)
        return self._current_prompt


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def readable_dataset_name(name: str) -> str:
    """A dataset's name as a prompt may show it: with the portal's ids taken out.

    A downloaded table is named after its files — ``<dataset_id>___<resource_id>``,
    two UUIDs on a CKAN portal, optionally behind a human-readable prefix
    (``parking-tickets__<resource_id>``). The ids identify the record in the
    portal and tell a model nothing about the data, so every UUID is removed
    along with the separators left around it; a name that is nothing but ids
    becomes ``"N/A"`` (the placeholder the metadata blocks already use for a
    missing value). A name with no UUID — an Opendatasoft dataset such as
    ``troncon_voie`` — is already informative and is returned unchanged."""
    cleaned = _UUID_RE.sub(" ", str(name or ""))
    cleaned = re.sub(r"[\s_]{2,}|(?:^|\s)[_-]+|[_-]+(?:$|\s)", " ", cleaned).strip(" _-")
    return cleaned or "N/A"


class DatasetDescription(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("dataset_description.md")

    def update(
        self,
        dataset_name: str,
        num_rows: int,
        num_columns: int,
        dataset_metadata: dict,
        column_details: dict,
        sample_data,
    ) -> str:
        return self._update(
            **{
                "dataset_name": readable_dataset_name(dataset_name),
                "num_rows": num_rows,
                "num_columns": num_columns,
                "dataset_metadata": dataset_metadata,
                "column_details": column_details,
                "sample_data": sample_data,
            }
        )
class LightDatasetDescription(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("light_dataset_description.md")

    def update(
        self,
        dataset_name: str,
        columns: dict,
        sample_data,matches
    ) -> str:
        return self._update(
            **{
                "dataset_name": dataset_name,
                "columns": columns,
                "matches":matches,
                "sample": sample_data,
            }
        )

class CandidatesDiscoveryPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("propose_discovery_tasks.md")

    def update(
        self,
        dataset_name: str,
        num_rows: int,
        num_columns: int,
        dataset_metadata: dict,
        column_details: dict,
        sample_data,
    ) -> str:
        query_dataset_prompt = DatasetDescription()
        query_dataset_description = query_dataset_prompt.update(
            dataset_name,
            num_rows,
            num_columns,
            dataset_metadata,
            column_details,
            sample_data,
        )

        return self._update(query_dataset_description=query_dataset_description)


class PairTaskSelectionPrompt(Prompt):
    """Prompt for selecting discovery operations on one concrete (Q, R) pair.

    Static instructions live at the top of the markdown; the per-pair payload
    (both dataset descriptions, cosine similarity, schema matches) is
    interpolated at the bottom so the shared prefix stays cacheable.
    """

    _prompt_path = PROMPT_PATH.joinpath("select_pair_tasks.md")

    def update(
        self,
        q_description: str,
        r_description: str,
        cosine_similarity: float,
        schema_matches: list,
    ) -> str:
        matches_block = (
            "\n".join(
                f"  {q_col} <-> {r_col} ({score:.3f})"
                for q_col, r_col, score in schema_matches
            )
            or "  (none above threshold)"
        )
        return self._update(
            q_dataset_description=q_description,
            r_dataset_description=r_description,
            cosine_similarity=f"{cosine_similarity:.3f}",
            schema_matches=matches_block,
        )


class PandasStatementGenerationPrompt(Prompt):
    """State container for the multi-table Pandas generation prompt.

    ``agent.py``'s ``StatementOrchestrator._build_generation_prompt`` builds the accumulated
    description blocks itself (once, after the per-dataset loop) and assigns
    them directly to ``_datasets_descriptions``/``_light_datasets_descriptions``
    before calling the base ``Prompt._update`` — it does not use a per-class
    accumulating ``update()`` method, so this class only needs to hold that
    state.
    """
    _prompt_path = PROMPT_PATH.joinpath("pandas_statement_generation.md")

    def __init__(self):
        super().__init__()
        self._datasets_descriptions = ""
        self._light_datasets_descriptions = ""


class SQLStatementGenerationPrompt(Prompt):
    """State container for the multi-table SQL generation prompt.

    See :class:`PandasStatementGenerationPrompt` for why this holds state only.
    """
    _prompt_path = PROMPT_PATH.joinpath("sql_statement_generation.md")

    def __init__(self):
        super().__init__()
        self._datasets_descriptions = ""
        self._light_datasets_descriptions = ""


class JudgementResponseGenerationPrompt(Prompt):
    """Code-judge instructions — the query payload travels in the USER
    message (see ``LLMStatementJudge.complete``), never in this template."""
    _prompt_path = PROMPT_PATH.joinpath("judge.md")

    def __init__(self):
        super().__init__()

    def update(self) -> str:
        return self._update()


class SingleTableJudgementResponseGenerationPrompt(Prompt):
    """Single-table variant of :class:`JudgementResponseGenerationPrompt`."""
    _prompt_path = PROMPT_PATH.joinpath("single_table_judge.md")

    def __init__(self):
        super().__init__()

    def update(self) -> str:
        return self._update()


class PlanJudgementPrompt(Prompt):
    """Plan-judge instructions for the plan judge panel: four plan layers, no
    question checks (the question was approved by the question stage before
    planning and is frozen), and every fix belongs in the plan. The plan
    payload still travels in the USER message (see ``JudgePanel``), never in
    this template. The checks keep the numbers 2-5 they had when the question
    layers were 1 and 6."""
    _prompt_path = PROMPT_PATH.joinpath("plan_judge.md")

    def __init__(self):
        super().__init__()

    def update(self) -> str:
        return self._update()


class ResponseGenerationPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("response_generation.md")

    def __init__(self):
        super().__init__()

    def update(self, question, data) -> str:
        return self._update(data=data, question=question)


class SingleTablePandasPrompt(Prompt):
    """State container for the single-table Pandas generation prompt.

    See :class:`PandasStatementGenerationPrompt` for why this holds state only.
    """
    _prompt_path = PROMPT_PATH.joinpath("single_table_pandas_statement_generation.md")

    def __init__(self):
        super().__init__()
        self._datasets_descriptions = ""
        self._light_datasets_descriptions = ""


class SingleTableSQLPrompt(Prompt):
    """State container for the single-table SQL generation prompt.

    See :class:`PandasStatementGenerationPrompt` for why this holds state only.
    """
    _prompt_path = PROMPT_PATH.joinpath("single_table_sql_statement_generation.md")

    def __init__(self):
        super().__init__()
        self._datasets_descriptions = ""
        self._light_datasets_descriptions = ""


# ---------------------------------------------------------------------------
# Validator correction prompts
# ---------------------------------------------------------------------------

class PandasValidatorCorrectionPrompt(Prompt):
    """
    Prompt fed to the LLM when StatementValidator needs to correct Pandas queries
    that failed static validation and/or were rejected by the judge.

    Template variables:
        {table_schemas}        — pre-formatted DatasetDescription text for all tables
        {column_statistics}    — the run's rendered column-statistics block
        {queries_with_errors}  — formatted pending queries with their errors/feedback
        {pydantic_constraint}  — JSON schema output format instructions
    """
    _prompt_path = PROMPT_PATH.joinpath("pandas_validator_correction.md")

    def update(
        self,
        table_schemas: str,
        column_statistics: str,
        queries_with_errors: str,
        pydantic_constraint: str,
    ) -> str:
        return self._update(
            table_schemas=table_schemas,
            column_statistics=column_statistics,
            queries_with_errors=queries_with_errors,
            pydantic_constraint=pydantic_constraint,
        )


class SQLValidatorCorrectionPrompt(Prompt):
    """
    Prompt fed to the LLM when StatementValidator needs to correct DuckDB SQL queries
    that failed static validation and/or were rejected by the judge.

    Template variables:
        {table_schemas}        — pre-formatted DatasetDescription text for all tables
        {column_statistics}    — the run's rendered column-statistics block
        {queries_with_errors}  — formatted pending queries with their errors/feedback
        {pydantic_constraint}  — JSON schema output format instructions
    """
    _prompt_path = PROMPT_PATH.joinpath("sql_validator_correction.md")

    def update(
        self,
        table_schemas: str,
        column_statistics: str,
        queries_with_errors: str,
        pydantic_constraint: str,
    ) -> str:
        return self._update(
            table_schemas=table_schemas,
            column_statistics=column_statistics,
            queries_with_errors=queries_with_errors,
            pydantic_constraint=pydantic_constraint,
        )


# ---------------------------------------------------------------------------
# One-off prompts used by non-Statement* collaborators (QueryPlanner,
# TableAnalyzer, GenerationCoordinator). Each still owns its own ``_prompt_path``
# ``Path`` object like every other class here, so those collaborators never
# need to import ``PROMPT_PATH`` (or build a path) themselves.
# ---------------------------------------------------------------------------

class GenerationEnrichmentPrompt(Prompt):
    """Enrichment block appended to the generation prompt in ``StatementClient``."""
    _prompt_path = PROMPT_PATH.joinpath("generation_enrichment.md")

    def update(self, table_analysis: str, planning_section: str) -> str:
        return self._update(
            table_analysis=table_analysis,
            planning_section=planning_section,
        )


class QueryPlannerPrompt(Prompt):
    """Structured-plan generation prompt used by ``QueryPlanner``."""
    _prompt_path = PROMPT_PATH.joinpath("query_planner.md")

    def update(
        self,
        task_statement: str,
        ops_statement: str,
        batch_note: str,
        time_context: str,
        table_links: str,
        table_aliases: str,
        table_analysis: str,
        table_sample: str,
        column_statistics: str,
        detected_languages: str,
        table_metadata: str = "",
        fixed_questions: str = "",
    ) -> str:
        return self._update(
            fixed_questions=fixed_questions,
            task_statement=task_statement,
            ops_statement=ops_statement,
            batch_note=batch_note,
            time_context=time_context,
            table_links=table_links,
            table_aliases=table_aliases,
            table_analysis=table_analysis,
            table_sample=table_sample,
            column_statistics=column_statistics,
            detected_languages=detected_languages,
            table_metadata=table_metadata,
        )


class TableAnalyzerBatchPrompt(Prompt):
    """Single batched-analysis prompt used by ``TableAnalyzer.analyze_batch``."""
    _prompt_path = PROMPT_PATH.joinpath("table_analyzer_batch.md")

    def update(self, aliases: str, languages: str, tables: str) -> str:
        return self._update(aliases=aliases, languages=languages, tables=tables)


# ── Search keywords prompt (orqa.retrieval.search_keywords) ──────────────

class BenchmarkSearchKeywordsPrompt(Prompt):
    """Question -> retrieval keywords, in the question's own language."""
    _prompt_path = PROMPT_PATH.joinpath("benchmark_search_keywords.md")

    def update(self, question: str) -> str:
        return self._update(question=question)


# ── Question stage (orqa.agent.question / QuestionStage) ────

class QuestionGeneratorPrompt(Prompt):
    """Verified retrieval anchor + table context -> one question per slot,
    written BEFORE any plan exists."""
    _prompt_path = PROMPT_PATH.joinpath("question_generator.md")

    def update(
        self,
        n_slots: int,
        languages: str,
        time_context: str,
        links_block: str,
        tables_block: str,
        slots_block: str,
    ) -> str:
        return self._update(
            n_slots=n_slots,
            languages=languages,
            time_context=time_context,
            links_block=links_block,
            tables_block=tables_block,
            slots_block=slots_block,
        )


class QuestionJudgePrompt(Prompt):
    """Question-judge instructions — the question and table context travel in
    the USER message (see ``JudgePanel``), never in this template."""
    _prompt_path = PROMPT_PATH.joinpath("question_judge.md")

    def update(self) -> str:
        return self._update()


def _render_plan_steps(plan: QueryPlan) -> str:
    """Render the ordered plan steps as a numbered, human-readable block.

    Renders each plan step (:class:`SQLPlanStep` or :class:`PandasPlanStep`)
    in execution order so the model generates code that follows the
    validated decomposition step-by-step. Only fields that carry information
    are emitted per step to keep the prompt compact.
    """
    lines: List[str] = []
    if plan.question:
        lines.append(f"Question: {plan.question}")
    # The plan's promised result shape — the validator mechanically checks
    # the executed result against expected_result_type and rejects the code
    # on a mismatch, so the generator must shape its final result variable
    # accordingly (e.g. a single number, not a 1-row DataFrame of context
    # columns, when the plan promises 'number').
    expected_type = getattr(plan, "expected_result_type", "") or ""
    if expected_type:
        lines.append(f"Expected result type: {expected_type} (ENFORCED — the executed result must have this shape)")
    expected_desc = getattr(plan, "expected_result_description", "") or ""
    if expected_desc:
        lines.append(f"Expected result: {expected_desc}")
    lines.append("Steps:")
    for step in plan.steps:
        lines.append(f"{step.order}. [{step.op}] {step.description}")
        if step.tables:
            lines.append(f"   - tables: {', '.join(step.tables)}")
        if step.columns:
            lines.append(f"   - columns: {', '.join(step.columns)}")
        # Columns this step CREATES, with what each is computed from. The
        # generator previously saw only `columns`, which mixes what a step
        # reads with what it writes — so the lineage the planner already
        # resolved (and had validated) was thrown away right before the code
        # that has to implement it was written.
        produces = getattr(step, "produces", None) or []
        if produces:
            rendered = ", ".join(
                f"{d.name} ← {d.operation}({', '.join(d.sources)})"
                if d.sources else f"{d.name} ← {d.operation}()"
                for d in produces
            )
            lines.append(f"   - produces: {rendered}")
        if step.columns_role:
            lines.append(
                f"   - columns_role: {json.dumps(step.columns_role, ensure_ascii=False)}"
            )
        if step.params:
            lines.append(
                f"   - params: {json.dumps(step.params, ensure_ascii=False)}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Column statistics block
#
# One header line per table (row count, then the table facts), then one line
# per column, `- name (type): fact; fact; ...`, each label next to its value.
# The phrases below are the contract the prompt legends teach (query_planner.md
# DATA QUALITY, the generation and correction prompts): a wording change here
# must be mirrored there. Every input is a raw value (see ColumnStat); all
# spelling of numbers, quoting and cutting happens here, so the block is
# byte-stable for a given TableStats.
# ---------------------------------------------------------------------------

# A text value longer than CUT_LONG (60) characters shows its first CUT_KEEP
# (59), quoted, with `…` OUTSIDE the quotes: the model can still match it by
# prefix (LIKE 'abc%'), and cutting values over 60 characters took the block
# from 73.0% to 71.4% of the old JSON block's size on the research sample.
# Both limits live in text_profile, which computes cut hints for exactly the
# values cut here: a second copy edited alone printed two different stored
# values identically again (audit T15).

# A column name is quoted when it holds one of the block's own delimiters (fact
# `; `, list `, `, `(type)`, `name:`), the `×` of `v ×n`, the notation of the
# table facts (`t = a + b`, `->`, `..`) or the `#` of shapes, or when it reads
# as a number ("2015", "1/2", "3.5%"), so it is never taken for a value.
_NAME_DELIMITERS = frozenset('():;,"×+=>#')
_NUMBER_LIKE_NAME = re.compile(r"[-+]?[\d.,/]+%?")

# json.dumps leaves these line breaks raw, and str.splitlines() splits on
# them: escaped, a stored value can never break a column's line in two.
_LINE_BREAK_ESCAPES = {ord(ch): f"\\u{ord(ch):04x}" for ch in "\x85\u2028\u2029"}


def _quote(text: str) -> str:
    """A text as a JSON string literal, non-ASCII kept readable."""
    return json.dumps(text, ensure_ascii=False).translate(_LINE_BREAK_ESCAPES)


def _text_value(text: str, hints: Optional[dict] = None) -> str:
    """A stored text value, quoted. Past 60 characters only its first 59
    (trailing whitespace dropped first) are quoted, and `…` follows OUTSIDE
    the quotes: a double-quoted string in the block is always a real stored
    value or the exact start of one. A cut hint (ColumnStat.cut_hints)
    lengthens the cut until no other stored value shares the prefix: the
    audit found two listed values printed identically, and a 59-char prefix
    matching 642 stored values."""
    limit = (hints or {}).get(text)
    if limit:
        kept = text.rstrip()
        if limit >= len(kept):
            return _quote(kept) + ("…" if kept != text else "")
        return _quote(kept[:limit]) + "…"
    if len(text) > CUT_LONG:
        return _quote(text.rstrip()[:CUT_KEEP]) + "…"
    return _quote(text)


def _visible_spacing(literal: str) -> str:
    """Non-ASCII whitespace written as `\\uXXXX`: case/space variants that
    differ only by a no-break or ideographic space must not look identical."""
    return "".join(
        f"\\u{ord(ch):04x}" if ch.isspace() and not ch.isascii() else ch
        for ch in literal
    )


def _shape(shape: str) -> str:
    """A digit-collapsed shape in backticks (a pattern, never a value), with
    control characters and line breaks JSON-escaped so a multi-line stored
    value cannot split the column's line."""
    return "`" + "".join(
        json.dumps(ch)[1:-1] if unicodedata.category(ch) in ("Cc", "Zl", "Zp") else ch
        for ch in shape
    ) + "`"


def _int_text(n: int) -> str:
    """Every digit of an int. str() refuses ints past 4300 digits (Python's
    int-to-str limit, ValueError); Decimal converts in binary with no limit."""
    try:
        return str(n)
    except ValueError:
        return format(Decimal(n), "f")


def _fmt_value(x: Any) -> str:
    """A stored number exactly: whole -> its digits (no `.0`, no exponent, no
    thousands separator), otherwise the shortest string that round-trips the
    float64, or the float32 when the value is one (float32 storage noise
    62.400001525878906 -> 62.4). Never rounded further: a plan copies these
    into filters, and 7-significant-digit rounding changed a shown bound in 52%
    of non-whole float columns."""
    if isinstance(x, numbers.Integral):
        return _int_text(int(x))
    x = float(x)
    if math.isfinite(x) and x.is_integer():
        return str(int(x))
    if x == float(np.float32(x)):
        return np.format_float_positional(np.float32(x), unique=True, trim="-")
    return np.format_float_positional(x, unique=True, trim="-")


def _fmt_bound(x: Any, upper: bool) -> str:
    """A range bound: exact like _fmt_value up to 7 significant digits, else
    rounded OUTWARD (min down, max up) to 7 significant digits but never
    coarser than cents, so money keeps its decimals and an inclusive filter
    copied from the bound still keeps every row. GIS area/length columns
    otherwise spend ~17 digit tokens per bound (Qwen3 tokenizes each digit)
    on precision no plan filters on."""
    s = _fmt_value(x)
    # A whole number (an int of any size, a whole float, inf) is returned as
    # is: rounding to a step of at most 0.01 never changes it, and quantizing
    # one of 27+ digits to cents overflowed Decimal's 28-digit precision
    # (InvalidOperation on a Paris id column whose glued ids parse to 2.47e35,
    # failing the whole table group). A non-whole float64 is below 2**52, so
    # the rounding below needs at most 16 + 2 digits.
    if "." not in s or len(s.lstrip("-").replace(".", "").lstrip("0")) <= 7:
        return s
    d = Decimal(s)
    step = min(Decimal(1).scaleb(d.adjusted() - 6), Decimal("0.01"))
    r = d.quantize(step, rounding=ROUND_CEILING if upper else ROUND_FLOOR)
    return format(r.normalize(), "f")


def _fmt_derived(x: float, iqr: float) -> str:
    """An interpolated quartile, which is no stored value: 3 significant
    digits, or 2 significant digits of the IQR when that is finer (a narrow
    spread around a large value), positional. Whole quartiles stay exact:
    an int column keeps them as ints, and through a float 10**18 + 3 printed
    1000000000000000000, a value that is not stored."""
    if isinstance(x, numbers.Integral):
        return _fmt_value(x)
    x = float(x)
    if not math.isfinite(x) or x.is_integer():
        return _fmt_value(x)
    exp = math.floor(math.log10(abs(x))) - 2
    # Quartiles spanning +-1.14e308 overflow the IQR to inf, and
    # floor(log10(inf)) raised OverflowError; the chained comparison also
    # holds for an int IQR of any size, which float() would overflow.
    if 0 < iqr < math.inf:
        exp = min(exp, math.floor(math.log10(iqr)) - 1)
    if exp >= 0:
        return str(int(round(x, -exp)))
    text = f"{x:.{-exp}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _value(v: Any, hints: Optional[dict] = None) -> str:
    """A stored value: text quoted (and cut), booleans `true`/`false`, numbers
    exact; None, the null group of a total-rows marker, `null`."""
    if isinstance(v, str):
        return _text_value(v, hints)
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return _fmt_value(v)


def _token_mark(v: Any, tokens: frozenset) -> str:
    """Marks a configured missing-value token kept in a complete list: a token
    can be a real category (a "-"/"+" sign flag), so the planner decides."""
    return " (missing token)" if isinstance(v, str) and v in tokens else ""


def _item(v: Any, rows: int, tokens: frozenset = frozenset(), hints: Optional[dict] = None) -> str:
    """`v ×n`: value v in n rows."""
    return f"{_value(v, hints)} ×{rows}{_token_mark(v, tokens)}"


def _name(name: str) -> str:
    """A column name, bare unless it could be misread (see _NAME_DELIMITERS);
    never cut."""
    if (
        name == ""
        or name != name.strip()
        or ".." in name
        or "->" in name
        or any(
            ch in _NAME_DELIMITERS
            or unicodedata.category(ch) == "Cc"
            or (ch.isspace() and not ch.isascii())
            for ch in name
        )
        or _NUMBER_LIKE_NAME.fullmatch(name)
    ):
        return _quote(name)
    return name


def _type_label(c: ColumnStat) -> str:
    """The profile's type label, or the short name of the pandas dtype."""
    if c.type_label:
        return c.type_label
    dtype = c.dtype.lower()  # nullable dtypes (Int64, Float64, boolean) too
    if dtype.startswith(("int", "uint")):
        return "int"
    if dtype.startswith("float"):
        return "float"
    if dtype.startswith("bool"):
        return "bool"
    if dtype.startswith("datetime64"):
        return "datetime"
    if dtype == "object":
        return "str"
    return c.dtype


def _count(n: int, noun: str) -> str:
    """`1 row`, `2 rows`: a count with its noun in the matching number (the
    audit read `1 rows repeat an earlier row` and `1 fully empty rows`)."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _usable_rows(c: ColumnStat, rows: int) -> int:
    """Rows holding a usable value: not null, blank or a missing token."""
    token_rows = sum(n for _, n in c.missing_tokens) + c.missing_tokens_other_rows
    return rows - c.nan_count - c.blank - token_rows


def _percent(share: float) -> str:
    """A share as a whole percent, never 0 for some rows nor 100 for not all."""
    p = round(100 * share)
    if p < 1 and share > 0:
        return "<1"
    if p > 99 and share < 1:
        return ">99"
    return str(p)


def _domain(c: ColumnStat, rows: int) -> str:
    """The first fact: what values the column holds, as a whole."""
    if c.cardinality == 0:
        return "all null"
    if c.same_as is not None:
        return f"same values as {_name(c.same_as)}"
    if c.blank and c.blank == rows - c.nan_count:
        return "only blank"
    tokens = frozenset(c.missing_token_values)
    if c.constant is not None:
        if c.nan_count or c.blank:
            # One stored value beside nulls is no constant: in a 1/blank flag
            # null means "no" (Paris cm: 1 ×48, 115 null), and `constant`
            # read with "null is missing" made every councillor a member.
            return f"1 value: {_item(c.constant, rows - c.nan_count - c.blank, tokens, c.cut_hints)}"
        return f"constant {_value(c.constant, c.cut_hints)}{_token_mark(c.constant, tokens)}"
    if c.values:
        if all(n == 1 for _, n in c.values):
            # " ×1" on every item would only repeat "each once".
            return f"{len(c.values)} values, each once: " + ", ".join(
                _value(v, c.cut_hints) + _token_mark(v, tokens) for v, _ in c.values
            )
        return f"{len(c.values)} values: " + ", ".join(_item(v, n, tokens, c.cut_hints) for v, n in c.values)
    if c.uniform_count and c.uniform_count > 1:
        # A balanced grid. A cut `most common` list hid it: Paris
        # arrondissement showed 5 of its 20 values ×13 as if more frequent
        # than the other 15.
        return f"{_usable_rows(c, rows) // c.uniform_count} distinct, each ×{c.uniform_count}"
    if c.usable_distinct is not None:
        # Exact for text: cardinality also counts blank and token spellings
        # (audit: 22 Welsh names seen once each beside 152 blanks read
        # "23 distinct" instead of "all distinct").
        if c.usable_distinct > 1 and c.usable_distinct == _usable_rows(c, rows):
            return "all distinct"
        return f"{c.usable_distinct} distinct"
    # Only non-text columns get here (the text profile always sets
    # usable_distinct), and they have no blank or token spellings.
    if 1 < rows - c.nan_count <= c.cardinality:
        return "all distinct"
    return f"{c.cardinality} distinct"


# The marks text_profile._number_format and its ambiguous check emit.
_SEPARATOR_WORDS = {",": "comma", ".": "dot", "space": "space"}


def _separator(mark: str) -> str:
    # An unmapped mark shows as a backticked shape rather than raising, which
    # would fail the whole table group.
    return _SEPARATOR_WORDS.get(mark) or _shape(mark)


def _number_text_facts(number: dict, ranges: bool) -> List[str]:
    """How to cast a number-text column: its format, then its parsed range
    (left out when the domain already lists every value), then the values
    stored without the column's affix, which are on another scale (NYC
    Percentage: 96 values "85%" and 12 fractions 0.855, merged in one range
    0.072..89 that made 0.855 read as 0.855%)."""
    # Separators are named in words and affixes written in backticks, so a
    # double-quoted string in the block stays a stored value (never a lone
    # "," that a generator could copy as a literal).
    if number.get("ambiguous"):
        return [f"number text: {_separator(number['ambiguous'])} ambiguous (thousands or decimal)"]
    if number.get("mixed"):
        return ["number text: decimal dot and comma mixed"]
    parts = [f"decimal {_separator(number['decimal'])}"] if "decimal" in number else []
    if "thousands" in number:
        parts.append(f"thousands {_separator(number['thousands'])}")
    affixes = [f"{key} {_shape(number[key])}" for key in ("prefix", "suffix", "unit") if key in number]
    parts.extend(affixes)
    # No separator or affix to remove: the text casts as is once trimmed.
    facts = ["number text: " + "; ".join(parts or ["plain"])]
    if ranges and "min" in number:
        facts.append(f"range {_fmt_bound(number['min'], False)}..{_fmt_bound(number['max'], True)}")
    bare = number.get("bare")
    if bare and affixes:
        fact = f"{_count(bare['rows'], 'row')} without the " + " and the ".join(affixes)
        if ranges:
            fact += f", range {_fmt_bound(bare['min'], False)}..{_fmt_bound(bare['max'], True)}"
        facts.append(fact)
    return facts


def _also(date: dict) -> str:
    """The column's other formats of the same family (seconds or not, 2- or
    4-digit year), which parse too and are inside the range: without them a
    reader parses with the main format alone and turns those rows NULL."""
    others = [f"{a['format']} ({_count(a['rows'], 'row')})" for a in date.get("also", [])]
    return "; also " + ", ".join(others) if others else ""


def _offsets(date: dict, parsed_rows: Optional[int]) -> str:
    """`offsets +00:00 on every value` / `on 1200 of 2000 values`: how many of
    the parsed ISO values carry a UTC offset. The generator keys its parse on
    it: with an offset on every value the text casts as TIMESTAMPTZ and is
    taken to UTC; with some, a mask splits the two parses (pandas otherwise
    lets a naive value inherit the preceding value's offset, so the result
    depended on the row order). More than 3 distinct offsets are not listed."""
    shown = "/".join(date["offsets"]) if date.get("offsets") else "(more than 3 distinct)"
    n = date.get("offset_rows")
    if n is None:
        return f"offsets {shown}"
    if parsed_rows is None or n >= parsed_rows:
        return f"offsets {shown} on every value"
    return f"offsets {shown} on {n} of {parsed_rows} values"


def _date_facts(date: dict, ranges: bool, example: bool, parsed_rows: Optional[int] = None) -> List[str]:
    """A date-text column's format and parsed range (a datetime-dtype column
    has no format, only the range); dates are always after `parsed`, and are
    UTC (a value with no offset is taken as UTC). A constant keeps only the
    format: its range repeats the one value (a 12.7M-row NYC column spent ~25
    tokens on `parsed 2026-07-26..2026-07-26`), and a listed domain already
    shows a stored example. `parsed_rows` is the number of rows holding a
    parsed value, the denominator of the offsets fact."""
    if date.get("time_only"):
        # A clock time with no date part: AS DATE / AS TIMESTAMP turn every
        # value NULL while AS TIME parses it (UK Start/End Time, all HH:MM,
        # were `date-like text, format not detected`).
        facts = [f"time text {date['format']}{_also(date)}"]
        if ranges and "min" in date:
            facts.append(f"range {date['min']}..{date['max']}")
            # The profile keeps a repeated extreme out of `most common`, so
            # only this fact shows it: 00:00 on 300 of 700 rows appeared
            # nowhere, and UK Avg. Time on Page hid "00:00:00" ×178965.
            for side in ("min", "max"):
                if date.get(f"{side}_rows"):
                    facts.append(f"repeated {side} {date[side]} in {_count(date[f'{side}_rows'], 'row')}")
        return facts
    facts = []
    if "format" in date:
        fmt = date["format"]
        # The offsets are their own fact (Bologna's columns are all +00:00, a
        # local clock stored as UTC, and grouping them in UTC made a phantom
        # first day, month or year of 4-8 rows), so the ISO label stays bare.
        has_offsets = "offsets" in date or "offset_rows" in date
        if has_offsets and fmt == "ISO with UTC offset":
            fmt = "ISO"
        # A month-name format pandas accepts but DuckDB spells differently.
        duck = f" (DuckDB: {date['duckdb_format']})" if date.get("duckdb_format") else ""
        order = f" or {date['alt_format']} (order unknown)" if date.get("alt_format") else ""
        shown = f", e.g. {_text_value(date['example'])}" if example and date.get("example") else ""
        facts.append(f"date text {fmt}{duck}{order}{_also(date)}{shown}")
        if has_offsets:
            facts.append(_offsets(date, parsed_rows))
    if ranges:
        if "years" in date:
            facts.append(f"years {date['years'][0]}..{date['years'][1]}")
        if "min" in date:
            # Coverage of the span: "every day present", "every month
            # present" or "k of n months present" (D3), so a plan knows
            # whether a period it scopes is actually there.
            coverage = f", {date['coverage']}" if date.get("coverage") else ""
            facts.append(f"parsed {date['min']}..{date['max']}{coverage}")
        for side in ("min", "max"):
            if date.get(f"{side}_rows"):
                facts.append(f"parsed {side} {date[side]} repeated in {_count(date[f'{side}_rows'], 'row')}")
    if date.get("time"):
        facts.append("time always " + "/".join(date["time"]))
    unit = date.get("unit")
    if unit == "month":
        facts.append("only first day of month")
    elif unit == "month_end":
        # 178027 Valencia rows, all month ends over 68 months, had no
        # granularity fact and read as daily data.
        facts.append("only last day of month")
    elif unit == "year":
        facts.append("only 1 January")
    return facts


def _nonconforming(label: str, c: ColumnStat) -> List[str]:
    """`not numbers (rows): ...`: the usable values a typed column's parse
    would turn into NULL, or that its code shape does not fit (`not
    matching`), one value or one shape with an example per group."""
    if not c.other:
        return []
    groups = [
        f"{_value(g['value'], c.cut_hints)} ×{g['rows']}"
        if "value" in g
        else f"{_shape(g['shape'])} ×{g['rows']} e.g. {_value(g['example'], c.cut_hints)}"
        for g in c.other
    ]
    return [f"{label} ({c.other_rows}): " + ", ".join(groups)]


def _kind_facts(c: ColumnStat, rows: int) -> List[str]:
    """The text kind (at most one is set), with its non-conforming values.

    A domain that lists every value (a constant or `<k> values:`) already
    shows the non-conforming values and the number-text bounds, so both are
    left out (Epigrafe repeated a 50-char listed value under `not numbers`
    and added `range 1..1`); a constant also drops its parsed date range."""
    constant = c.constant is not None
    listed = constant or bool(c.values)

    def nonconforming(label: str) -> List[str]:
        return [] if listed else _nonconforming(label, c)

    if c.number:
        return _number_text_facts(c.number, not listed) + nonconforming("not numbers")
    if c.date:
        label = "not times" if c.date.get("time_only") else "not dates"
        # Rows holding a parsed value: usable rows minus the non-conforming.
        parsed_rows = _usable_rows(c, rows) - c.other_rows
        return _date_facts(c.date, not constant, not listed, parsed_rows) + nonconforming(label)
    if c.date_like_shape:
        return [f"date-like text, format not detected, shape {_shape(c.date_like_shape)}"]
    if c.digits:
        d = c.digits
        width = _count(d["width"], "digit") if "width" in d else f"{d['widths'][0]}..{d['widths'][1]} digits"
        return [f"digit code {width}, {_count(d['zero_led'], 'row')} with leading zeros"] + nonconforming("not codes")
    if c.geometry:
        return [f"geometry {c.geometry}"]
    if c.text_kind == "free text":
        return [f"free text, median {_count(c.text_length, 'char')}"]
    if c.text_kind:
        return [c.text_kind]
    if c.shape:
        # The rows outside the shape were the ones that mattered each time:
        # open age bands "014 e prima"/"110 e oltre", the latest month stored
        # as "MAY 2026", 53 column-shifted rows of a broken CSV quote.
        return [f"code shape {_shape(c.shape)}"] + nonconforming("not matching")
    return []


def _far_high(far: dict) -> str:
    """The values above the widest gap at the top, or the range of a separate
    block of codes above it, with the bound every other row stays under.
    Spelled like the range's upper bound, so the max never appears twice in
    two spellings (`range ..37315906.67` beside `far high 37315906.66603644`
    cost ~17 digit tokens); plans exclude these rows by the bound, never by
    listing values."""
    values = far["values"]
    if far.get("block"):
        head = f"far high block {_fmt_bound(min(values), False)}..{_fmt_bound(max(values), True)}"
    else:
        head = "far high " + ", ".join(_fmt_bound(v, True) for v in values)
    return f"{head} ({_count(far['rows'], 'row')}, all others <= {_fmt_bound(far['below'], True)})"


# Missing-token spellings listed before the rest are folded into one count:
# NYC STREET_HIGHWAY listed 8 spellings covering 0.02% of its rows (~60 tokens).
MISSING_TOKENS_SHOWN = 4


def _missing_tokens(c: ColumnStat) -> str:
    """`missing tokens "n/a" ×40, "-" ×2, +3 rows in 2 other spellings`: at most
    MISSING_TOKENS_SHOWN spellings, most rows first; the rest as one count. The
    number of other spellings is given only when the profile listed every
    spelling, or when it gives the count of unlisted ones
    (missing_tokens_other {rows, spellings})."""
    plausible = set(c.missing_tokens_plausible)
    shown, hidden = c.missing_tokens[:MISSING_TOKENS_SHOWN], c.missing_tokens[MISSING_TOKENS_SHOWN:]
    items = [
        _item(v, n, hints=c.cut_hints) + (" (looks like a value)" if v in plausible else "")
        for v, n in shown
    ]
    rest = sum(n for _, n in hidden) + c.missing_tokens_other_rows
    if rest:
        other = c.missing_tokens_other or {}
        if other.get("spellings") is not None:
            where = _count(len(hidden) + int(other["spellings"]), "other spelling")
        elif c.missing_tokens_other_rows:
            where = "other spellings"
        else:
            where = _count(len(hidden), "other spelling")
        items.append(f"+{_count(rest, 'row')} in {where}")
    return "missing tokens " + ", ".join(items)


def _sequence(s: dict) -> str:
    """`sequence step 1, 22 gaps, rows ascending`: distinct whole numbers on an
    evenly stepped run, a record id or row number."""
    return f"sequence step {_fmt_value(s['step'])}, {_count(s['gaps'], 'gap')}, rows {s['row_order']}"


def _identifier(c: ColumnStat) -> str:
    """`identifier: <evidence>`: the structural evidence that the column names
    records, never a verdict (a measure constant per category reads the same
    way; the planner judges). A sequence is restated here, so the `sequence`
    fact is not rendered a second time."""
    ident = c.identifier
    kind = ident.get("kind")
    names = ", ".join(map(_name, ident.get("of") or []))
    if kind == "sequence":
        return f"identifier: record id ({_sequence(c.sequence) if c.sequence else 'sequence'})"
    if kind == "unique":
        return "identifier: unique in every row"
    if kind == "unique_distinct_rows":
        return "identifier: unique in every distinct row"
    if kind == "code_of":
        return f"identifier: code of {names}"
    if kind == "row_key":
        return f"identifier: part of the row key ({names})"
    if kind == "fixes":
        return f"identifier: each value fixes {names}"
    # An evidence kind this renderer does not know: shown rather than raised,
    # which would fail the whole table group.
    return f"identifier: {kind}"


def _filled_down(filled: dict) -> str:
    """`filled down: 5 of 72 rows hold a value, each starting a block of blank
    rows`: a spreadsheet fill-down (UK ministers' meetings: the minister is
    written once, then blank until the next one)."""
    verb = "holds" if filled["filled"] == 1 else "hold"
    return (f"filled down: {filled['filled']} of {filled['rows']} rows {verb} a value, "
            "each starting a block of blank rows")


def _column_line(c: ColumnStat, rows: int) -> str:
    """`- name (type): fact; fact; ...`, facts in the fixed order: how to
    write a name holding U+FFFD, domain, nulls, text cleanliness, numeric
    shape, text kind, identifier evidence, frequent values, numeric tail,
    patterns. A fact with nothing to report is left out."""
    facts = [_domain(c, rows)]
    if c.nan_count and c.cardinality:
        facts.append(f"{c.nan_count} null ({_percent(c.nan_count / rows)}%)")
    if c.same_as is None and c.cardinality:
        # A complete list already shows every token and every spelling.
        listed = c.constant is not None or bool(c.values)
        if c.blank and facts[0] != "only blank":
            facts.append(f"{c.blank} blank")
        if c.filled_down:
            facts.append(_filled_down(c.filled_down))
        if c.missing_tokens and not listed:
            facts.append(_missing_tokens(c))
        if c.untrimmed:
            merges = f" ({c.untrimmed_merges} also stored trimmed)" if c.untrimmed_merges else ""
            facts.append(f"{c.untrimmed} untrimmed{merges}")
        if c.variants and not listed:
            v = c.variants
            fact = f"case/space variants: {_count(v['groups'], 'group')}, {_count(v['rows'], 'row')}"
            if v.get("examples"):
                examples = ", ".join(_visible_spacing(_text_value(x, c.cut_hints)) for x in v["examples"])
                fact += f" (e.g. {examples})"
            facts.append(fact)
        # Identifier evidence replaces the numeric tail: quartiles, far high,
        # repeated extremes and the negative count of a code are noise, and a
        # sequence is restated inside the identifier fact.
        ident = c.identifier is not None
        if c.date_number:
            # Whole numbers that are calendar dates (20200416): the parsed
            # range stands in for the numeric range and quartiles; a constant
            # already shows the one value.
            facts.append("date as number yyyymmdd")
            if c.constant is None:
                facts.append(f"parsed {c.date_number['min']}..{c.date_number['max']}")
        else:
            if c.numeric_min is not None:
                facts.append(f"range {_fmt_bound(c.numeric_min, False)}..{_fmt_bound(c.numeric_max, True)}")
            if c.sequence and not (ident and c.identifier.get("kind") == "sequence"):
                facts.append(_sequence(c.sequence))
            if c.quartiles and not ident:
                iqr = c.quartiles[-1] - c.quartiles[0]
                facts.append("quartiles " + ", ".join(_fmt_derived(q, iqr) for q in c.quartiles))
        facts.extend(_kind_facts(c, rows))
        if ident:
            facts.append(_identifier(c))
        if c.top_values:
            facts.append("most common " + ", ".join(_item(v, n, hints=c.cut_hints) for v, n in c.top_values))
        if c.top_values_tied:
            # Never a cut inside a tie: Bologna candidato listed 4 of its 676
            # candidates ×452, picked by file order, as the most common.
            t = c.top_values_tied
            facts.append(
                f"+{t['values']} more ×{t['rows']} each"
                if c.top_values
                else f"most common: {_count(t['values'], 'value')} ×{t['rows']} each"
            )
        if not ident:
            if c.far_high:
                facts.append(_far_high(c.far_high))
            for side in ("min", "max"):
                if side in c.repeated_extreme:
                    rep = c.repeated_extreme[side]
                    facts.append(f"repeated {side} {_fmt_value(rep['value'])} in {_count(rep['rows'], 'row')}")
            if c.negative_rows:
                facts.append(f"{c.negative_rows} negative")
        if c.infinite_rows:
            facts.append(f"{c.infinite_rows} infinite")
        if c.patterns:
            facts.append("patterns " + ", ".join(f"{_shape(p)} ×{n}" for p, n in c.patterns))
    if is_undecodable(c.column):
        # First, next to the name it is about (prepended last: facts[0] is
        # read as the domain above).
        facts.insert(0, _undecodable_name(c.column))
    return f"- {_name(c.column)} ({_type_label(c)}): " + "; ".join(facts)


def _undecodable_name(name: str) -> str:
    """How to write a header holding U+FFFD (a byte of the source file that
    could not be decoded, e.g. the "£" of "Total £"): the model otherwise
    writes the symbol it thinks was lost. Python and JSON take the ASCII
    escape; DuckDB has no escape for it in an identifier."""
    return (
        "name holds U+FFFD, an undecodable character that is part of it: "
        f"write {escaped_literal(name)} in Python or JSON, the character itself in SQL"
    )


def _names(names: Sequence[str]) -> str:
    """One column name bare, several as `(a, b)`."""
    return _name(names[0]) if len(names) == 1 else "(" + ", ".join(map(_name, names)) + ")"


def _repeated_rows(repeated: dict, fill_cols: Sequence[str] = ()) -> str:
    """`<k> rows repeat an earlier row`, with the evidence that tells double
    entries from real repeats: a near-unique column the repeats still agree on,
    or none at all (UK parking tickets: 8317 repeats, 7855 of them real
    tickets in a log no column identifies, 462 copies of an embedded header
    row ×463). With fill-down columns, the count that remains once they are
    forward-filled, when it differs: `(none once Minister is filled down)` /
    `(12 once Minister is filled down)` (the blank rows under one minister
    repeat each other only until the fill)."""
    rows = repeated["rows"]
    # rows 0: repeats that exist only once the fill-down columns are filled.
    head = "no row repeats" if rows == 0 else "1 row repeats" if rows == 1 else f"{rows} rows repeat"
    after = repeated.get("after_fill")
    if fill_cols and after is not None and after != rows:
        left = "none" if after == 0 else str(after)
        verb = "is" if len(fill_cols) == 1 else "are"
        return f"{head} an earlier row ({left} once {', '.join(map(_name, fill_cols))} {verb} filled down)"
    unique = repeated.get("unique_col")
    who = f"identical even in {_name(unique)}" if unique is not None else "no column identifies rows"
    return f"{head} an earlier row ({who}, largest ×{repeated['largest']})"


def _total_rows(group: dict) -> str:
    """Rollup rows: the marker, how many of the rows carrying it are sums, and
    the group they sum. NYC's 540 null-marker rows held 392 sums; 72 were
    alone in their group, so `exclude the marker rows` without the count and
    the group dropped whole indicators."""
    rows, of = group["rows"], group["of"]
    count = f"{rows} of {of} rows" if of != rows else _count(rows, "row")
    # An empty group: one dimension, no grand total.
    verb = "holds" if rows == 1 else "hold"
    head = f"total rows: {_name(group['col'])} {_value(group['value'])} ({count}) {verb}"
    if not group["group"]:
        # One dimension: each marker row is the sum of every other row.
        return f"{head} the sum of all other rows"
    return f"{head} sums of the other rows with the same ({', '.join(map(_name, group['group']))})"


def _table_header(table: TableStats) -> str:
    """`Table_0: 6279 rows; <table fact>; ...`, facts in the fixed order (no
    fact contains `; `), every name list in column order."""
    s = table.structure
    # `1 row`: 9 of 1029 sampled real tables have one row (Paris plub_senat).
    facts = [f"{table.alias}: {_count(table.num_rows, 'row')}"]
    if table.sampled_from is not None:
        # The counts below are the sample's; the code runs on every row.
        facts = [
            f"{table.alias}: {_count(table.sampled_from, 'row')}",
            f"the facts below come from a uniform sample of {table.num_rows} of them",
        ]
    if s.get("row_key_col") is not None:
        # What a row is, first: the one column whose identifier evidence is a
        # sequence or uniqueness (A); a multi-column key is `one row per (a, b)`.
        facts.append(f"one row per {_name(s['row_key_col'])}")
    if s.get("empty_rows"):
        facts.append(_count(s["empty_rows"], "fully empty row"))
    if s.get("repeated_rows"):
        fill_cols = list(s.get("fill_down") or s.get("fill_down_cols") or ())
        facts.append(_repeated_rows(s["repeated_rows"], fill_cols))
    if s.get("total_row"):
        t = s["total_row"]
        label = (
            f"{_name(t['label_col'])} {_value(t['label'])}"
            if t.get("label_col") is not None
            else "labels blank"
        )
        facts.append(
            f"total row: {_name(t['measure'])} = {_value(t['value'])} "
            f"equals the sum of the other rows ({label})"
        )
    facts.extend(_total_rows(g) for g in s.get("total_rows", []))
    grain = s.get("grain")
    if grain:
        cols, nested = grain["cols"], grain.get("nested", {})
        facts.append("one row per (" + ", ".join(map(_name, cols)) + ")")
        # A part that restarts in every parent group is an id only together
        # with its parents: counting its distinct values alone undercounts.
        facts.extend(f"{_name(part)} numbered within {_names(nested[part])}" for part in cols if part in nested)
    if s.get("one_to_one"):
        pairs = []
        for entry in s["one_to_one"]:
            # `except` is omitted when 0.
            cols, broken = entry["cols"], entry.get("except", 0)
            pair = "(" + ", ".join(map(_name, cols)) + ")"
            # The rows breaking a near-perfect pairing are a cleaning target
            # (UK WD16CD: 53 rows shifted by a broken CSV quote), not a
            # second dimension.
            pairs.append(f"{pair} except {_count(broken, 'row')}" if broken else pair)
        facts.append("one-to-one: " + ", ".join(pairs))
    determines = s.get("determines", [])
    for a, bs in determines:
        facts.append(f"each {_name(a)} has one " + ", ".join(map(_name, bs)))
    if determines and s.get("determines_more"):
        # The 6-edge cap dropped some: say so, or the list reads as complete.
        facts[-1] += f" (+{s['determines_more']} more)"
    for total, parts in s.get("sum_columns", []):
        facts.append(f"{_name(total)} = " + " + ".join(map(_name, parts)))
    return "; ".join(facts)


def render_column_statistics(stats: Sequence[TableStats]) -> str:
    """Render the per-table column-statistics block the planner, generation
    and correction prompts share: a header line per table with its table
    facts, then one labelled line per column.

    Labelled lines rather than JSON: each label sits next to its value, keys
    and empty fields cost nothing, and numbers are spelled with no digit noise
    (Qwen3 spends one token per digit). Text values are JSON-quoted and
    numbers never are, so the model can tell the stored text "1170" from the
    number 1170.
    """
    if not stats:
        return "(no column statistics available)"
    lines: List[str] = []
    for table in stats:
        lines.append(_table_header(table))
        lines.extend(_column_line(c, table.num_rows) for c in table.columns)
    return "\n".join(lines)


def build_generation_prompt(
    base_prompt: str,
    plan: Optional[QueryPlan] = None,
    stats: Optional[Sequence[TableStats]] = None,
) -> str:
    """Build the plan-grounded generation prompt.

    Assembles the generation prompt from the caller-provided ``base_prompt``
    plus, in order (run-stable sections first, the per-call plan last, so the
    run's several generation calls share the longest possible cached prefix):

    * ``### Column Statistics`` — the per-table column-statistics block from
      ``stats``, rendered only when statistics are supplied.
    * ``### Structured Plan`` — the ordered plan steps of ``plan``, rendered
      only when a plan is supplied. Placed last because it is the only
      per-call section.

    All plan/statistics parameters are optional and default to ``None``; when
    absent the corresponding section renders nothing.

    Args:
        base_prompt: The base generation prompt supplied by the caller.
        plan: The validated structured plan whose ordered steps ground the
            generation. Omitted section when ``None``.
        stats: Per-table column statistics rendered into the prompt. Omitted
            section when ``None`` or empty.

    Returns:
        The fully-assembled generation prompt string.
    """
    # Section order is cache-driven: the several generation calls of one run
    # share ``base_prompt`` and ``stats`` verbatim, so those come first and
    # form a common prefix the provider's prompt cache serves on every call
    # after the first. The Structured Plan — the only part unique to each
    # call — is appended LAST so it never invalidates the shared prefix.
    sections: List[str] = [base_prompt]

    if stats:
        sections.append(
            f"### Column Statistics\n{render_column_statistics(stats)}"
        )

    if plan is not None:
        sections.append(f"### Structured Plan\n{_render_plan_steps(plan)}")

    return "\n\n".join(sections)
