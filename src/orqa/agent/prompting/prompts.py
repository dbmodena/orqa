import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, get_args

from .models import _ML_TASK_TYPES, PandasQueryPlan, SQLQueryPlan, TableStats

logger = logging.getLogger(__name__)

T = TypeVar("T")

QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]

# NOTE: This module now lives inside the `prompting` package
# (`src/orqa/agent/prompting/prompts.py`), one directory deeper than the
# original `src/orqa/agent/prompting.py`. An extra `.parent` keeps the resolved
# path pointing at `<repo>/conf/prompts`.
PROMPT_PATH = Path(__file__).parent.parent.parent.parent.parent.joinpath("conf", "prompts")

# Per-task-type ML skill markdowns (classification.md, regression.md, ...), a
# sibling of the prompts directory.
SKILLS_PATH = PROMPT_PATH.parent.joinpath("skills")

# Single source of truth for skill ORDER, derived from the same Literal the
# plan schema uses (models._ML_TASK_TYPES) rather than a separately
# maintained tuple. Used to canonicalize skill-set injection (generation AND
# plan-judge) so two plans proposing the same set of skills in a different
# declaration order still produce byte-identical injected text — see
# PlanJudgementPrompt.update.
_CANONICAL_SKILL_ORDER: tuple = get_args(_ML_TASK_TYPES)


# Every ``conf/skills/{task_type}.md`` is ONE file structured into these
# three top-level (`#`) sections, so a skill's guidance across all three
# pipeline stages stays next to itself instead of drifting apart across
# separate files: what the GENERATION model should write, what the PLAN
# JUDGE should check about proposing the skill (justified by the question?),
# and what the CODE JUDGE should check about the skill's implementation
# (technically sound?). Each stage still only ever receives its OWN
# extracted section — this is a single source of truth for authors, not a
# merge of what gets sent to any one model.
_SKILL_SECTION_HEADERS = {
    "generation": "Generation",
    "plan_judge": "Plan Judge Check",
    "code_judge": "Code Judge Check",
}


def load_skill_section(task_type: str, stage: str) -> str:
    """Load one pipeline stage's section from a skill's consolidated markdown.

    ``stage`` is one of ``"generation"``, ``"plan_judge"``, ``"code_judge"``
    (see ``_SKILL_SECTION_HEADERS``). Used by :func:`build_skill_sections`
    (generation), :class:`PlanJudgementPrompt` (is proposing this skill
    justified by the question?), and :class:`JudgementResponseGenerationPrompt`
    / :class:`SingleTableJudgementResponseGenerationPrompt` (is THIS code's
    implementation of an already-approved skill correct?) — the plan judge
    never re-litigates the skill's implementation and the code judge never
    re-litigates whether it belongs; each stage's section covers only its
    own question.
    """
    header = _SKILL_SECTION_HEADERS[stage]
    return _load_prompt(SKILLS_PATH / f"{task_type}.md", section=header)


def _canonical_task_types(task_types: Optional[Sequence[str]]) -> List[str]:
    """De-duplicate and sort ``task_types`` into ``_CANONICAL_SKILL_ORDER``.

    Plans derive ``task_types`` in first-appearance order within their own
    steps (see QueryPlanner._derive_task_types), so two plans using the same
    SET of skills can list them in different orders. Canonicalizing here
    means every caller building skill-conditioned text from a task_types set
    (the plan judge and the code judge) produces byte-identical output for
    the same set regardless of that per-plan ordering — the whole point being
    to keep the number of distinct rendered variants equal to the number of
    distinct skill SETS in play, not the number of orderings.
    """
    present = set(task_types or [])
    return [t for t in _CANONICAL_SKILL_ORDER if t in present]


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
                "dataset_name": dataset_name,
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
    message (see ``LLMStatementJudge.complete``), never in this template.

    Not byte-identical across every call: one ``**Skill Check — {skill}
    (code)**`` section is appended per distinct skill the query's APPROVED
    plan used (see ``load_skill_section(..., "code_judge")``), canonicalized
    via :func:`_canonical_task_types` so every call judging the same SET of
    skills still shares one cached prefix, same principle as
    :class:`PlanJudgementPrompt`. A plain query (``task_types=()``)
    reproduces the exact previous static behaviour."""
    _prompt_path = PROMPT_PATH.joinpath("judge.md")

    def __init__(self):
        super().__init__()

    def update(self, task_types: Optional[Sequence[str]] = None) -> str:
        ordered = _canonical_task_types(task_types)
        sections = "\n\n".join(
            load_skill_section(task_type, "code_judge") for task_type in ordered
        )
        return self._update(skill_check_sections=sections)


class SingleTableJudgementResponseGenerationPrompt(Prompt):
    """Single-table variant of :class:`JudgementResponseGenerationPrompt`;
    same skill-conditioned Check-3 injection, same static-system / dynamic-
    user split."""
    _prompt_path = PROMPT_PATH.joinpath("single_table_judge.md")

    def __init__(self):
        super().__init__()

    def update(self, task_types: Optional[Sequence[str]] = None) -> str:
        ordered = _canonical_task_types(task_types)
        sections = "\n\n".join(
            load_skill_section(task_type, "code_judge") for task_type in ordered
        )
        return self._update(skill_check_sections=sections)


class PlanJudgementPrompt(Prompt):
    """Plan-judge instructions for the plan judge panel — the plan payload
    still travels in the USER message (see ``JudgePanel``), never in this
    template, so this stays a pure function of ``task_types`` with no
    per-call data in it.

    Not byte-identical across every call anymore: one ``**Skill Check**``
    section is appended per distinct skill the CURRENT plan proposes, drawn
    from each skill's consolidated ``conf/skills/{task_type}.md`` (see
    ``load_skill_section(..., "plan_judge")``), because the skill-specific
    pitfalls (a timeseries forecast anchored past the data's own range, a
    classification feature already stripped of variance by an earlier filter,
    ...) can't be named precisely by one generic, one-size-fits-all Check 4.
    ``task_types`` is canonicalized (:func:`_canonical_task_types`) before
    rendering, so this is still a byte-identical, cacheable prefix across
    every call that judges the SAME SET of skills, regardless of the order a
    given plan happened to declare them in — a plain plan (``task_types=()``)
    reproduces the exact previous behaviour (no sections appended)."""
    _prompt_path = PROMPT_PATH.joinpath("plan_judge.md")

    def __init__(self):
        super().__init__()

    def update(self, task_types: Optional[Sequence[str]] = None) -> str:
        ordered = _canonical_task_types(task_types)
        sections = "\n\n".join(
            load_skill_section(task_type, "plan_judge") for task_type in ordered
        )
        return self._update(skill_check_sections=sections)


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
        {queries_with_errors}  — formatted pending queries with their errors/feedback
        {pydantic_constraint}  — JSON schema output format instructions
    """
    _prompt_path = PROMPT_PATH.joinpath("pandas_validator_correction.md")

    def update(self, table_schemas: str, queries_with_errors: str, pydantic_constraint: str) -> str:
        return self._update(
            table_schemas=table_schemas,
            queries_with_errors=queries_with_errors,
            pydantic_constraint=pydantic_constraint,
        )


class SQLValidatorCorrectionPrompt(Prompt):
    """
    Prompt fed to the LLM when StatementValidator needs to correct DuckDB SQL queries
    that failed static validation and/or were rejected by the judge.

    Template variables:
        {table_schemas}        — pre-formatted DatasetDescription text for all tables
        {queries_with_errors}  — formatted pending queries with their errors/feedback
        {pydantic_constraint}  — JSON schema output format instructions
    """
    _prompt_path = PROMPT_PATH.joinpath("sql_validator_correction.md")

    def update(self, table_schemas: str, queries_with_errors: str, pydantic_constraint: str) -> str:
        return self._update(
            table_schemas=table_schemas,
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
    ) -> str:
        return self._update(
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
        )


class TableAnalyzerBatchPrompt(Prompt):
    """Single batched-analysis prompt used by ``TableAnalyzer.analyze_batch``."""
    _prompt_path = PROMPT_PATH.joinpath("table_analyzer_batch.md")

    def update(self, aliases: str, languages: str, tables: str) -> str:
        return self._update(aliases=aliases, languages=languages, tables=tables)


class SkillUsageInstructionPrompt(Prompt):
    """Static instruction appended after an injected skill markdown section.

    Carries no template variables — ``conf/prompts/skill_usage_instruction.md``
    has no ``{...}`` placeholders — but is still a ``Prompt`` subclass rather
    than a raw file read so ``GenerationCoordinator`` never needs its own path.
    """
    _prompt_path = PROMPT_PATH.joinpath("skill_usage_instruction.md")

    def update(self) -> str:
        return self._update()


# ---------------------------------------------------------------------------
# Generation-prompt assembly (moved from ``utility/generation_coordinator.py``)
#
# ``build_skill_sections`` is shared by two independent callers — the
# generation prompt (``build_generation_prompt`` below) and
# ``StatementValidator``'s correction prompts (static-error and
# judge-feedback paths) — so a query's skill context is re-injected
# identically on every correction cycle, not just at initial generation.
# That cross-module reuse is exactly why this assembly logic lives here
# rather than inside the generation-orchestration module that happens to
# be its main caller.
# ---------------------------------------------------------------------------

# Instruction appended to the injected skill section. It tells the model to
# use the skill only when the plan actually calls for an ML-style operation,
# and otherwise to fall back to plain pandas (design §6, skill section).
_SKILL_USAGE_INSTRUCTION = SkillUsageInstructionPrompt().update()


def _render_time_context() -> str:
    """Render the "### Time Context" block injected alongside the timeseries skill.

    The real system datetime — a concrete "now" for the model to anchor
    relative-time language in the question ("predict tomorrow", "as of today")
    against — plus the anchoring rule: the data's observed range usually ends
    well before "now", so the forecast target must follow the series' own last
    observed timestamp, not the current date.
    """
    # Date-level granularity on purpose: a full timestamp changes on every
    # call and would invalidate the provider's cached prompt prefix for every
    # token that follows it; the date is stable across a whole day of runs.
    return (
        "### Time Context\n"
        f"- Current date: {datetime.now().date().isoformat()}\n"
        "- The tables' data may end well before this datetime. Anchor the "
        "forecast to the LAST timestamp actually observed in the series "
        "(train on the observed history, forecast the period(s) immediately "
        "after its end) — never to the current datetime when the data ends "
        "earlier than it."
    )


def build_skill_sections(plan: Optional[Any]) -> List[str]:
    """Render one ``### Skill: {task_type}`` section per ML task type in
    ``plan.task_types`` (Pandas plans only; SQL plans and ``None`` yield none).

    Module-level so it has a single source of truth shared by both the
    *generation* prompt (:func:`build_generation_prompt`) and the
    *correction* prompts (``StatementValidator``'s static-error and
    judge-feedback paths). Without this, a query's skill context would only
    ever be injected once, at initial generation — a later correction cycle
    (static-validation retry or judge-feedback rewrite) would have no idea
    the query was ever bound to a skill, and the LLM would silently drop it
    (e.g. reach for `sklearn` instead of `TabPFNRegressor`) while "fixing"
    something unrelated.

    When ``timeseries`` is among the plan's task types, a ``### Time Context``
    block is appended after that skill's section (see :func:`_render_time_context`).
    """
    task_types = getattr(plan, "task_types", None) or []
    sections = [
        f"### Skill: {task_type}\n"
        f"{load_skill_section(task_type, 'generation')}\n\n"
        f"{_SKILL_USAGE_INSTRUCTION}"
        for task_type in task_types
    ]
    if "timeseries" in task_types:
        sections.append(_render_time_context())
    return sections


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
    task_types = getattr(plan, "task_types", None)
    if task_types:
        lines.append(f"Task types: {', '.join(task_types)}")
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
        if step.columns_role:
            lines.append(
                f"   - columns_role: {json.dumps(step.columns_role, ensure_ascii=False)}"
            )
        if step.params:
            lines.append(
                f"   - params: {json.dumps(step.params, ensure_ascii=False)}"
            )
    return "\n".join(lines)


def _render_column_statistics(stats: Sequence[TableStats]) -> str:
    """Render the per-table column-statistics block.

    Mirrors the rendering style of
    :meth:`orqa.agent.agents.QueryPlanner.QueryPlanner._render_statistics` so the plan
    and generation prompts present statistics consistently, without coupling the
    two components (QueryPlanner keeps its own copy untouched).
    """
    if not stats:
        return "(no column statistics available)"
    payload = []
    for table in stats:
        payload.append(
            {
                "alias": table.alias,
                "num_rows": table.num_rows,
                "columns": [
                    {
                        "column": c.column,
                        "dtype": c.dtype,
                        "cardinality": c.cardinality,
                        "null_ratio": round(c.null_ratio, 4),
                        "numeric_min": c.numeric_min,
                        "numeric_max": c.numeric_max,
                        "numeric_mean": c.numeric_mean,
                        "top_values": c.top_values,
                    }
                    for c in table.columns
                ],
            }
        )
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def build_generation_prompt(
    base_prompt: str,
    plan: Optional[QueryPlan] = None,
    stats: Optional[Sequence[TableStats]] = None,
) -> str:
    """Build the skill-injected, plan-grounded generation prompt.

    Assembles the generation prompt from the caller-provided ``base_prompt``
    plus, in order (run-stable sections first, the per-call plan last, so the
    run's several generation calls share the longest possible cached prefix):

    * ``### Column Statistics`` — the per-table column-statistics block from
      ``stats``, rendered only when statistics are supplied.
    * ``### Skill: {task_type}`` — one section per ML task type in
      ``plan.task_types`` (Pandas plans only; SQL plans never carry
      ``task_types``), each injecting that task type's self-contained
      markdown from ``conf/skills/{task_type}.md`` together with usage
      guidance. Injection is unconditional: whichever task types the plan
      calls for get their markdown injected, with no further gating. When
      ``timeseries`` is among them, a ``### Time Context`` block follows
      (see :func:`build_skill_sections`).
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
    # after the first. The skill sections (drawn from a small fixed set) come
    # next, and the Structured Plan — the only part unique to each call — is
    # appended LAST so it never invalidates the shared prefix.
    sections: List[str] = [base_prompt]

    if stats:
        sections.append(
            f"### Column Statistics\n{_render_column_statistics(stats)}"
        )

    task_types = getattr(plan, "task_types", None) or []
    sections.extend(build_skill_sections(plan))

    if plan is not None:
        sections.append(f"### Structured Plan\n{_render_plan_steps(plan)}")

    prompt = "\n\n".join(sections)

    if task_types:
        logger.info(
            "Skill(s) %s injected into generation prompt:\n%s",
            ", ".join(task_types), prompt,
        )

    return prompt
