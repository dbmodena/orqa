import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union
from uuid import uuid4


from orqa.agent.llm_client.LLMClientStructured import LLMClientStructured
from orqa.agent.planning.provenance import plan_derived_columns
from orqa.agent.prompting import GenerationEnrichmentPrompt, build_generation_prompt
from orqa.agent.structured_outputs import Query, QueryPlan
from orqa.agent.prompting.models import PandasQueryPlan, SQLQueryPlan, TableStats
from orqa.utils.pipeline_logger import PipelineLogger


# ── Alias substitution ────────────────────────────────────────────────────────
# Centralized alias substitution logic.
#
# Extracted from the duplicated `clean_table_names` static methods in
# StatementClient and StatementValidator. Provides word-boundary-aware regex
# substitution of original dataset IDs (e.g. "qvir-knu3") with canonical
# aliases (e.g. "Table_0").

class AliasSubstitution:
    """Centralized alias substitution logic.

    Replaces original dataset identifiers with canonical aliases using
    pre-compiled word-boundary-aware regex patterns that handle both
    dash-separated and underscore-separated variants.
    """

    def __init__(self, aliases: dict[str, str]) -> None:
        """
        Args:
            aliases: Mapping of canonical alias → original dataset ID
                     e.g. {"Table_0": "qvir-knu3", "Table_1": "abc-def"}
        """
        # Invert: original_name → alias
        self._inverted: dict[str, str] = {v: k for k, v in aliases.items()}

        # Pre-compile word-boundary regex patterns for each original name
        # and its underscore variant.
        # Pattern: not preceded by a dot or word char, not followed by a word char
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for original_name, alias in self._inverted.items():
            underscore_variant = original_name.replace("-", "_")
            for name in {original_name, underscore_variant}:
                pattern = re.compile(
                    r"(?<![.\w])" + re.escape(name) + r"(?!\w)"
                )
                self._patterns.append((pattern, alias))

        # Pre-compile verification patterns (same logic, just for detection)
        self._verify_patterns: list[tuple[re.Pattern[str], str]] = []
        for original_name in self._inverted:
            underscore_variant = original_name.replace("-", "_")
            for name in {original_name, underscore_variant}:
                pattern = re.compile(
                    r"(?<![.\w])" + re.escape(name) + r"(?!\w)"
                )
                self._verify_patterns.append((pattern, name))

    def substitute(self, text: str) -> str:
        """Replace all original table names (dash and underscore variants)
        with canonical aliases using word-boundary-aware regex.

        Args:
            text: Input text potentially containing original table names.

        Returns:
            Text with all original names replaced by their canonical aliases.
        """
        result = text
        for pattern, alias in self._patterns:
            result = pattern.sub(alias, result)
        return result

    def substitute_query(self, query: dict) -> dict:
        """Apply substitution to code, question, motivation, and tables[].name fields.

        Args:
            query: A query dict with fields like code, question, motivation, tables.

        Returns:
            A new query dict with substitutions applied to relevant fields.
        """
        result = dict(query)

        # Substitute text fields
        for field in ("code", "question", "motivation"):
            if field in result and isinstance(result[field], str):
                result[field] = self.substitute(result[field])

        # Substitute table names in the tables list
        if "tables" in result and isinstance(result["tables"], list):
            new_tables = []
            for table in result["tables"]:
                if isinstance(table, dict):
                    new_table = dict(table)
                    if "name" in new_table and isinstance(new_table["name"], str):
                        new_table["name"] = self.substitute(new_table["name"])
                    new_tables.append(new_table)
                else:
                    new_tables.append(table)
            result["tables"] = new_tables

        return result

    def verify(self, text: str) -> list[str]:
        """Return list of original names still present at word boundaries.

        Args:
            text: Text to check for remaining original table names.

        Returns:
            List of original names (or their underscore variants) found in text.
            Should be empty after successful substitution.
        """
        found: list[str] = []
        for pattern, name in self._verify_patterns:
            if pattern.search(text):
                found.append(name)
        return found


logger = logging.getLogger(__name__)

# Output budget of a generation call and of a correction call (see
# QueryPlanner.GENERATION_MAX_TOKENS for the mechanism: a reasoning model's
# hidden thinking counts against max_tokens, and a cut answer surfaces as a
# "Pydantic error"). Measured on the code stage
# (experiments/structured_stability, 2026-11-03): the largest answers were
# 4548 tokens for generation and 5158 for a correction against the former
# 8000 cap — 1.7–2x margin on those groups, none on the first longer payload.
# 32000 is the explicit budget every structured call now carries; the client
# escalates further on a cut.
GENERATION_MAX_TOKENS = 32000


class LLMClientStatementGenerator(LLMClientStructured):
    """
    LiteLLM client with YAML configuration and structured output support.

    Responsibility: generate the *initial* set of queries from a prompt.
    Static validation and correction are now handled by LLMStatementValidator
    (see StatementValidator.py).  This class only retries on JSON / Pydantic
    parse errors — it does NOT run PandasValidator / SQLValidator internally.
    """

    # The system message's first line (the schema follows it).
    preamble = "You are an expert Data Engineer."
    # Code is a long output: under native JSON-schema decoding the constrained
    # decoder looped the SELECT list of a 22-step plan's query until the
    # repetition guard stopped it — the planner's failure mode again — so the
    # two clients that return code stay prompt-only (the judges' short
    # verdicts keep native decoding on).
    native_schema = False

    def __init__(self, config_path: Path):
        super().__init__(config_path, "querying")
        self.fallback_response_model = self._load_pydantic_response_model("fallback_querying")
        self._log = PipelineLogger()


    def enrich_prompt_with_table_analysis(
        self,
        prompt: str,
        table_analyses: list[dict],
    ) -> str:
        """Append the table-analysis enrichment block to ``prompt``.

        Called ONCE per run on the run-stable base prompt (see
        ``StatementOrchestrator._run`` in ``agent.py``) — before the per-plan generation loop — so the
        enrichment sits in the shared, cacheable prefix of every generation
        call rather than trailing each call's unique plan section.
        """
        enrichment = GenerationEnrichmentPrompt().update(
            table_analysis=json.dumps(
                {"tables": table_analyses}, indent=2, ensure_ascii=False, default=str
            ),
            planning_section="",
        )
        return f"{prompt}\n\n{enrichment}"

    # ------------------------------------------------------------------
    # JSON repair utilities
    # ------------------------------------------------------------------

    def _repair_json(self, content: str) -> dict:

        if not isinstance(content, str):
            raise json.JSONDecodeError("content is not a string", str(content), 0)

        # Stage 1 – fast path
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Stage 2 – fix triple-quoted strings
        patched = self._fix_triple_quotes(content)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

        # Stage 3 – escape literal control characters inside JSON strings
        escaped = self._escape_literal_controls(patched)
        try:
            return json.loads(escaped)
        except json.JSONDecodeError:
            pass

        # Stage 4 – structural fixes (trailing commas, unclosed delimiters)
        structural = self._fix_structural_issues(escaped)
        try:
            return json.loads(structural)
        except json.JSONDecodeError:
            pass

        # Final raise so the caller receives a meaningful JSONDecodeError
        return json.loads(structural)

    def _fix_structural_issues(self, content: str) -> str:
        """
        Fix common structural JSON problems produced by LLMs:
        - Trailing commas before } or ]
        - Unterminated string at end of document
        - Unclosed braces / brackets (truncated LLM output)
        """
        # 1. Remove trailing commas before a closing delimiter
        fixed = re.sub(r',\s*(?=[}\]])', '', content)

        # 2. Close any unterminated string at the very end of the document.
        unescaped_quotes = re.findall(r'(?<!\\)"', fixed)
        if len(unescaped_quotes) % 2 != 0:
            fixed = fixed.rstrip() + '"'

        # 3. Count open braces/brackets and close any that were never closed
        closer_map = {'{': '}', '[': ']'}
        stack: list[str] = []
        in_str = False
        i = 0
        while i < len(fixed):
            ch = fixed[i]
            if in_str:
                if ch == '\\':
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in ('{', '['):
                    stack.append(closer_map[ch])
                elif ch in ('}', ']'):
                    if stack and stack[-1] == ch:
                        stack.pop()
            i += 1

        fixed = fixed.rstrip()
        for closer in reversed(stack):
            fixed += closer

        return fixed

    def _fix_triple_quotes(self, content: str) -> str:
        """Replace \"\"\"...\"\"\" with a properly escaped single-quoted JSON string."""
        pattern = r'"""[\s\S]*?"""'
        matches = list(re.finditer(pattern, content))
        offset = 0
        result = content
        for match in matches:
            original = match.group(0)
            inner = original[3:-3].strip()
            escaped = (
                inner
                .replace('\\', '\\\\')   # must be first
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '')
                .replace('\t', ' ')
            )
            replacement = f'"{escaped}"'
            start = match.start() + offset
            end = match.end() + offset
            result = result[:start] + replacement + result[end:]
            offset += len(replacement) - len(original)
        return result

    def _escape_literal_controls(self, content: str) -> str:
        """
        Scan JSON text and escape any literal control characters that appear
        inside string values (i.e. between unescaped double-quotes).
        """
        result = []
        in_string = False
        i = 0
        while i < len(content):
            ch = content[i]
            if in_string:
                if ch == '\\':
                    result.append(ch)
                    i += 1
                    if i < len(content):
                        result.append(content[i])
                        i += 1
                    continue
                elif ch == '"':
                    in_string = False
                    result.append(ch)
                elif ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                else:
                    result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                    result.append(ch)
                else:
                    result.append(ch)
            i += 1
        return ''.join(result)

    # Keep the old name as a thin wrapper so nothing external breaks.

    # ------------------------------------------------------------------
    # Table-alias validation
    # ------------------------------------------------------------------

    def _find_missing_tables(self, query_code: str, all_aliases: list[str]) -> list[str]:
        """
        Return the aliases from *all_aliases* that are NOT referenced in
        *query_code*.  Matching is word-boundary aware so e.g. "Table_0"
        inside "Table_00" is not counted as a hit.
        """
        return [
            alias for alias in all_aliases
            if not re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', query_code)
        ]

    def _finalize_query_set(self, queries: list[dict]) -> dict:
        """Validate every generated query as a :class:`Query`, return a query-set dict.

        This is the final gate before ``complete()`` hands its result back to the
        caller: each dict in ``queries`` (already schema-checked once at parse
        time via ``self.response_model.model_validate``, but possibly mutated
        since — by ``clean_query_set``, the table-analysis merge, or the
        table-alias retry loop) is re-validated as a :class:`Query`. A query
        that fails re-validation is dropped (logged, not raised) rather than
        failing the whole batch — one malformed query shouldn't sink every
        other query in the same run.

        ``Query`` itself no longer carries the planner-owned question-level
        metadata (``query_plan``, ``question_keywords``,
        ``translated_question_keywords``, ``translated_question``,
        ``detected_language``, ``topic``, ``story`` — see
        ``prompting.models.SQLQueryPlan``/``PandasQueryPlan``), which
        ``complete()`` merges onto each query dict from the plan before this
        method runs. Re-validating strictly through ``Query`` would silently
        drop those fields, so each validated query's canonical fields are
        merged back onto the ORIGINAL dict (which still carries them) rather
        than the other way around.
        """
        validated: list[dict] = []
        for q in queries:
            try:
                query_obj = Query.model_validate(q)
            except ValidationError as exc:
                logger.warning(
                    "Dropping a generated query that failed Query validation "
                    "in the final gate (client_id=%r): %s",
                    q.get("client_id") if isinstance(q, dict) else None,
                    exc,
                )
                continue
            merged = dict(q) if isinstance(q, dict) else {}
            merged.update(query_obj.model_dump())
            validated.append(merged)
        return {"queries": validated}

    def _enforce_single_query(self, queries: list[dict]) -> list[dict]:
        """Keep at most one query — every generation call is bound to a single plan.

        Every call to :meth:`complete` is scoped to exactly one query plan
        (``precomputed_plan`` — see ``QueryPlanner.plan_batch``) and the prompt
        asks for exactly one query in return. If the model nonetheless returns
        more than one, keep only the first — the plan this call was given
        only applies to one query, so any extras aren't traceable to a
        plan of their own and would silently break the one-plan-to-one-query
        contract downstream.
        """
        if len(queries) <= 1:
            return queries
        logger.warning(
            "Generation call bound to a single plan returned %d queries; "
            "keeping only the first and dropping the rest.",
            len(queries),
        )
        return queries[:1]

    def _partition_queries(
        self, queries: list[dict], all_aliases: list[str]
    ) -> tuple[list[dict], list[dict]]:
        """
        Split *queries* into (good, bad) based on whether every alias in
        *all_aliases* appears in each query's ``code`` field.

        Returns:
            good: queries that reference all aliases.
            bad:  queries that are missing one or more aliases.
        """
        good, bad = [], []
        for q in queries:
            missing = self._find_missing_tables(q.get("code", ""), all_aliases)
            if missing:
                bad.append({**q, "_missing_tables": missing})
            else:
                good.append(q)
        return good, bad

    def _build_unused_tables_feedback(
        self, bad_queries: list[dict], all_aliases: list[str]
    ) -> str:
        """Build a feedback message listing each offending query and its missing tables."""
        lines = [
            "Some generated queries do not reference all required tables.",
            "Rewrite ONLY the queries listed below so that every table alias is used.",
            f"Required aliases: {', '.join(all_aliases)}\n",
        ]
        for i, q in enumerate(bad_queries, 1):
            missing = q.get("_missing_tables", [])
            # Show a short excerpt of the code so the LLM can identify the query
            code_preview = q.get("code", "")[:120].replace("\n", " ")
            lines.append(
                f"  Query {i} — missing {', '.join(sorted(missing))}:\n"
                f"    code preview: {code_preview!r}"
            )
        lines.append(
            "\nReturn ONLY the corrected queries (same JSON schema as before)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        dataframes,
        table_names,
        *,
        typology: str = "SQL",
        involved_cols=None,
        matches=None,
        metadata=None,
        languages=None,
        precomputed_plan: dict,
    ) -> Any:
        """
        Generate the initial set of queries from *prompt*, then validate that
        every query references all table aliases.  Queries that pass are kept;
        only the failing ones are sent back for correction (up to
        ``MAX_TABLE_ALIAS_RETRIES`` extra rounds).  After all retries, any
        still-failing queries are silently dropped and only the good ones are
        returned.

        The LLM call itself is the shared budget-aware structured loop
        (``LLMClientStructured.complete``: explicit ``GENERATION_MAX_TOKENS``
        budget with escalation on a cut answer, shape coercion, one repair
        call, the repetition guard); only the post-processing — alias
        substitution, the plan fields, the single-query rule, alias-coverage
        retries, the final gate — lives here.

        Args:
            precomputed_plan: The specific plan (dict form) this generation
                call is bound to — one plan out of ``QueryPlanner.plan_batch``'s
                several independent plans. It is already injected into
                ``prompt`` upstream (by ``GenerationCoordinator``), so no plan
                section is added here and the model sees exactly one plan.
                Its own ``tables`` (name/reason/columns_involved/description/
                keywords, already judged by the plan panel) is what gets
                stamped onto every generated query below — table-level
                analysis is no longer merged in here separately, since the
                generation LLM no longer produces a ``tables`` field at all.
        """
        MAX_TABLE_ALIAS_RETRIES = 3

        usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        all_errors: list[str] = []
        all_aliases: list[str] = list(table_names.keys())  # e.g. ["Table_0", "Table_1"]

        # Reuse the caller's already-computed plan verbatim — this method never
        # runs its own query-planner LLM call. Both the plan AND the
        # table-analysis enrichment are already injected into `prompt`
        # upstream (GenerationCoordinator adds the plan; the orchestrator
        # enriches the run-stable base prompt once via
        # ``enrich_prompt_with_table_analysis``), so nothing is appended here
        # and the run's generation calls keep their shared prompt prefix.
        plan = precomputed_plan
        self._log.query_plan(plan)

        # Copy the planner-owned question-level metadata onto every generated
        # query. These fields are no longer part of the Query schema (the
        # generation LLM never produces them) — they were already decided
        # once, during planning, and every query bound to this plan/call
        # shares that same metadata. `tables` (name/reason/columns_involved/
        # description/keywords) is one of them: the plan's own `tables[]`
        # (List[Table]) was already judged by the plan panel
        # (PlanJudgment.table_check) and checked for full alias coverage by
        # QueryPlanner.validate_plan, so it is simply copied here rather than
        # asked of the generation LLM a second time.
        plan_fields = {
            "query_plan": plan.get("query_plan", ""),
            "question_keywords": plan.get("question_keywords", []),
            "translated_question_keywords": plan.get("translated_question_keywords", []),
            "translated_question": plan.get("translated_question", ""),
            "detected_language": plan.get("detected_language", ""),
            "topic": plan.get("topic", ""),
            "story": plan.get("story", ""),
            "tables": plan.get("tables", []),
            # The slot's target tier, judged on the question upstream and
            # pinned onto the plan — not estimated from the plan, and not
            # part of the generation LLM's own output.
            "difficulty": plan.get("difficulty", ""),
            # Plan-declared result contract, judged by the plan panel
            # (PlanJudgment.expected_result_check) and mechanically enforced
            # against the executed result by the validators
            # (QueryValidator._check_expected_result_type).
            "expected_result_type": plan.get("expected_result_type", ""),
            "expected_result_description": plan.get("expected_result_description", ""),
            # Columns the plan's steps declare they CREATE (see each step's
            # `produces`). Resolved and checked at planning time, so the
            # validator can tell a legitimate derived column from a
            # hallucinated one by exact lookup instead of pattern-matching
            # the generated code — see QueryValidator.prefilter_dataframes.
            "derived_columns": sorted(plan_derived_columns(plan)),
        }

        def ask(user_prompt: str) -> list[dict]:
            """One structured generation call -> the cleaned, plan-stamped
            queries it returned ([] when every attempt failed)."""
            result, usage = LLMClientStructured.complete(
                self, user_prompt, root_key="queries", max_tokens=GENERATION_MAX_TOKENS
            )
            for key in usage_total:
                usage_total[key] += (usage or {}).get(key, 0)
            if not result:
                return []
            cleaned = self.clean_query_set(result, table_names)
            queries = cleaned.get("queries", []) or []
            for q in queries:
                q.update(plan_fields)
            return queries

        # ── Phase 1: initial generation ──────────────────────────────────────
        queries = ask(prompt)
        if not queries:
            all_errors.append("Generation produced no valid query set after every attempt.")
            return (
                self._finalize_query_set([]),
                usage_total,
                all_errors,
                self.config["model"],
                plan,
            )

        # ── Phase 2: table-alias validation with targeted retry loop ───────────
        good_queries, bad_queries = self._partition_queries(
            self._enforce_single_query(queries), all_aliases
        )

        for alias_attempt in range(MAX_TABLE_ALIAS_RETRIES):
            if not bad_queries:
                break  # Nothing left to fix

            feedback = self._build_unused_tables_feedback(bad_queries, all_aliases)
            # Strip the internal _missing_tables key before sending to LLM
            bad_queries_clean = [
                {k: v for k, v in q.items() if k != "_missing_tables"}
                for q in bad_queries
            ]
            all_errors.append(
                f"Table-alias check (attempt {alias_attempt + 1}): "
                f"{len(bad_queries)} query/queries missing aliases."
            )
            retried = ask(
                f"{prompt}\n\n"
                f"Previous output (with issues):\n{json.dumps({'queries': bad_queries_clean}, default=str)}\n\n"
                f"{feedback}"
            )
            if not retried:
                all_errors.append(f"Table-alias retry {alias_attempt + 1} produced no valid query set.")
                continue
            # Re-partition: some may now be fixed, others still broken
            newly_good, bad_queries = self._partition_queries(retried, all_aliases)
            good_queries.extend(self._enforce_single_query(newly_good))

        # After all alias retries, silently drop any still-failing queries
        if bad_queries:
            all_errors.append(
                f"Dropping {len(bad_queries)} query/queries that still fail alias check "
                f"after {MAX_TABLE_ALIAS_RETRIES} retries."
            )

        return (
            self._finalize_query_set(good_queries),
            usage_total,
            all_errors,
            self.config["model"],
            plan,
        )

    def _normalize_response(self, json_data: Any, root_key: str | None = "queries") -> Any:
        if root_key is None:
            return json_data
        if isinstance(json_data, list):
            return {root_key: json_data}
        if isinstance(json_data, dict):
            if root_key not in json_data:
                if len(json_data) > 0:
                    return {root_key: [json_data]}
            return json_data
        return {root_key: [json_data]}


    @staticmethod
    def clean_query_set(query_set: dict, aliases: dict) -> dict:
        """Apply alias substitution to all queries in a query set.

        Uses AliasSubstitution to replace original table names with canonical
        aliases in the code field of each query, as well as question,
        motivation, and tables[].name fields.
        """
        sub = AliasSubstitution(aliases)
        cleaned = {**query_set, "queries": []}
        for query in query_set.get("queries", []):
            cleaned["queries"].append(sub.substitute_query(query))
        return cleaned


# ── Generation coordinator ────────────────────────────────────────────────────
# Generation orchestration and the opaque ``client_id`` bookkeeping key.
#
# :class:`GenerationCoordinator` owns the *initial generation* phase of the pipeline. Per
# the design (see the component table in the requirements, Requirement 16), it:
#
# * Builds the final generation prompt by delegating to
#   :func:`orqa.agent.prompting.build_generation_prompt` (plan-grounded —
#   task 8.2), calls the injected ``generate_fn`` with it, and
# * Assigns every generated query a unique, opaque ``client_id`` used purely as
#   an **internal** bookkeeping key (``plan_by_client_id``, judge-result maps,
#   ...) — never something the LLM is asked to declare or echo back. Every
#   downstream match (correction, judging) is positional, since generation,
#   correction, and judging each process queries one at a time or in strict
#   input order; nothing ever reads an id back out of a model response.
#
# This module implements **tasks 8.1 and 8.2**: the ``client_id`` assignment
# (8.1) plus orchestrating the plan-grounded prompt build and the generation
# call itself (8.2). The prompt-assembly logic (rendering plan steps and
# column statistics) lives in :mod:`orqa.agent.prompting.prompts` — it is
# shared with ``StatementValidator``'s correction prompts, so it belongs with
# the rest of the prompting module rather than here.
#
# Why assign ids programmatically rather than trust the model to mint them: the
# ``client_id`` is the key the validator and judge use to map results back to
# the source query without trusting positional order *within this module's own
# bookkeeping* — but that mapping is built and read entirely by our own code
# (see ``plan_by_client_id`` and ``all_judgments_by_client_id`` in ``agent.py``),
# never by asking the model to produce or preserve an id in
# its output.

logger = logging.getLogger(__name__)

QueryPlan = Union[SQLQueryPlan, PandasQueryPlan]

# Length of an assigned client_id token. Eight hex characters of a uuid4 give an
# opaque, collision-resistant, human-copyable short id.
_CLIENT_ID_LENGTH = 8


class GenerationCoordinator:
    """Orchestrates generation and enforces the ``client_id`` contract.

    Args:
        id_factory: Optional zero-argument callable returning a fresh id string.
            Injected for deterministic testing; defaults to an 8-char uuid4 hex
            token. The generator guarantees uniqueness across a single
            assignment regardless of the factory (regenerating on collision).
    """

    def __init__(self, id_factory: Optional[Callable[[], str]] = None):
        self._id_factory = id_factory or self._default_id_factory

    # ------------------------------------------------------------------
    # client_id assignment (Requirement 16.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_id_factory() -> str:
        """Return a fresh opaque short id (8 hex chars of a uuid4)."""
        return uuid4().hex[:_CLIENT_ID_LENGTH]

    def _new_client_id(self, taken: set) -> str:
        """Return a fresh id not already present in ``taken``.

        Regenerates on the (astronomically unlikely) event of a collision so the
        assigned ids are always unique within a single query set. Falls back to a
        full uuid4 hex if the factory keeps colliding (e.g. a constant test stub).
        """
        for _ in range(8):
            candidate = self._id_factory()
            if candidate and candidate not in taken:
                return candidate
        # Factory is degenerate (constant/empty) — guarantee uniqueness ourselves.
        return uuid4().hex

    def assign_client_ids(self, queries: List[dict]) -> List[dict]:
        """Assign a unique, non-empty ``client_id`` to each query.

        Each query dict is given a fresh opaque ``client_id`` (Requirement 16.1).
        Any ``client_id`` already present on an incoming query is overwritten so
        the generator remains the single source of truth for the identifier —
        the model is never trusted to mint one, and never asked to.

        Args:
            queries: The generated query dicts (each mutated in place).

        Returns:
            The same list, with every query carrying a unique ``client_id``.
        """
        assigned: set = set()
        for query in queries:
            if not isinstance(query, dict):
                logger.warning("Skipping client_id assignment for non-dict query: %r", query)
                continue
            client_id = self._new_client_id(assigned)
            query["client_id"] = client_id
            assigned.add(client_id)
        return queries

    # ------------------------------------------------------------------
    # Generation orchestration
    # ------------------------------------------------------------------

    def generate(
        self,
        base_prompt: str,
        generate_fn: Callable[[str], Any],
        plan: Optional[QueryPlan] = None,
        stats: Optional[Sequence[TableStats]] = None,
    ) -> tuple[dict, dict, list, str]:
        """Generate queries and assign each a unique ``client_id``.

        Args:
            base_prompt: The base generation prompt.
            generate_fn: A callable that takes the fully-built prompt and returns
                the model's query set. It may return either a bare
                ``{"queries": [...]}`` dict, or the richer
                ``(queryset_dict, usage, errors, model)`` tuple produced by
                :class:`LLMClientStatementGenerator.complete`. Injected so the LLM
                client can be mocked in tests.
            plan: Optional structured plan whose ordered steps ground generation
                (Requirement 20.1). Passed through to
                :func:`orqa.agent.prompting.build_generation_prompt`.
            stats: Optional per-table column statistics rendered into the prompt
                (Requirement 20.2).

        Returns:
            ``(query_set, usage, errors, model)`` where every query in
            ``query_set["queries"]`` carries a unique, non-empty ``client_id``.
        """
        prompt = build_generation_prompt(base_prompt, plan, stats)
        raw = generate_fn(prompt)
        query_set, usage, errors, model = self._normalize_result(raw)

        queries = query_set.get("queries", [])
        if not isinstance(queries, list):
            logger.warning("Generation returned a non-list 'queries'; coercing to empty list.")
            queries = []
        query_set["queries"] = self.assign_client_ids(queries)
        return query_set, usage, errors, model

    @staticmethod
    def _normalize_result(raw: Any) -> tuple[dict, dict, list, str]:
        """Normalise a generate_fn return into ``(query_set, usage, errors, model)``.

        Accepts either the 4-tuple returned by
        :meth:`LLMClientStatementGenerator.complete` or a bare query-set dict.
        """
        usage: dict = {}
        errors: list = []
        model: str = ""

        if isinstance(raw, tuple):
            query_set = raw[0] if len(raw) > 0 else {}
            usage = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
            errors = raw[2] if len(raw) > 2 and isinstance(raw[2], list) else []
            model = raw[3] if len(raw) > 3 and isinstance(raw[3], str) else ""
        else:
            query_set = raw

        if not isinstance(query_set, dict):
            logger.warning("Generation returned a non-dict query set; coercing to empty.")
            query_set = {"queries": []}
        return query_set, usage, errors, model
