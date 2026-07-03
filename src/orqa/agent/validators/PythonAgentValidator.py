"""Python coding-agent validator (revised ``PandasValidator``).

This is the general Python-code validator described in design §4. Where the
original :class:`~orqa.agent.validators.PandasValidator.PandasValidator` was
specialised for concat/merge relational code, ``PythonAgentValidator`` validates
arbitrary Python data code produced across task types (classification,
regression, timeseries, imputation, causal) while retaining the relational
structural checks for join/union task types.

Task 6.1 (this file) implements:
  * :class:`ValidationOutcome` — the four-category validation result contract.
  * :data:`BASE_ALLOWLIST` — the always-allowed import modules.
  * :func:`ast_extract_imports` — AST-based top-level import extraction that
    replaces the old substring ``UNAUTHORIZED_COMMANDS`` matching.
  * disallowed-import classification: code importing a module outside
    ``BASE_ALLOWLIST | allowlist`` is a ``quality_rejection``.

Later tasks fill in the stubs below:
  * 6.2 — task-type-aware structural checks (connectivity / cartesian product).
  * 6.3 — execution sandbox with ML/default time budgets and row caps.
  * 6.4 — outcome classification / correction routing.

Requirements: 13.1, 13.2, 13.3.
"""

from __future__ import annotations

import ast
import importlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import polars as pl
from pydantic import BaseModel

from .PandasValidator import PandasValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import allowlist
# ---------------------------------------------------------------------------

# Modules that are ALWAYS permitted regardless of skill selection. This is the
# base of the effective allowlist; the skill registry's ``import_allowlist`` is
# unioned on top (e.g. ``tabpfn_client`` when the TabPFN skill is selected).
#
# The set covers the core data-analysis stack (pandas/polars/numpy — ``pd``/
# ``pl`` are pre-injected into the sandbox namespace but explicit imports are
# tolerated), plus the safe standard-library modules generated data code
# commonly relies on. Deliberately EXCLUDED: ``os``, ``sys``, ``subprocess``,
# ``socket``, ``shutil``, ``pathlib`` and other side-effecting/IO modules — the
# sandbox must not reach the filesystem, network, or interpreter internals.
BASE_ALLOWLIST: Set[str] = {
    # Core data stack (aliases included defensively; AST yields real module names).
    "pandas",
    "polars",
    "numpy",
    "pd",
    "pl",
    "np",
    # Math / statistics.
    "math",
    "statistics",
    "decimal",
    "fractions",
    "random",
    # Safe stdlib basics.
    "datetime",
    "collections",
    "itertools",
    "functools",
    "operator",
    "re",
    "json",
    "string",
    "warnings",
    "typing",
}


# ---------------------------------------------------------------------------
# Validation outcome contract
# ---------------------------------------------------------------------------

class ValidationOutcome(BaseModel):
    """The result of validating a single generated query.

    Every validation resolves to exactly one of four categories (Req 13.1):

    * ``ok`` — code passed all checks and executed successfully.
    * ``execution_failure`` — a fixable runtime/syntax error (syntax, NameError,
      KeyError, dtype, timeout). Only this category loops back for correction.
    * ``quality_rejection`` — a structural problem (disallowed import, disjoint
      query, cartesian product). Not routed for code-correction retry.
    * ``budget_exceeded`` — the overall wall-clock/token ceiling was hit.
    """

    query_id: str  # client_id (opaque, echoed)
    passed: bool
    category: str  # one of: ok | execution_failure | quality_rejection | budget_exceeded
    error: str = ""
    row_count: Optional[int] = None
    elapsed_ms: Optional[float] = None

    # -- convenience constructors -------------------------------------------

    @classmethod
    def ok(
        cls,
        query_id: str,
        row_count: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
    ) -> "ValidationOutcome":
        return cls(
            query_id=query_id,
            passed=True,
            category="ok",
            row_count=row_count,
            elapsed_ms=elapsed_ms,
        )

    @classmethod
    def execution_failure(
        cls,
        query_id: str,
        error: str,
        elapsed_ms: Optional[float] = None,
    ) -> "ValidationOutcome":
        return cls(
            query_id=query_id,
            passed=False,
            category="execution_failure",
            error=error,
            elapsed_ms=elapsed_ms,
        )

    @classmethod
    def quality_rejection(cls, query_id: str, error: str) -> "ValidationOutcome":
        return cls(
            query_id=query_id,
            passed=False,
            category="quality_rejection",
            error=error,
        )

    @classmethod
    def budget_exceeded(cls, query_id: str, error: str = "") -> "ValidationOutcome":
        return cls(
            query_id=query_id,
            passed=False,
            category="budget_exceeded",
            error=error,
        )


# Valid category literals, exposed for validation/testing.
VALIDATION_CATEGORIES = (
    "ok",
    "execution_failure",
    "quality_rejection",
    "budget_exceeded",
)


# ---------------------------------------------------------------------------
# AST-based import extraction (replaces substring UNAUTHORIZED_COMMANDS)
# ---------------------------------------------------------------------------

def ast_extract_imports(code: str) -> Set[str]:
    """Return the set of TOP-LEVEL module names imported by ``code``.

    Handles every import form via the ``ast`` module rather than substring
    matching:

    * ``import x``          -> ``x``
    * ``import x.y``        -> ``x``           (top-level package only)
    * ``import x as z``     -> ``x``           (alias ignored; real module kept)
    * ``from x import y``   -> ``x``
    * ``from x.y import z`` -> ``x``           (top-level package only)

    Relative imports (``from . import y`` / ``from .pkg import z``) have no
    top-level absolute module name and are skipped.

    Raises:
        SyntaxError: when ``code`` cannot be parsed. Callers that want to treat
            unparseable code as an execution failure should catch this.
    """
    tree = ast.parse(code)
    modules: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # ``level > 0`` marks a relative import (no absolute top-level name).
            if node.level and node.level > 0:
                continue
            if node.module:
                modules.add(node.module.split(".")[0])

    return modules


def find_disallowed_imports(code: str, allowlist: Iterable[str]) -> List[str]:
    """Return the sorted top-level modules imported by ``code`` that are not
    permitted by ``BASE_ALLOWLIST | allowlist``.

    ``allowlist`` is the skill-derived import allowlist (e.g. ``{"tabpfn_client"}``
    when the TabPFN skill is selected). An empty list means every import is
    allowed.

    Raises:
        SyntaxError: propagated from :func:`ast_extract_imports` when ``code``
            cannot be parsed.
    """
    effective = BASE_ALLOWLIST | set(allowlist or ())
    imports = ast_extract_imports(code)
    return sorted(imports - effective)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

# Task types that trigger the relational structural checks (task 6.2).
RELATIONAL_TASK_TYPES = {"join", "union"}

# Task types treated as machine-learning for sandbox time budgeting (task 6.3).
ML_TASK_TYPES = {"classification", "regression", "timeseries", "imputation", "causal"}


# ---------------------------------------------------------------------------
# Execution-sandbox budgets and result contract (task 6.3)
# ---------------------------------------------------------------------------

# Wall-clock time budgets (seconds). ML task types get a larger budget because
# model fitting/inference (TabPFN, imputation, causal, timeseries) is far slower
# than relational/transform code. Values follow design §4 ("Execution sandbox").
DEFAULT_TIME_BUDGET = 20   # seconds — relational / filter / transform code
ML_TIME_BUDGET = 120       # seconds — classification/regression/timeseries/imputation/causal

# Hard cap on the number of rows returned from the sandbox for ALL task types
# (Req 14.5). The result frame is truncated to at most this many rows. This is a
# module-level constant (NOT a class attribute) so it does not shadow
# ``PandasValidator.MAX_RESULT_ROWS`` (a separate 5M-row memory guard on the
# relational execution path).
MAX_RESULT_ROWS = 1000

# Names in the effective allowlist that are conventional import aliases rather
# than real importable module names. They are mapped to the module actually
# imported and then exposed in the namespace under the alias.
_MODULE_ALIASES: Dict[str, str] = {"np": "numpy", "pd": "pandas", "pl": "polars"}

# Builtins exposed inside the sandbox. Deliberately excludes ``__import__``,
# ``open``, ``eval``, ``exec``, ``compile``, ``input`` and anything else that
# reaches the filesystem, network, or interpreter internals. Data code and ML
# code only need value constructors, iteration helpers, and simple reflection.
_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "len": len, "range": range, "enumerate": enumerate, "sorted": sorted,
    "zip": zip, "map": map, "filter": filter, "reversed": reversed,
    "any": any, "all": all, "int": int, "float": float, "str": str,
    "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "frozenset": frozenset, "isinstance": isinstance, "issubclass": issubclass,
    "getattr": getattr, "hasattr": hasattr, "setattr": setattr,
    "type": type, "repr": repr, "print": print, "format": format,
    "True": True, "False": False, "None": None,
}


@dataclass
class SandboxResult:
    """Structured result of a single sandboxed execution (task 6.3 → 6.4).

    This is the contract that :meth:`PythonAgentValidator.sandbox_exec` returns
    and that the task-6.4 ``validate`` pipeline classifies into a
    :class:`ValidationOutcome`. The mapping is:

    * ``ok=True``                         -> ``ValidationOutcome.ok`` (or an
      "empty result" quality signal when ``row_count == 0``, decided by 6.4).
    * ``ok=False`` and ``timed_out=True`` -> ``execution_failure`` with the
      timeout message (Req 14.4).
    * ``ok=False`` otherwise              -> ``execution_failure`` carrying the
      runtime/syntax error (fixable by the correction LLM).

    Fields:
        ok:         Whether execution completed without raising and without
                    timing out.
        result:     The value produced by the final expression (or the last
                    DataFrame/Series bound in the namespace). Row-capped for
                    frame-like results. ``None`` on failure.
        error:      Human-readable error message ("" when ``ok``).
        error_type: Exception class name (e.g. ``KeyError``, ``TimeoutError``);
                    "" when ``ok``.
        row_count:  Number of rows in the (post-cap) result when it is
                    frame/series-like, else ``None``.
        elapsed_ms: Wall-clock execution time in milliseconds.
        timed_out:  ``True`` when the time budget was exceeded (Req 14.4).
        truncated:  ``True`` when the result was capped to ``MAX_RESULT_ROWS``
                    rows (Req 14.5).
    """

    ok: bool
    result: Any = None
    error: str = ""
    error_type: str = ""
    row_count: Optional[int] = None
    elapsed_ms: Optional[float] = None
    timed_out: bool = False
    truncated: bool = False


class PythonAgentValidator(PandasValidator):
    """General Python data-code validator (revised ``PandasValidator``).

    Extends :class:`PandasValidator` so the existing sandbox, connectivity, and
    cartesian-product machinery are reused. This validator adds:

    * AST-based import-allowlist enforcement (task 6.1, implemented here).
    * Task-type-aware structural checks (task 6.2, stub below).
    * A budgeted execution sandbox (task 6.3, stub below).
    * Four-category outcome classification and correction routing (task 6.4).
    """

    def __init__(self, *args, allowlist: Optional[Iterable[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Effective import allowlist additions from selected skills (the union
        # with BASE_ALLOWLIST is computed at check time). Defaults to empty.
        self.allowlist: Set[str] = set(allowlist or ())

    # -- task 6.1: import-allowlist enforcement ------------------------------

    def check_imports(
        self,
        code: str,
        query_id: str,
        allowlist: Optional[Iterable[str]] = None,
    ) -> Optional[ValidationOutcome]:
        """Enforce the import allowlist on ``code`` (Requirements 13.2, 13.3).

        Extracts imports via AST and returns a ``quality_rejection``
        :class:`ValidationOutcome` when the code imports any module outside
        ``BASE_ALLOWLIST | allowlist``. Returns ``None`` when every import is
        permitted so the caller can proceed to structural checks/execution.

        ``allowlist`` overrides the instance allowlist when provided.
        """
        effective_allowlist = (
            set(allowlist) if allowlist is not None else self.allowlist
        )
        disallowed = find_disallowed_imports(code, effective_allowlist)
        if disallowed:
            return ValidationOutcome.quality_rejection(
                query_id,
                "disallowed import: " + ", ".join(disallowed),
            )
        return None

    # -- task 6.2: task-type-aware structural checks -------------------------

    def run_structural_checks(
        self,
        code: str,
        task_types: Iterable[str],
        query_id: str = "",
    ) -> Optional[ValidationOutcome]:
        """Run relational structural checks only for join/union task types.

        The connectivity graph check and the Cartesian-product check inherited
        from :class:`PandasValidator` are relational-only concerns: they detect
        disjoint sub-results and accidental cross joins in code that combines
        multiple tables. They are meaningless — and produce false rejections —
        for pure ML / filter / transform code (classification, regression,
        timeseries, imputation, causal, …), which legitimately operates on a
        single frame and never "joins" anything.

        Gating (Requirements 13.4 and 13.5):

        * WHERE the plan's task types intersect
          :data:`RELATIONAL_TASK_TYPES` (``{"join", "union"}``), both structural
          checks run and a failure is surfaced as a ``quality_rejection``
          :class:`ValidationOutcome` (Req 13.4).
        * OTHERWISE the method is a no-op and returns ``None`` — code is NOT
          rejected for relational disjointness / Cartesian structure (Req 13.5).

        Returns:
            ``None`` when the checks are skipped (non-relational plan) or when
            all structural checks pass. A ``quality_rejection`` outcome when a
            relational plan's code is structurally invalid.

        The two underlying checks are reused as-is from :class:`PandasValidator`:

        * :meth:`PandasValidator._check_pandas_cartesian` returns a list of
          human-readable warnings (empty when clean).
        * :meth:`PandasValidator._check_query_connectivity` raises
          :class:`ValueError` describing the disjoint table groups.
        """
        if not (set(task_types) & RELATIONAL_TASK_TYPES):
            # Non-relational plan (Req 13.5): do not run relational-disjointness
            # / Cartesian rejection. Effectively a no-op.
            return None

        # Cartesian-product check (reused). Non-empty list == dangerous code.
        cartesian_warnings = self._check_pandas_cartesian(code)
        if cartesian_warnings:
            return ValidationOutcome.quality_rejection(
                query_id,
                "Dangerous query — possible Cartesian product:\n"
                + "\n".join(f"  - {w}" for w in cartesian_warnings),
            )

        # Connectivity graph check (reused). Raises ValueError on disjoint code.
        try:
            self._check_query_connectivity(code)
        except ValueError as exc:
            return ValidationOutcome.quality_rejection(query_id, str(exc))

        return None

    # -- task 6.3: execution sandbox with budgets ----------------------------

    def _build_sandbox_namespace(self, allowlist: Iterable[str]) -> Dict[str, Any]:
        """Build the restricted execution namespace (Requirement 14.1).

        The namespace exposes ONLY:

        * ``pd`` and ``pl`` — always available data-frame libraries.
        * The table aliases — each dataframe bound under its table name.
        * The modules in the effective import allowlist
          (``BASE_ALLOWLIST | allowlist``) — imported here and exposed by name.
          When the TabPFN skill is selected, ``allowlist`` contains
          ``tabpfn_client`` so ``import tabpfn_client`` becomes usable in the
          sandbox; otherwise it is simply absent from the namespace.

        Modules that cannot be imported (unavailable optional dependencies, or
        allowlist entries that are aliases rather than real modules) are skipped
        silently — they are conventional aliases like ``np`` handled via
        :data:`_MODULE_ALIASES`, or optional extras such as ``tabpfn_client``
        that are only installed with the ``ml`` extra.

        Nothing outside the allowlist (``os``, ``sys``, ``subprocess``, …) is
        ever placed in the namespace, and ``__builtins__`` is restricted to
        :data:`_SAFE_BUILTINS`.
        """
        effective = BASE_ALLOWLIST | set(allowlist or ())
        namespace: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "pl": pl,
        }

        for name in effective:
            real_name = _MODULE_ALIASES.get(name, name)
            try:
                namespace[name] = importlib.import_module(real_name)
            except Exception:
                # Alias without a matching module, or an optional dependency that
                # is not installed. Not exposing it is the safe default.
                continue

        # Table aliases: bind each dataframe under its (already-sanitised) name.
        for df, alias in zip(self.dataframes, self.table_names):
            namespace[alias] = df

        return namespace

    def _run_code(self, code: str, namespace: Dict[str, Any]) -> Any:
        """Execute ``code`` in ``namespace`` and return its result value.

        The final top-level statement is treated as the query result:

        * When it is an expression, every preceding statement is executed and
          the expression is evaluated and returned.
        * Otherwise all statements are executed and the most recently bound
          DataFrame/Series (pandas or polars) in the namespace is returned, or
          ``None`` when no frame-like value exists.

        Raises whatever the executed code raises (including ``SyntaxError`` from
        parsing); the caller converts these into a failed :class:`SandboxResult`.
        """
        tree = ast.parse(code)
        if not tree.body:
            return None

        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            if len(tree.body) > 1:
                head = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(head, "<sandbox>", "exec"), namespace)
            expr = ast.Expression(last.value)
            return eval(compile(expr, "<sandbox>", "eval"), namespace)

        exec(compile(tree, "<sandbox>", "exec"), namespace)
        frame_types = (pd.DataFrame, pd.Series, pl.DataFrame, pl.Series)
        for value in reversed(list(namespace.values())):
            if isinstance(value, frame_types):
                return value
        return None

    def _cap_result_rows(self, result: Any) -> tuple[Any, Optional[int], bool]:
        """Cap ``result`` to :data:`MAX_RESULT_ROWS` rows (Requirement 14.5).

        Applies to every task type. Frame/series-like results longer than the
        cap are truncated to the first ``MAX_RESULT_ROWS`` rows via ``.head``
        (supported by both pandas and polars). Returns
        ``(result, row_count, truncated)`` where ``row_count`` is the row count
        of the returned (post-cap) value, or ``None`` when the result has no
        length.
        """
        if result is None:
            return None, None, False

        try:
            length = len(result)
        except TypeError:
            # Scalar / non-sized result (e.g. a float from a regression score).
            return result, None, False

        if length > MAX_RESULT_ROWS and hasattr(result, "head"):
            return result.head(MAX_RESULT_ROWS), MAX_RESULT_ROWS, True
        return result, length, False

    def _execute_with_budget(
        self,
        code: str,
        namespace: Dict[str, Any],
        time_budget: float,
    ) -> tuple[Any, Optional[BaseException], float, bool]:
        """Run ``_run_code`` under a wall-clock ``time_budget`` (Requirement 14.4).

        Uses a worker thread joined with a timeout, which works on every
        platform including Windows (``signal.alarm`` is POSIX-only and cannot be
        used here). When the worker does not finish within ``time_budget`` the
        call returns ``timed_out=True``; the daemon worker is abandoned and torn
        down with the interpreter rather than being force-killed.

        Returns ``(result, error, elapsed_ms, timed_out)``.
        """
        holder: Dict[str, Any] = {}

        def worker() -> None:
            try:
                holder["result"] = self._run_code(code, namespace)
            except BaseException as exc:  # noqa: BLE001 — surfaced to the caller
                holder["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        start = time.perf_counter()
        thread.start()
        thread.join(time_budget)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if thread.is_alive():
            return None, None, elapsed_ms, True
        return holder.get("result"), holder.get("error"), elapsed_ms, False

    def sandbox_exec(
        self,
        code: str,
        task_types: Iterable[str],
        allowlist: Optional[Iterable[str]] = None,
    ) -> SandboxResult:
        """Execute ``code`` in a constrained sandbox with time/row budgets.

        Implements Requirement 14:

        * 14.1 — the namespace exposes only ``pd``, ``pl``, the table aliases,
          and the effective-allowlist modules (see
          :meth:`_build_sandbox_namespace`).
        * 14.2 — ML task types (``task_types`` intersecting
          :data:`ML_TASK_TYPES`) run under :data:`ML_TIME_BUDGET`.
        * 14.3 — all other task types run under :data:`DEFAULT_TIME_BUDGET`.
        * 14.4 — a timeout is reported as an execution failure (``ok=False``,
          ``timed_out=True``) carrying a timeout error message.
        * 14.5 — the result is row-capped to :data:`MAX_RESULT_ROWS` for every
          task type.

        Args:
            code: The (already import-checked / preprocessed) code to run.
            task_types: The plan's task types, used only to pick the time budget.
            allowlist: Skill-derived import allowlist additions. Defaults to the
                instance ``allowlist`` when omitted; unioned with
                :data:`BASE_ALLOWLIST` to form the effective allowlist.

        Returns:
            A :class:`SandboxResult` for the task-6.4 classifier to interpret.
        """
        effective_allowlist = (
            set(allowlist) if allowlist is not None else set(self.allowlist)
        )
        is_ml = bool(set(task_types) & ML_TASK_TYPES)
        time_budget = ML_TIME_BUDGET if is_ml else DEFAULT_TIME_BUDGET

        namespace = self._build_sandbox_namespace(effective_allowlist)
        result, error, elapsed_ms, timed_out = self._execute_with_budget(
            code, namespace, time_budget
        )

        # 14.4 — timeout reported as an execution failure.
        if timed_out:
            return SandboxResult(
                ok=False,
                error=(
                    f"Execution exceeded the {time_budget}s time budget "
                    f"({'ML' if is_ml else 'default'} task budget). "
                    "Likely cause: an expensive fit/inference, a runaway loop, "
                    "or an unbounded join. Reduce data volume, cap iterations, "
                    "or add explicit join keys."
                ),
                error_type="TimeoutError",
                elapsed_ms=elapsed_ms,
                timed_out=True,
            )

        # Any raised exception is a (fixable) execution failure.
        if error is not None:
            return SandboxResult(
                ok=False,
                error=str(error) or repr(error),
                error_type=type(error).__name__,
                elapsed_ms=elapsed_ms,
            )

        # 14.5 — cap result rows for all task types.
        capped, row_count, truncated = self._cap_result_rows(result)
        return SandboxResult(
            ok=True,
            result=capped,
            row_count=row_count,
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    # -- task 6.4: outcome classification / routing --------------------------

    @staticmethod
    def _extract_query_id(query: Any) -> str:
        """Return the opaque, echoed ``client_id`` for ``query`` (Req 16 contract).

        The ``client_id`` becomes :attr:`ValidationOutcome.query_id` and is echoed
        unchanged through validation/correction/judging/assembly. The generation
        models grow a real ``client_id`` field in task 8; until then this method
        is duck-typed so it works with both dict-shaped and object-shaped queries
        and falls back gracefully when no id is present.
        """
        if isinstance(query, dict):
            cid = query.get("client_id") or query.get("id")
        else:
            cid = getattr(query, "client_id", None)
            if cid is None:
                cid = getattr(query, "id", None)
        if cid is not None and str(cid) != "":
            return str(cid)
        # Graceful fallback: a stable per-object token so outcomes remain keyed.
        return str(id(query))

    @staticmethod
    def _extract_code(query: Any) -> str:
        """Return the generated code string from a dict- or object-shaped query."""
        if isinstance(query, dict):
            return (query.get("code") or "").strip()
        return (getattr(query, "code", "") or "").strip()

    def _preprocess_code(self, code: str) -> str:
        """Normalise generated code for structural checks and execution.

        Unlike :meth:`PandasValidator._clean_pandas` (which flattens everything
        onto a single ``;``-joined line and is tuned for single-expression pandas
        chains), this preprocessor PRESERVES newlines and indentation so that
        arbitrary multi-line Python data code — function definitions, loops, ML
        fit/predict blocks — still parses. It only:

        * removes wrapping markdown backticks, and
        * strips ``import`` / ``from ... import`` lines (the modules were already
          allowlist-checked on the RAW code via :meth:`check_imports`, and the
          sandbox pre-injects the permitted modules into the namespace, so leaving
          the import statements in would either be redundant or fail for optional
          extras that are not installed).
        """
        if not code:
            return ""
        text = code.replace("`", "")
        kept = [
            line
            for line in text.splitlines()
            if not re.match(r"^\s*(import\s+|from\s+\S+\s+import\b)", line)
        ]
        return "\n".join(kept).strip()

    def classify(self, result: "SandboxResult", query_id: str) -> ValidationOutcome:
        """Map a :class:`SandboxResult` onto exactly one :class:`ValidationOutcome`.

        Classification (Req 13.1) — the sandbox has already separated the failure
        modes, so this is a total, deterministic mapping:

        * ``timed_out`` or a raised exception (``ok is False``) -> ``execution_failure``
          (fixable by the correction LLM; the only category routed back — Req 13.6).
        * ``ok is True`` -> ``ok``. An EMPTY result (``row_count == 0``) is NOT an
          execution failure: per design §Data-Models the "empty" status is an
          assembly-time :class:`ExecutionTrace` concern, so the validator returns
          ``ok`` with ``row_count=0`` and lets assembly flag emptiness.
        """
        if not result.ok:
            return ValidationOutcome.execution_failure(
                query_id, result.error, elapsed_ms=result.elapsed_ms
            )
        return ValidationOutcome.ok(
            query_id, row_count=result.row_count, elapsed_ms=result.elapsed_ms
        )

    @staticmethod
    def should_retry(outcome: ValidationOutcome) -> bool:
        """Routing predicate for the validate-to-judge loop (Requirement 13.6).

        ONLY ``execution_failure`` outcomes are routed back to the code-correction
        LLM for a retry. ``quality_rejection`` (structural: disallowed import,
        disjoint query, Cartesian product) carries structured guidance but is NOT
        retried as a code fix; ``budget_exceeded`` stops the loop; ``ok`` proceeds
        to judging. The correction loop itself is the agent's job (task 10.2) — this
        predicate is the single source of truth it consults.
        """
        return outcome.category == "execution_failure"

    def validate(
        self,
        query: Any,
        dfs: Optional[List] = None,
        aliases: Optional[List[str]] = None,
        allowlist: Optional[Iterable[str]] = None,
        task_types: Optional[Iterable[str]] = None,
        budget: Any = None,
    ) -> ValidationOutcome:
        """Validate one generated query and return a classified ``ValidationOutcome``.

        Pipeline (design §4):

        1. Budget guard (defensive, duck-typed — ``BudgetGuard`` lands in task 10):
           if ``budget`` exposes ``exceeded()`` and it is truthy, short-circuit to
           ``budget_exceeded`` before doing any work.
        2. Import-allowlist enforcement on the RAW code via AST — a module outside
           ``BASE_ALLOWLIST | allowlist`` is a ``quality_rejection`` (Req 13.3);
           unparseable code is an ``execution_failure`` (fixable).
        3. Preprocess (strip imports / normalise, preserving structure).
        4. Structural checks — connectivity + Cartesian — but ONLY when the plan's
           task types intersect ``{join, union}`` (Req 13.4/13.5); a structural
           failure is a ``quality_rejection``.
        5. Sandbox execution under the ML/default time budget with row capping
           (task 6.3), then classify the :class:`SandboxResult` (Req 13.1).

        The four categories are mutually exclusive and every path returns exactly
        one of them. Only ``execution_failure`` is routed back for correction
        (see :meth:`should_retry`).

        Note: ``dfs``/``aliases`` are accepted for interface completeness; the
        sandbox and structural checks operate on the dataframes/table names the
        validator was constructed with (consistent with the task-6.3 sandbox),
        so this method does not mutate instance state mid-call.
        """
        query_id = self._extract_query_id(query)
        task_types = list(task_types or [])
        effective_allowlist = (
            set(allowlist) if allowlist is not None else set(self.allowlist)
        )

        # 1. Budget guard (duck-typed; BudgetGuard arrives in task 10).
        if budget is not None:
            exceeded = getattr(budget, "exceeded", None)
            if callable(exceeded):
                try:
                    if exceeded():
                        return ValidationOutcome.budget_exceeded(
                            query_id, "overall budget ceiling exceeded"
                        )
                except Exception:  # noqa: BLE001 — a broken guard must not block validation
                    logger.debug(
                        "budget.exceeded() check raised; continuing validation",
                        exc_info=True,
                    )

        raw_code = self._extract_code(query)
        if not raw_code:
            return ValidationOutcome.execution_failure(
                query_id, "query 'code' field is missing or empty"
            )

        # 2. Import-allowlist enforcement on the raw code (AST-based).
        try:
            import_outcome = self.check_imports(raw_code, query_id, effective_allowlist)
        except SyntaxError as exc:
            return ValidationOutcome.execution_failure(query_id, f"SyntaxError: {exc}")
        if import_outcome is not None:
            return import_outcome  # quality_rejection

        # 3. Preprocess (strip imports, normalise; structure preserved).
        code = self._preprocess_code(raw_code)
        if not code:
            return ValidationOutcome.execution_failure(
                query_id, "no executable code remained after preprocessing"
            )

        # 4. Structural checks — relational task types only (Req 13.4/13.5).
        structural_outcome = self.run_structural_checks(code, task_types, query_id)
        if structural_outcome is not None:
            return structural_outcome  # quality_rejection

        # 5. Sandbox execution + classification (Req 13.1).
        result = self.sandbox_exec(code, task_types, effective_allowlist)
        return self.classify(result, query_id)
