from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import difflib
import logging
import math
import pickle
import queue as queue_module
import re
import multiprocessing as mp
import numbers
import warnings
import numpy as np
import pandas as pd

# Generated code routinely computes correlations/statistics over messy
# real-world data, where a constant (zero-variance) column makes the result
# mathematically undefined (0/0 = NaN) rather than wrong — numpy's
# cov/corrcoef internals (df.corr(), Series.corr()) print a RuntimeWarning
# for this well-defined edge case regardless, which reads as an alarming
# crash signal it isn't. Silenced by _sandbox_worker (below) around the
# actual execution — just this message family, not RuntimeWarning at
# large, so a genuine numeric problem (e.g. a real overflow) still surfaces.
_BENIGN_NUMPY_STATS_WARNING = r"(invalid value encountered|divide by zero encountered|Degrees of freedom)"

from orqa.utils import prepare_dataframe

TECHNICAL_TERMS = [
    'dataframe', 'dataframes', 'dataset', 'datasets',
    'schema', 'dtype', 'index', 'indices',
    'null', 'nan', 'none',
    'record', 'records', 'row', 'rows',
    'column', 'columns', 'field', 'fields',
    'table', 'tables', 'entry', 'entries',
    'query', 'select', 'distinct',
    'join', 'merge', 'union', 'concat',
    'groupby', 'group by', 'order by',
    'pivot', 'unpivot', 'melt',
    'primary key', 'foreign key',
    'pd.', 'df.', 'sql', 'duckdb',
    'str.lower', 'astype', 'fillna', 'dropna',
    'coefficient',
    'pearson', 'spearman', 'kendall',
    'covariance', 'r-squared',
    'percentile', 'quantile', 'variance',
    'p-value', 'hypothesis', 'significance',
    'aggregate', 'aggregation',
    'reshape', 'subset', 'slice',
]

TERM_SUGGESTIONS = {
    'join':        '"combine", "match", or "link" — e.g. "link customers with their orders".',
    'merge':       '"combine" or "bring together" — e.g. "combine sales data with location info".',
    'coefficient': 'Drop it — ask "how much do X and Y correlate" or "is there a correlation between X and Y" without naming the statistic.',
    'pearson':     'Drop the method name — ask "how much do X and Y correlate" or "is there a correlation between X and Y", never which formula computes it.',
    'spearman':    'Drop the method name — ask "how much do X and Y correlate" or "is there a correlation between X and Y", never which formula computes it.',
    'kendall':     'Drop the method name — ask "how much do X and Y correlate" or "is there a correlation between X and Y", never which formula computes it.',
    'groupby':     '"for each" or "broken down by" — e.g. "average revenue for each region".',
    'group by':    '"for each" or "per" — e.g. "total sales per category".',
    'aggregate':   '"total" or "combined" — e.g. "total revenue per store".',
    'aggregation': '"summary" or "total" — e.g. "summary of sales by region".',
    'filter':      '"where", "only", or "that have" — e.g. "restaurants with more than 10 inspections".',
    'query':       'Describe the business question directly — e.g. "Which customers spent the most last month?".',
    'select':      '"find", "show", or "list" — e.g. "Show the top 10 restaurants by revenue".',
    'pivot':       '"broken down by" or "compared across" — e.g. "Revenue across regions and categories".',
    'null':        '"missing" or "without a value" — e.g. "restaurants without a listed address".',
    'nan':         '"missing" or "not available" — e.g. "restaurants where the phone number is not available".',
    'none':        '"missing" or "not available" — e.g. "restaurants without a listed address".',
    'schema':      'Describe the data directly — e.g. "restaurant name, address, and inspection date".',
    'dataframe':   '"data" or describe the subject — e.g. "the restaurant data".',
    'dataframes':  '"data" or describe the subject — e.g. "the restaurant data".',
    'dataset':     '"data" or name the subject — e.g. "the inspection records".',
    'datasets':    '"data" or name the subject — e.g. "the inspection records".',
    'row':         'Name the subject — e.g. "each restaurant" instead of "each row".',
    'rows':        'Name the subject — e.g. "restaurants" instead of "rows".',
    'column':      'Name the information — e.g. "the restaurant name" instead of "the name column".',
    'columns':     'Name the information — e.g. "the name and address" instead of "the columns".',
    'table':       'Name the subject directly — e.g. "the restaurants" instead of "the table".',
    'tables':      'Name the subject directly — e.g. "the restaurants" instead of "the tables".',
    'field':       'Name the information — e.g. "the phone number" instead of "the field".',
    'fields':      'Name the information — e.g. "the phone number and address" instead of "the fields".',
    'record':      'Name the subject — e.g. "each restaurant" instead of "each record".',
    'records':     'Name the subject — e.g. "restaurants" instead of "records".',
    'entry':       'Name the subject — e.g. "each inspection" instead of "each entry".',
    'entries':     'Name the subject — e.g. "inspections" instead of "entries".',
    'index':       'Name what distinguishes it — e.g. describe the identifying detail, not "the index".',
    'indices':     'Name what distinguishes them, or name the subject directly instead of "indices".',
    'union':       '"combined" or "across both" — e.g. "restaurants across both lists".',
}


logger = logging.getLogger(__name__)


# Cap on error text shipped back from the sandbox: exception messages can
# embed whole DataFrame reprs, and the text ends up verbatim in LLM feedback.
_SANDBOX_ERROR_MAX_CHARS = 2000


def _sandbox_error_text(e: BaseException) -> str:
    msg = str(e) or type(e).__name__
    if len(msg) > _SANDBOX_ERROR_MAX_CHARS:
        msg = msg[:_SANDBOX_ERROR_MAX_CHARS] + " … [truncated]"
    return msg


def _sandbox_worker(queue: mp.Queue, fn, args: tuple) -> None:
    """Execute *fn* inside the sandbox process.

    The parent process enforces a timeout via p.join(timeout=...).
    No OS-level memory limit is applied.

    Every outcome — including BaseException escapes like a generated
    ``raise SystemExit`` and results the queue cannot pickle — must be
    reported through the queue: an empty queue makes the parent report a
    misleading "sandbox crashed" error.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=_BENIGN_NUMPY_STATS_WARNING
            )
            result = fn(*args)
    except MemoryError:
        queue.put((
            "error",
            "MemoryError",
            "Query ran out of memory. "
            "Likely cause: large intermediate result or cartesian product.\n"
            "Fix: pre-filter rows, select only needed columns, or add stricter join conditions."
        ))
        return
    except BaseException as e:
        # BaseException on purpose: generated code can raise SystemExit /
        # KeyboardInterrupt, which `except Exception` would let kill the
        # worker silently (empty queue -> bogus "segfault" report upstream).
        queue.put(("error", type(e).__name__, _sandbox_error_text(e)))
        return

    # mp.Queue pickles in a background feeder thread, so an unpicklable
    # result raises AFTER put() returns and the item is silently lost.
    # Pre-pickle here so the failure is caught and reported as feedback.
    try:
        payload = pickle.dumps(result)
    except MemoryError:
        queue.put((
            "error",
            "MemoryError",
            "Query result too large to return from the sandbox.\n"
            "Fix: aggregate or filter the result down before returning it."
        ))
        return
    except BaseException as e:
        queue.put((
            "error",
            "TypeError",
            f"Query result of type {type(result).__name__!r} cannot be "
            f"returned from the sandbox ({_sandbox_error_text(e)}).\n"
            "End the query with a DataFrame, Series, or plain Python value."
        ))
        return
    queue.put(("ok", payload))


class QueryValidator(ABC):

    DEFAULT_TIMEOUT   = 300
    DEFAULT_MEM_LIMIT = 512

    def __init__(
        self,
        dataframes:   List,
        table_names:  List[str],
        lookup_dict:  dict,
        mem_limit_mb: int = DEFAULT_MEM_LIMIT,
        timeout:      int = DEFAULT_TIMEOUT,
    ):
        self.dataframes   = dataframes
        self.table_names  = table_names
        self.lookup_dict  = lookup_dict
        self.mem_limit_mb = mem_limit_mb
        self.timeout      = timeout

        self.validation_errors: list = []
        self.good_queries:      dict = {}
        self.errors:            list = []
        # Aliases the QUESTION leaked (set by _check_table_names_in_question).
        # Kept separate from `unused_tables`, which that check also writes but
        # with the OPPOSITE meaning (the aliases it did NOT find) — reusing one
        # attribute for both made the leak feedback name the wrong tables.
        self.question_table_leaks: set = set()

    # ------------------------------------------------------------------
    # Table Name Sanitization
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_name(name: str) -> str | None:
        """Convert a single table name to a valid Python identifier.

        Rules:
        1. Replace all non-alphanumeric, non-underscore characters with '_'
        2. If result starts with a digit, prefix with '_'
        3. If result is empty or all underscores, return None
           (caller handles fallback with Table_{index})
        """
        # Replace all non-alphanumeric, non-underscore characters with '_'
        sanitized = re.sub(r'[^A-Za-z0-9_]', '_', name)

        # If result is empty or all underscores, return None
        if not sanitized or all(c == '_' for c in sanitized):
            return None

        # If result starts with a digit, prefix with '_'
        if sanitized[0].isdigit():
            sanitized = '_' + sanitized

        return sanitized

    def _sanitize_table_names(self, names: list[str]) -> tuple[list[str], dict[str, str]]:
        """Sanitize a list of table names, resolving collisions.

        For each name in *names*:
        1. Call ``_sanitize_name()``; use ``Table_{index}`` as fallback when
           the result is None (all-illegal-character names).
        2. Resolve collisions by appending numeric suffixes (``_2``, ``_3``, …).

        Returns:
            (sanitized_names, rewrite_map) where *rewrite_map* maps each
            original name to its sanitized counterpart (only entries that
            actually changed are included).
        """
        sanitized_names: list[str] = []
        rewrite_map: dict[str, str] = {}
        seen: dict[str, int] = {}  # base_name -> count of times seen

        for index, raw_name in enumerate(names):
            base = self._sanitize_name(raw_name)
            if base is None:
                base = f"Table_{index}"

            # Resolve collisions
            if base in seen:
                seen[base] += 1
                unique_name = f"{base}_{seen[base]}"
                # Keep incrementing if the suffixed name also collides
                while unique_name in seen:
                    seen[base] += 1
                    unique_name = f"{base}_{seen[base]}"
                seen[unique_name] = 1
                sanitized_names.append(unique_name)
                rewrite_map[raw_name] = unique_name
            else:
                seen[base] = 1
                sanitized_names.append(base)
                if raw_name != base:
                    rewrite_map[raw_name] = base

        return sanitized_names, rewrite_map

    def _rewrite_query(self, query: str, rewrite_map: dict[str, str]) -> str:
        """Replace all occurrences of original table names in query with sanitized versions.

        Keys are processed longest-first to avoid partial replacements when one
        name is a prefix of another.  Uses word-boundary regex (``\\b``) so that
        only standalone identifier occurrences are replaced.
        """
        # Sort by length descending so longer names are replaced first
        for original in sorted(rewrite_map.keys(), key=len, reverse=True):
            sanitized = rewrite_map[original]
            query = re.sub(rf'\b{re.escape(original)}\b', sanitized, query)
        return query

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------
    # Exception types the sandbox worker may report back by name. Anything
    # unknown becomes RuntimeError (message text is preserved either way).
    _SANDBOX_EXC_TYPES = {
        "KeyError": KeyError, "ValueError": ValueError, "TypeError": TypeError,
        "MemoryError": MemoryError, "SyntaxError": SyntaxError,
        "IndentationError": SyntaxError, "TabError": SyntaxError,
        "TimeoutError": TimeoutError, "NameError": NameError,
        "AttributeError": AttributeError, "IndexError": IndexError,
        "ZeroDivisionError": ZeroDivisionError, "OverflowError": OverflowError,
        "RecursionError": RecursionError, "ImportError": ImportError,
        "ModuleNotFoundError": ImportError, "ArithmeticError": ArithmeticError,
        "FloatingPointError": ArithmeticError, "UnicodeDecodeError": ValueError,
        "UnicodeEncodeError": ValueError, "StopIteration": RuntimeError,
        "SystemExit": RuntimeError, "KeyboardInterrupt": RuntimeError,
        "MergeError": ValueError, "ParserError": ValueError,
        "OutOfBoundsDatetime": ValueError, "InvalidIndexError": KeyError,
        "DataError": ValueError, "SpecificationError": ValueError,
        "DuplicateLabelError": ValueError, "IndexingError": IndexError,
        # duckdb exceptions the SQL validator doesn't translate itself.
        "OutOfMemoryException": MemoryError, "CatalogException": KeyError,
        "SyntaxException": SyntaxError,
    }

    def _run_in_sandbox(self, fn, args: tuple = ()) -> Any:
        result_queue = mp.Queue()
        p = mp.Process(
            target=_sandbox_worker,
            args=(result_queue, fn, args),
            daemon=True,
        )
        p.start()
        p.join(timeout=self.timeout)

        if p.is_alive():
            p.kill()
            p.join()
            raise TimeoutError(
                f"Query exceeded {self.timeout}s timeout limit.\n"
                "Likely causes: cartesian product from missing/incorrect join keys, "
                "overly relaxed filter producing too many rows, or expensive aggregation.\n"
                "Fix: add explicit join keys (on=), tighten WHERE/filter conditions, "
                "or pre-aggregate before joining."
            )

        # Never trust queue.empty() right after join(): the feeder thread may
        # still be flushing the worker's item. Block briefly on get() instead.
        try:
            message = result_queue.get(timeout=5)
        except queue_module.Empty:
            exit_code = p.exitcode
            hint = (
                " Exit code -9 usually means the OS killed it (out of memory)."
                if exit_code == -9 else ""
            )
            raise RuntimeError(
                f"Sandbox process exited unexpectedly without returning a result "
                f"(exit code: {exit_code}).{hint}\n"
                "The query may have crashed the interpreter or been killed by the OS.\n"
                "Fix: reduce the data processed — pre-filter rows, select fewer "
                "columns, or aggregate earlier."
            )

        status, *rest = message
        if status == "error":
            exc_type_name, exc_msg = rest[0], rest[1]
            exc_type = self._SANDBOX_EXC_TYPES.get(exc_type_name, RuntimeError)
            if exc_type is RuntimeError and exc_type_name not in self._SANDBOX_EXC_TYPES:
                exc_msg = f"{exc_type_name}: {exc_msg}"
            raise exc_type(exc_msg)

        # The worker ships the result pre-pickled (see _sandbox_worker).
        try:
            return pickle.loads(rest[0])
        except Exception as exc:
            raise RuntimeError(
                f"Sandbox result could not be decoded: {exc}\n"
                "End the query with a DataFrame, Series, or plain Python value."
            )

    # ------------------------------------------------------------------
    # Expected-result-type enforcement (plan contract)
    # ------------------------------------------------------------------

    @staticmethod
    def _describe_result_shape(result: Any) -> str:
        """Human-readable one-liner of what the executed result actually is,
        for the mismatch feedback message."""
        try:
            if isinstance(result, pd.DataFrame):
                return (
                    f"a DataFrame with {result.shape[0]} row(s) and "
                    f"{result.shape[1]} column(s) ({', '.join(map(str, result.columns[:6]))}"
                    f"{', …' if result.shape[1] > 6 else ''})"
                )
            if isinstance(result, pd.Series):
                return f"a Series of {len(result)} value(s) (dtype {result.dtype})"
            if isinstance(result, bool):
                return f"a boolean ({result})"
            if isinstance(result, numbers.Number):
                return f"a single number ({result})"
            if isinstance(result, str):
                shown = result if len(result) <= 60 else result[:60] + "…"
                return f"a string ({shown!r})"
            if isinstance(result, (list, tuple)):
                return f"a {type(result).__name__} of {len(result)} element(s)"
        except Exception:
            pass
        return f"a {type(result).__name__}"

    @staticmethod
    def _unwrap_scalar(r: Any) -> Any:
        """1x1 DataFrame or 1-element Series/list -> the inner scalar value."""
        try:
            if isinstance(r, pd.DataFrame) and r.shape == (1, 1):
                return r.iloc[0, 0]
            if isinstance(r, pd.Series) and len(r) == 1:
                return r.iloc[0]
            if isinstance(r, (list, tuple)) and len(r) == 1:
                return r[0]
        except Exception:
            pass
        return r

    @classmethod
    def _matches_expected_result_type(cls, result: Any, expected: str) -> bool:
        """True when ``result``'s shape satisfies the plan's declared
        ``expected_result_type``.

        Deliberately generous at the borders so the check catches GROSS
        mismatches (a 50-row table where one number was promised; a bare
        number where a per-group table was promised) without burning
        correction cycles on packaging trivia:
          * a 1x1 DataFrame / 1-element Series wrapping a scalar counts as
            that scalar type;
          * an indexed Series counts as 'table' (a per-group aggregate
            legitimately comes back as a Series);
          * a single-column DataFrame counts as 'list'.
        Polars frames/series are duck-typed via shape/len rather than
        imported here.
        """
        _unwrap_scalar = cls._unwrap_scalar

        is_df = isinstance(result, pd.DataFrame) or (
            hasattr(result, "shape") and hasattr(result, "columns")
        )
        is_series = (not is_df) and (
            isinstance(result, pd.Series)
            or (hasattr(result, "dtype") and hasattr(result, "__len__") and not isinstance(result, str))
        )

        if expected == "table":
            return is_df or is_series
        if expected == "list":
            if is_series or isinstance(result, (list, tuple)):
                return True
            if is_df:
                try:
                    return result.shape[1] == 1
                except Exception:
                    return False
            return False

        scalar = _unwrap_scalar(result)
        # np.bool_ covers numpy's boolean scalar under BOTH numpy 1.x
        # (numpy.bool_) and 2.x (numpy.bool) — it is not a subclass of
        # Python bool, so isinstance(scalar, bool) alone misses it.
        is_bool = isinstance(scalar, (bool, np.bool_))
        if expected == "boolean":
            # bool first: bool is a subclass of int, so the number branch
            # would otherwise swallow it.
            return is_bool
        if expected == "number":
            if is_bool:
                return False
            return isinstance(scalar, numbers.Number) or (
                hasattr(scalar, "item") and isinstance(getattr(scalar, "item", lambda: None)(), numbers.Number)
            )
        if expected == "text":
            return isinstance(scalar, str)
        return True  # unknown/absent declaration — nothing to enforce

    def _check_expected_result_type(self, query: dict, result: Any) -> None:
        """Enforce the plan's declared result contract on the executed result.

        Raises ``ValueError`` with a correction-ready message when the
        result's shape contradicts ``expected_result_type`` — the message is
        what the coding agent sees in its correction prompt, so it names the
        contract, what actually came back, and what to change.
        """
        expected = str(query.get("expected_result_type") or "").strip().lower()
        if expected not in ("table", "list", "number", "text", "boolean"):
            return  # no declared contract (older plans) — nothing to enforce
        if self._matches_expected_result_type(result, expected):
            return

        description = str(query.get("expected_result_description") or "").strip()
        described = f'\nThe plan describes the expected result as: "{description}"' if description else ""
        raise ValueError(
            "RESULT TYPE MISMATCH — the executed result does not have the "
            "shape the approved plan promised.\n"
            f"Expected result type (from the plan): '{expected}'.{described}\n"
            f"The code actually returned: {self._describe_result_shape(result)}.\n"
            "Fix the CODE's final result — do not change the question or the "
            "plan: "
            + {
                "number": "end with the single numeric value itself (e.g. "
                          "`result = float(total)`), not a DataFrame of "
                          "context columns around it.",
                "boolean": "end with the single True/False answer itself "
                           "(e.g. `result = bool(condition)`), not a "
                           "DataFrame or a count.",
                "text": "end with the single string value itself (e.g. the "
                        "winning category's name), not a table containing it.",
                "list": "end with one Series / single-column selection of "
                        "the values, not a multi-column DataFrame or a "
                        "single scalar.",
                "table": "end with the tabular result (a DataFrame, or a "
                         "per-group Series), not a lone scalar value.",
            }[expected]
        )

    def _check_result_finite(self, result: Any) -> None:
        """Reject a NaN/inf scalar final answer.

        A `number`-typed result is the plan's single final answer, so NaN/inf
        means the underlying computation is mathematically undefined (e.g. a
        correlation coefficient with zero variance on one side, or a ratio
        dividing by zero) — never a valid business answer, regardless of how
        many tables fed into it. Unlike ``_is_empty_result``/
        ``_empty_result_is_error`` (gated off for multi-table queries because
        sample-data key mismatches can legitimately yield zero rows), this
        check always applies: an undefined number is never the right shape
        for ANY query, single- or multi-table.
        """
        scalar = self._unwrap_scalar(result)
        if isinstance(scalar, bool) or isinstance(scalar, np.bool_):
            return
        if not isinstance(scalar, numbers.Number):
            return
        try:
            finite = math.isfinite(float(scalar))
        except (TypeError, ValueError):
            return
        if finite:
            return
        raise ValueError(
            "UNDEFINED RESULT — the executed result is NaN/infinite, not a "
            "valid business answer.\n"
            "This usually means a correlation (or other ratio) was computed "
            "against a column that has zero variance in this query's scope "
            "(e.g. an averaged year derived from a table already filtered "
            "to a single year), or a division landed on zero.\n"
            "Fix the CODE if the zero-variance/zero-denominator input is a "
            "mistake; if the input is genuinely constant given the plan's "
            "own scope, the question cannot be meaningfully answered this "
            "way and the PLAN needs a different final computation."
        )

    # ------------------------------------------------------------------
    # Main validation loop
    # ------------------------------------------------------------------
    def validate_queries(self, result: Dict) -> Tuple[bool, Dict, Dict, List[str]]:
        all_valid = True

        # Sanitize table names early so downstream methods receive valid identifiers
        sanitized_names, rewrite_map = self._sanitize_table_names(self.table_names)
        # Build a mapping from original ordered_name -> sanitized name
        sanitize_lookup = dict(zip(self.table_names, sanitized_names))

        for idx, q in enumerate(result["queries"]):
            actual_query = q
            try:
                dataframes, ordered_names = self.prefilter_dataframes(
                    actual_query['tables'], q.get("code") or ""
                )
                raw_code = q.get("code") or ""
                if not raw_code.strip():
                    raise ValueError("query 'code' field is missing or empty")

                query_code = self.replace_aliases(raw_code, self.lookup_dict)
                # Rewrite original table name references to sanitized identifiers
                query_code = self._rewrite_query(query_code, rewrite_map)
                query_code = self._preprocess_query(query_code.strip())
                actual_query["code"] = query_code
                # Map ordered_names to their sanitized counterparts
                sanitized_ordered = [sanitize_lookup[n] for n in ordered_names]
                result_data = self._execute_query(query_code, dataframes, sanitized_ordered)

                # FIX: empty-result rejection is now gated through _empty_result_is_error(),
                # which subclasses can override. For multi-table queries, sample data may
                # have no overlapping keys, so a logically correct merge will return 0 rows —
                # rejecting it here generates spurious correction cycles and token waste.
                if self._is_empty_result(result_data) and self._empty_result_is_error(ordered_names):
                    raise ValueError(self._build_empty_result_feedback())
                # Plan contract: the executed result must have the shape the
                # approved plan declared (expected_result_type). Checked after
                # the empty-result gate so an empty result gets its own, more
                # specific feedback rather than a shape complaint.
                self._check_expected_result_type(actual_query, result_data)
                self._check_result_finite(result_data)
                if not self._check_table_usage(query_code, sanitized_names):
                    raise ValueError(self._build_unused_tables_feedback())
                if not self._check_tables_field_coverage(actual_query):
                    raise ValueError(self._build_tables_field_coverage_feedback())
                if self._check_table_names_in_question(actual_query["question"]):
                    raise ValueError(self._build_question_tables_feedback())

                #technical_terms = self._check_technical_terms_in_question(actual_query["question"])
                #if technical_terms:
                #    raise ValueError(self._build_technical_terms_feedback(technical_terms))
                self.good_queries[idx] = actual_query

            except Exception as e:
                all_valid = False
                error_msg = e.args[0] if e.args else str(e)
                self.validation_errors.append({"query": actual_query, "error": f"{type(e).__name__}: {error_msg}"})
                self.errors.append(f"Error {type(e).__name__}: {error_msg}")

        # The second element is a legacy placeholder: no caller consumes it
        # (StatementValidator._run_static_validator discards it into
        # `_conv_errors` and builds its own per-query correction prompts from
        # `self.errors`/`self.validation_errors` instead — see ErrorFormatter,
        # which corrects exactly one query per call with no cross-query
        # batching). Kept only so the 4-tuple return shape doesn't change.
        return all_valid, {}, self.good_queries, self.errors

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _preprocess_query(self, query: str) -> str:
        pass

    @abstractmethod
    def _run_query(self, query: str, dataframes: list, table_names: list) -> Any:
        pass

    def _execute_query(self, query: str, dataframes: list, table_names: list) -> Any:
        return self._run_in_sandbox(self._run_query, args=(query, dataframes, table_names))

    @abstractmethod
    def _build_empty_result_feedback(self) -> str:
        pass

    @abstractmethod
    def _get_language_name(self) -> str:
        pass

    def _empty_result_is_error(self, ordered_names: list) -> bool:
        """
        Controls whether an empty execution result is treated as a hard validation error.
        Default: always reject empty results.
        Subclasses can override to be more lenient — e.g. skip rejection for multi-table
        queries where sample data may have no overlapping keys.
        """
        return True

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def prefilter_dataframes(self, tables, code: str = ""):
        df_by_name = dict(zip(self.table_names, self.dataframes))
        # Build reverse map: file-path/UUID → canonical table name (e.g. "xahu-rkwn" → "Table_0")
        # so that tables[].name entries still holding raw dataset IDs are resolved correctly.
        alias_by_file = {v: k for k, v in self.lookup_dict.items()}
        # Also index underscore variants (e.g. "qvir_knu3") so that table name
        # entries generated by LLMs — which convert dashes to underscores to form
        # valid identifiers — resolve to the correct canonical alias.
        for canonical, file_id in list(self.lookup_dict.items()):
            if "-" in file_id:
                alias_by_file.setdefault(file_id.replace("-", "_"), canonical)

        _logger = logging.getLogger(__name__)
        dataframes, ordered_names = [], []
        for table in tables:
            raw_name = table["name"]
            # Resolve UUID/file-path to canonical name if needed; fall back to raw_name
            name = alias_by_file.get(raw_name, raw_name)
            dataframe = df_by_name.get(name)
            if dataframe is None:
                available_tables = self.table_names
                suggestions = self._suggest_columns(name, available_tables)
                suggestion_msg = ""
                if suggestions:
                    suggestion_msg = f"\nDid you mean: {', '.join(suggestions)}?"
                raise KeyError(
                    f"Table '{name}' not found. Available: {self.table_names}.{suggestion_msg}\n"
                    "Check for a typo or stale table name."
                )
            # Safety net: ensure column labels are strings even if the DataFrame
            # was not loaded through the standard pipeline path.
            dataframe = prepare_dataframe(dataframe, alias=name, logger=_logger)
            cols = table.get("columns_involved") or []
            if cols:
                missing = [c for c in cols if c not in dataframe.columns]
                if missing and code:
                    # A `derive`/`aggregate` step's own OUTPUT column
                    # legitimately doesn't exist in the raw table yet — the
                    # code creates it itself (df['x'] = ..., .assign(x=...),
                    # a named-aggregation x=('col', 'func'), or a rename
                    # target). columns_involved is copied straight from the
                    # plan's declared columns (structured_outputs.py Table.
                    # columns_involved), which doesn't distinguish "read
                    # from the raw table" from "produced by this plan's own
                    # steps" — so a missing column the code itself creates
                    # is not a hallucinated/typo'd raw-column reference.
                    missing = [
                        c for c in missing
                        if not self._is_column_created_by_code(c, code)
                    ]
                if missing:
                    available_cols = list(dataframe.columns)
                    suggestions_per_col = []
                    for m in missing:
                        col_suggestions = self._suggest_columns(m, available_cols)
                        if col_suggestions:
                            suggestions_per_col.append(f"  '{m}' → did you mean: {', '.join(col_suggestions[:3])}?")
                    suggestion_msg = ""
                    if suggestions_per_col:
                        suggestion_msg = "\nSuggestions:\n" + "\n".join(suggestions_per_col)
                    raise KeyError(
                        f"columns_involved has unknown columns in '{name}': {missing}.{suggestion_msg}\n"
                        "Check for typos or stale references."
                    )
                # Reconcile the declaration with the code instead of slicing
                # the frame to it: executing on a columns_involved-sliced frame
                # turned every real-but-undeclared column reference into a fake
                # runtime KeyError — with a suggestion list computed from the
                # UNSLICED frame, so the feedback ("Closest columns: X" for a
                # KeyError on X itself) sent correction loops in circles.
                # Execution always gets the full prepared frame; columns the
                # code references that exist but were not declared are added
                # to columns_involved so the saved metadata stays truthful.
                if code:
                    # Check membership directly against each real column name
                    # (quoted form found in the code) rather than extracting
                    # "identifier-shaped" tokens from quotes: a column with
                    # spaces/punctuation/non-ASCII characters or a purely
                    # numeric name (see utils.clean_columns — these are no
                    # longer dropped) never matches `[A-Za-z_][A-Za-z0-9_]*`,
                    # so the old token-extraction regex silently never
                    # detected such a column as "used", even when the code
                    # correctly referenced it in quotes.
                    undeclared = [
                        c for c in dataframe.columns
                        if c not in cols and (f"'{c}'" in code or f'"{c}"' in code)
                    ]
                    if undeclared:
                        table["columns_involved"] = list(cols) + undeclared
            dataframes.append(dataframe)
            ordered_names.append(name)
        return dataframes, ordered_names

    def replace_aliases(self, code: str, aliases: dict) -> str:
        if code is None:
            return ""
        for table_name, alias in aliases.items():
            # Also handle the underscore variant of the alias (e.g. "qvir_knu3"
            # for "qvir-knu3") since LLMs convert dashes to underscores to
            # produce valid SQL / Python identifiers.
            alias_variants = {alias}
            if "-" in alias:
                alias_variants.add(alias.replace("-", "_"))
            for variant in alias_variants:
                code = code.replace(f'"{variant}"', table_name)
                code = code.replace(f"'{variant}'", table_name)
                code = re.sub(rf'\b{re.escape(variant)}\b', table_name, code)
        return code

    def _is_empty_result(self, result: Any) -> bool:
        if result is None:
            return True
        if hasattr(result, 'empty'):
            return result.empty
        if hasattr(result, 'is_empty'):
            return result.is_empty()
        if hasattr(result, '__len__'):
            return len(result) == 0
        return False

    def _check_table_usage(self, query_text: str, table_names: list[str] | None = None) -> bool:
        # Word-boundary match, not plain substring: a bare `in` check treats
        # "Table_1" as "used" whenever "Table_10" (or a sanitizer collision
        # suffix like "Table_0_2") appears in the query, letting a query that
        # never actually joins "Table_1" pass the all-tables-used check.
        names = table_names if table_names is not None else self.table_names
        if len(names) == 1:
            if re.search(rf'\b{re.escape(names[0])}\b', query_text):
                return True
            self.unused_tables = {names[0]}
            return False
        self.unused_tables = {
            t for t in names if not re.search(rf'\b{re.escape(t)}\b', query_text)
        }
        return len(self.unused_tables) == 0

    def _check_table_names_in_question(self, question: str) -> bool:
        question_lower = question.lower()
        found = {
            t for t in self.table_names
            if re.search(rf'\b{re.escape(t.lower())}\b', question_lower)
        }
        self.question_table_leaks = found
        self.unused_tables = set(self.table_names) - found
        return len(found) > 0

    def _check_technical_terms_in_question(self, question: str) -> List[str]:
        question_lower = question.lower()
        return [t for t in TECHNICAL_TERMS if re.search(rf'\b{re.escape(t)}\b', question_lower)]

    def _check_tables_field_coverage(self, query: dict) -> bool:
        alias_by_name = {v: k for k, v in self.lookup_dict.items()}
        raw_tables = query.get("tables") or query.get("Tables") or []
        normalised = []
        for t in raw_tables:
            entry = dict(t)
            raw_name = entry.get("name", "").strip()
            entry["name"] = alias_by_name.get(raw_name, raw_name)
            normalised.append(entry)
        query.pop("Tables", None)
        query["tables"] = normalised
        declared = {t["name"] for t in normalised}
        self.tables_field_missing = set(self.table_names) - declared
        self.tables_field_extra   = declared - set(self.table_names)
        return len(self.tables_field_missing) == 0

    # ------------------------------------------------------------------
    # Feedback builders
    # ------------------------------------------------------------------
    def _build_tables_field_coverage_feedback(self) -> str:
        lines = [
            "'tables' must carry exactly one entry per table this query uses — "
            f"required aliases: {', '.join(self.table_names)}."
        ]
        if self.tables_field_missing:
            lines.append("Missing entries: " + ", ".join(sorted(self.tables_field_missing)))
        if self.tables_field_extra:
            lines.append(
                "Unknown aliases (remove — they are not tables in this query): "
                + ", ".join(sorted(self.tables_field_extra))
            )
        lines.append(
            "Fix: each entry needs 'name' (the exact alias, verbatim) and "
            "'columns_involved' (only the columns the code actually reads)."
        )
        return "\n".join(lines)

    def _build_unused_tables_feedback(self) -> str:
        missing = ", ".join(sorted(self.unused_tables))
        return (
            f"Unused table(s): {missing} — every table in this query's plan must be "
            f"referenced by the code. Required aliases: {', '.join(self.table_names)}.\n"
            "Fix: bring the missing table in through the relationship the plan declares "
            "for it (its verified join/union keys), and actually USE one of its columns "
            "in the output, a filter, or an aggregation — a table referenced only as a "
            "join key still counts as unused. Never drop a table, and never invent a "
            "relationship that the plan does not list."
        )

    def _build_question_tables_feedback(self) -> str:
        leaked = ", ".join(sorted(self.question_table_leaks)) or "a table alias"
        return (
            f"The question names a table alias ({leaked}). It is read by someone who "
            "has never seen these tables.\n"
            "Fix: replace the alias with the real-world subject that table holds — the "
            "topic, entity type, agency, place, or period from its description — and "
            "leave the rest of the wording unchanged."
        )

    def _build_technical_terms_feedback(self, technical_terms_found: List[str]) -> str:
        found_str = ", ".join(technical_terms_found)
        lines = [
            f"Question contains technical terms: {found_str}.",
            "Rephrase as a non-technical business question.",
        ]
        specific = [f"{t} → {TERM_SUGGESTIONS[t]}" for t in technical_terms_found if t in TERM_SUGGESTIONS]
        if specific:
            lines.append("Suggestions: " + " | ".join(specific))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Column suggestion & error wrapping helpers
    # ------------------------------------------------------------------
    MAX_RESULT_ROWS = 5_000_000

    def _check_result_size(self, result) -> None:
        """Detect results exceeding 5M rows before full materialization.

        Raises MemoryError if the result has more than MAX_RESULT_ROWS rows.
        """
        if result is None:
            return
        row_count = None
        if hasattr(result, 'shape'):
            row_count = result.shape[0]
        elif hasattr(result, '__len__'):
            row_count = len(result)
        if row_count is not None and row_count > self.MAX_RESULT_ROWS:
            raise MemoryError(
                f"Query result has {row_count:,} rows, exceeding the {self.MAX_RESULT_ROWS:,} row limit.\n"
                "Fix: add WHERE/filter conditions, use LIMIT, or aggregate before returning."
            )

    @staticmethod
    def _is_column_created_by_code(col: str, code: str) -> bool:
        """Whether *code* itself creates column *col* rather than reading it
        from the raw table — a `derive`/`aggregate` step's own output,
        legitimately absent from the raw DataFrame before the code runs.

        Heuristic, not a parse: matches the common ways pandas code
        introduces a new column, so a false POSITIVE just lets a query
        through to real execution (which still raises an accurate error if
        the column genuinely never materializes) — cheaper than a false
        NEGATIVE, which permanently kills a correct query on a misleading
        "typo" message before it ever runs.
        """
        if not code:
            return False
        escaped = re.escape(col)
        # df['col'] = ...  /  df.loc[mask, 'col'] = ...  (bracket assignment
        # target — the quoted name immediately precedes `] =`, not `==`).
        if re.search(rf"""['"]{escaped}['"]\s*\]\s*=(?!=)""", code):
            return True
        # .assign(col=...), .agg(col=('src', 'func')), pd.NamedAgg output —
        # bare identifier immediately followed by `=` (kwarg-style, not a
        # comparison), not itself a dict/attribute access.
        if re.search(rf"(?<![.\w]){escaped}\s*=(?!=)", code):
            return True
        # .rename(columns={'old': 'col'}) — appears as a mapping's target value.
        if re.search(rf":\s*['\"]{escaped}['\"]", code):
            return True
        return False

    def _suggest_columns(self, missing_col: str, available: list[str], max_suggestions: int = 10) -> list[str]:
        """Return up to *max_suggestions* column names from *available* ordered by
        ascending edit distance from *missing_col*.

        Uses difflib.get_close_matches with a generous cutoff so that even
        moderately distant matches are surfaced.  The result is further sorted
        by SequenceMatcher ratio (descending) to guarantee the first element is
        the closest match.
        """
        if not available or not missing_col:
            return []

        # get_close_matches returns results sorted by similarity (best first)
        # Use a low cutoff to be generous with suggestions
        matches = difflib.get_close_matches(
            missing_col, available, n=max_suggestions, cutoff=0.3
        )
        return matches

    @staticmethod
    def _wrap_unknown_exception(e: Exception) -> str:
        """Wrap an unknown exception type into a structured error message.

        Includes the exception type name and at most 200 characters of the
        original message.
        """
        type_name = type(e).__name__
        raw_msg = str(e)
        truncated = raw_msg[:200] if len(raw_msg) > 200 else raw_msg
        return f"{type_name}: {truncated}"

