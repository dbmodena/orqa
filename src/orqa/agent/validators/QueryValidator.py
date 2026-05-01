from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import difflib
import json
import logging
import re
import sys
import multiprocessing as mp
import pandas as pd

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
    'correlate', 'correlation', 'coefficient',
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
    'correlate':   '"tend to increase/decrease together" or "relationship between".',
    'correlation': '"relationship" or "pattern" — e.g. "Is there a pattern between X and Y?".',
    'groupby':     '"for each" or "broken down by" — e.g. "average revenue for each region".',
    'group by':    '"for each" or "per" — e.g. "total sales per category".',
    'aggregate':   '"total" or "combined" — e.g. "total revenue per store".',
    'aggregation': '"summary" or "total" — e.g. "summary of sales by region".',
    'filter':      '"where", "only", or "that have" — e.g. "restaurants with more than 10 inspections".',
    'query':       'Describe the business question directly — e.g. "Which customers spent the most last month?".',
    'select':      '"find", "show", or "list" — e.g. "Show the top 10 restaurants by revenue".',
    'pivot':       '"broken down by" or "compared across" — e.g. "Revenue across regions and categories".',
    'null':        '"missing" or "without a value" — e.g. "restaurants without a listed address".',
    'nan':         '"missing" or "not available" — e.g. "entries where the phone number is not available".',
    'schema':      'Describe the data directly — e.g. "restaurant name, address, and inspection date".',
    'dataframe':   '"data" or describe the subject — e.g. "the restaurant data".',
    'dataset':     '"data" or name the subject — e.g. "the inspection records".',
    'row':         'Name the subject — e.g. "each restaurant" instead of "each row".',
    'column':      'Name the information — e.g. "the restaurant name" instead of "the name column".',
    'union':       '"combined" or "across both" — e.g. "restaurants across both lists".',
}


logger = logging.getLogger(__name__)


def _set_memory_limit_unix(mem_limit_bytes: int) -> None:
    """Set the process virtual-memory address-space limit on Unix via resource.setrlimit.

    The limit is set as current usage + the configured budget so that only
    *new* allocations made by the query are constrained (the Python runtime
    and loaded DataFrames are already in memory).

    Logs a warning and continues without the limit if the call fails for any
    reason (e.g. unsupported platform, permission error, or the resource module
    is unavailable on Windows).
    """
    try:
        import resource
        # Current virtual-memory size (soft limit baseline)
        try:
            import os
            current_usage = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            # Better: use actual process RSS via /proc if available
            try:
                with open(f'/proc/{os.getpid()}/statm', 'r') as f:
                    pages = int(f.read().split()[1])  # resident pages
                    current_usage = pages * os.sysconf('SC_PAGE_SIZE')
            except (FileNotFoundError, OSError):
                pass
        except (ValueError, OSError):
            current_usage = 256 * 1024 * 1024  # conservative fallback

        effective_limit = current_usage + mem_limit_bytes
        resource.setrlimit(resource.RLIMIT_AS, (effective_limit, effective_limit))
    except ImportError:
        logger.warning(
            "resource module not available on this platform; "
            "skipping memory limit enforcement"
        )
    except Exception as exc:
        logger.warning(
            "Failed to set memory limit via resource.setrlimit: %s. "
            "Continuing without OS-level memory limit.",
            exc,
        )


def _set_memory_limit_windows(mem_limit_bytes: int) -> None:
    """Set process memory limit on Windows using the Job Object API via ctypes.

    Creates a Job Object, configures it with a ProcessMemoryLimit, and assigns
    the current process to the job.  Logs a warning and continues without the
    limit if any step fails (e.g. missing API, permission error, or non-Windows
    platform).
    """
    try:
        import ctypes
        from ctypes import wintypes

        # --- ctypes structure definitions (Windows-only) ---
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        # Constants
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
        JobObjectExtendedLimitInformation = 9

        kernel32 = ctypes.windll.kernel32

        # 1. Create a Job Object
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            logger.warning(
                "CreateJobObjectW returned NULL; "
                "skipping Windows memory limit enforcement"
            )
            return

        # 2. Configure the extended limit information
        # On Windows the limit is absolute (not incremental).  The child process
        # already consumes memory for the Python runtime, loaded libraries, and
        # deserialized DataFrames.  We add the configured budget ON TOP of the
        # current committed memory so the limit only constrains *new* allocations
        # made by the query itself.
        import os
        try:
            import psutil
            current_usage = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            # psutil may not be available; fall back to a conservative 256MB estimate
            current_usage = 256 * 1024 * 1024

        effective_limit = current_usage + mem_limit_bytes

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = effective_limit

        # 3. Apply the limit to the Job Object
        success = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not success:
            logger.warning(
                "SetInformationJobObject failed; "
                "skipping Windows memory limit enforcement"
            )
            return

        # 4. Assign the current process to the Job Object
        process_handle = kernel32.GetCurrentProcess()
        success = kernel32.AssignProcessToJobObject(job, process_handle)
        if not success:
            logger.warning(
                "AssignProcessToJobObject failed; "
                "skipping Windows memory limit enforcement"
            )
            return

    except Exception as exc:
        logger.warning(
            "Failed to set memory limit via Windows Job Object API: %s. "
            "Continuing without OS-level memory limit.",
            exc,
        )


def _sandbox_worker(queue: mp.Queue, fn, args: tuple, mem_limit_bytes: int = 0) -> None:
    """Execute *fn* inside the sandbox process.

    If *mem_limit_bytes* > 0, the process memory is capped via the appropriate
    platform mechanism: resource.setrlimit on Unix, or the Windows Job Object
    API on Windows.  If the platform API is unavailable, a warning is logged
    and execution continues (the parent-side _check_input_memory pre-check
    still applies).
    """
    # Enforce memory limit in child process
    if mem_limit_bytes > 0:
        if sys.platform == "win32":
            _set_memory_limit_windows(mem_limit_bytes)
        else:
            _set_memory_limit_unix(mem_limit_bytes)

    try:
        queue.put(("ok", fn(*args)))
    except MemoryError as e:
        queue.put((
            "error",
            "MemoryError",
            f"Query exceeded memory limit ({mem_limit_bytes // (1024 * 1024)}MB). "
            "Likely cause: large intermediate result or cartesian product.\n"
            "Fix: pre-filter rows, select only needed columns, or add stricter join conditions."
        ))
    except Exception as e:
        queue.put(("error", type(e).__name__, str(e)))


class QueryValidator(ABC):

    DEFAULT_TIMEOUT   = 180
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
        self.mem_limit    = mem_limit_mb * 1024 * 1024
        self.timeout      = timeout

        self.validation_errors: list = []
        self.good_queries:      dict = {}
        self.errors:            list = []

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
    def _run_in_sandbox(self, fn, args: tuple = ()) -> Any:
        queue = mp.Queue()
        p = mp.Process(
            target=_sandbox_worker,
            args=(queue, fn, args, self.mem_limit),
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

        if queue.empty():
            exit_code = p.exitcode
            raise RuntimeError(
                f"Sandbox process exited unexpectedly without returning a result "
                f"(exit code: {exit_code}).\n"
                "The query may have caused a segfault or been killed by the OS."
            )

        status, *rest = queue.get()
        if status == "error":
            exc_type_name, exc_msg = rest[0], rest[1]
            exc_type = {
                "KeyError": KeyError, "ValueError": ValueError, "TypeError": TypeError,
                "MemoryError": MemoryError, "SyntaxError": SyntaxError,
                "TimeoutError": TimeoutError, "NameError": NameError,
                "AttributeError": AttributeError, "IndexError": IndexError,
                "ZeroDivisionError": ZeroDivisionError, "OverflowError": OverflowError,
                "MergeError": ValueError, "ParserError": ValueError,
                "OutOfBoundsDatetime": ValueError, "InvalidIndexError": KeyError,
            }.get(exc_type_name, RuntimeError)
            raise exc_type(exc_msg)
        return rest[0]

    # ------------------------------------------------------------------
    # Memory pre-check
    # ------------------------------------------------------------------
    def _check_input_memory(self) -> None:
        total_mb = sum(
            df.memory_usage(deep=True).sum()
            for df in self.dataframes
            if isinstance(df, pd.DataFrame)
        ) / (1024 ** 2)
        if total_mb > self.mem_limit_mb:
            raise MemoryError(
                f"Input data is {total_mb:.1f}MB, exceeds {self.mem_limit_mb}MB limit.\n"
                "Pre-filter your data before running this query."
            )

    # ------------------------------------------------------------------
    # Main validation loop
    # ------------------------------------------------------------------
    def validate_queries(self, result: Dict) -> Tuple[bool, Dict, Dict]:
        all_valid = True

        try:
            self._check_input_memory()
        except MemoryError as e:
            return False, [{"role": "user", "content": f"MemoryError: {e}"}], {}, [str(e)]

        # Sanitize table names early so downstream methods receive valid identifiers
        sanitized_names, rewrite_map = self._sanitize_table_names(self.table_names)
        # Build a mapping from original ordered_name -> sanitized name
        sanitize_lookup = dict(zip(self.table_names, sanitized_names))

        for idx, q in enumerate(result["queries"]):
            actual_query = q
            try:
                dataframes, ordered_names = self.prefilter_dataframes(actual_query['tables'])
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
                if not self._check_table_usage(query_code, sanitized_names):
                    raise ValueError(self._build_unused_tables_feedback())
                if not self._check_tables_field_coverage(actual_query):
                    raise ValueError(self._build_tables_field_coverage_feedback())
                if self._check_table_names_in_question(actual_query["question"]):
                    raise ValueError(self._build_question_tables_feedback())

                technical_terms = self._check_technical_terms_in_question(actual_query["question"])
                if technical_terms:
                    raise ValueError(self._build_technical_terms_feedback(technical_terms))
                self.good_queries[idx] = actual_query

            except Exception as e:
                all_valid = False
                error_msg = e.args[0] if e.args else str(e)
                self.validation_errors.append({"query": actual_query, "error": f"{type(e).__name__}: {error_msg}"})
                self.errors.append(f"Error {type(e).__name__}: {error_msg}")

        if all_valid:
            return True, {}, self.good_queries, self.errors
        return False, self._build_feedback(), self.good_queries, self.errors

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
    def prefilter_dataframes(self, tables):
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
                dataframes.append(dataframe[[c for c in cols if c in dataframe.columns]])
            else:
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
        names = table_names if table_names is not None else self.table_names
        if len(names) == 1:
            if names[0] in query_text:
                return True
            self.unused_tables = {names[0]}
            return False
        self.unused_tables = {t for t in names if t not in query_text}
        return len(self.unused_tables) == 0

    def _check_table_names_in_question(self, question: str) -> bool:
        question_lower = question.lower()
        found = {t for t in self.table_names if t.lower() in question_lower}
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
        lines = ["'tables' field must have one entry per table used."]
        if self.tables_field_missing:
            lines.append("Missing: " + ", ".join(sorted(self.tables_field_missing)))
        if self.tables_field_extra:
            lines.append("Unknown (remove): " + ", ".join(sorted(self.tables_field_extra)))
        lines.append("Each entry needs: 'name' (exact alias), 'reason', 'join_justification'.")
        return "\n".join(lines)

    def _build_unused_tables_feedback(self) -> str:
        #missing = ", ".join(
        #    f"{t} ({self.lookup_dict.get(t, '?')})" for t in sorted(self.unused_tables)
        #)
        missing = ", ".join(
            f"{t}" for t in sorted(self.unused_tables)
        )
        return (
            f"Query must reference ALL tables. Missing: {missing}.\n"
            f"Required: {', '.join(self.table_names)}. Join all tables into a single result."
        )

    def _build_question_tables_feedback(self) -> str:
        return (
            "Do not use table names in the question. "
            "Write it as a business user with no knowledge of table names would."
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

    def _find_repeated_errors(self, error_texts: list) -> list:
        rules = []
        checked = set()
        patterns = [
            (
                "str.lower() on literal",
                lambda e: "was called on a string literal" in e,
                "NEVER call .str.lower() on a quoted string. Use DataFrame['col'].str.lower(). "
                "Normalise via .assign(_key=Table_X['col'].str.lower()) before merge, then on='_key'."
            ),
            (
                "stale Series in chained merge",
                lambda e: "stale series" in e.lower() or ("stale" in e.lower() and "left_on" in e.lower()),
                "NEVER use Table_X['col'] as left_on/right_on after a prior .merge(). "
                "Normalise ALL keys with .assign() before the chain."
            ),
            (
                "merge() with no keys",
                lambda e: "no join keys" in e.lower(),
                "Always specify on=, left_on=, or right_on= in every .merge() call."
            ),
            (
                "NameError: merge not defined",
                lambda e: "name 'merge' is not defined" in e,
                "Use Table_A.merge(Table_B, ...) — not bare merge(...)."
            ),
            (
                "duplicate suffixes",
                lambda e: "duplicate columns" in e.lower() and "suffixes" in e.lower(),
                "Suffixes must be unique across ALL merge steps: ('_T1','_T2'), ('_T12','_T3'), etc."
            ),
            (
                "missing tables",
                lambda e: "must reference all provided tables" in e.lower(),
                "Every query MUST join ALL provided tables — none can be omitted."
            ),
            (
                "disjoint sub-queries",
                lambda e: "disjoint query" in e.lower(),
                "All table groups must form ONE connected result via merge/join/concat — no parallel chains."
            ),
        ]
        for label, matcher, rule in patterns:
            count = sum(1 for e in error_texts if matcher(e))
            if count >= 2 and label not in checked:
                rules.append(f"[repeated {count}x] {rule}")
                checked.add(label)
        return rules

    def _build_feedback(self) -> list:
        feedback_lines = [
            f"Invalid {self._get_language_name()} queries — fix the ones listed below:\n",
        ]
        queries = {"queries": []}

        repeated = self._find_repeated_errors([err["error"] for err in self.validation_errors])
        if repeated:
            feedback_lines.insert(0,
                "RECURRING MISTAKES — apply to ALL queries before fixing:\n"
                + "\n".join(f"  * {r}" for r in repeated) + "\n"
            )

        for idx, err in enumerate(self.validation_errors, start=1):
            queries["queries"].append(err["query"])
            feedback_lines.append(f"Query {idx}:\n{err['error']}\n")

        return [
            {"role": "system", "content": json.dumps(queries, indent=2)},
            {"role": "user",   "content": "\n".join(feedback_lines)},
        ]