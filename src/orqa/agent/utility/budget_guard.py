"""Overall budget ceiling for the statement-generation retry loops.

The statement-generation pipeline nests several retry loops (generation retries x
validation cycles x judge iterations x correction retries). Left unbounded these
multiply, so a single generation can consume unbounded wall-clock time or tokens.

:class:`BudgetGuard` enforces a single overall ceiling across *all* of those loops:
an overall wall-clock ceiling AND an overall token ceiling. Loops call
:meth:`start` once at the beginning, feed observed token usage through
:meth:`add_tokens`, and check :meth:`exceeded` at each iteration boundary. When the
guard reports the budget is exceeded the caller stops gracefully and returns
whatever was approved so far -- the guard itself never raises.

Limits are loaded from the ``query_generation`` section of a workflow config via
:meth:`from_config`, matching the ``load_config`` conventions used elsewhere
(``src/conf/config.py``). Sensible, generous defaults are applied when the config
keys are absent.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Config keys read from the ``query_generation`` section.
_WALL_SECONDS_KEY = "max_pool_time"
_TOTAL_TOKENS_KEY = "max_total_tokens"

# Generous fallback ceilings, applied when a config key is absent. These are large
# enough not to interfere with normal runs while still bounding pathological loops.
DEFAULT_MAX_WALL_SECONDS: float = 3600.0  # one hour
DEFAULT_MAX_TOTAL_TOKENS: int = 2_000_000


class BudgetGuard:
    """Enforce an overall wall-clock and token ceiling across all retry loops.

    Usage::

        budget = BudgetGuard.from_config(config_path)
        budget.start()
        while not budget.exceeded():
            usage = do_work()
            budget.add_tokens(usage)  # int or a usage dict

    The guard is deliberately side-effect free apart from accumulating tokens and
    reading a monotonic clock: :meth:`exceeded` never raises and never mutates
    state, so callers can poll it freely at loop boundaries.
    """

    def __init__(self, max_wall_seconds: float, max_total_tokens: int):
        self.max_wall_seconds = max_wall_seconds
        self.max_total_tokens = max_total_tokens
        self._t0: Optional[float] = None
        self._tokens: int = 0

    def start(self) -> None:
        """Begin (or restart) the wall-clock timer.

        Uses :func:`time.perf_counter` (monotonic) so the elapsed measurement is
        unaffected by system-clock adjustments.
        """
        self._t0 = time.perf_counter()

    def add_tokens(self, usage: Union[int, dict, None]) -> None:
        """Accumulate token usage.

        Accepts either a raw token count (``add_tokens(n)``) or a usage mapping
        (``add_tokens({"total_tokens": n})``) so it can be fed directly from the
        LLM client usage dicts used throughout the pipeline. ``None`` and missing
        keys contribute zero.
        """
        if usage is None:
            return
        if isinstance(usage, dict):
            n = usage.get("total_tokens", 0) or 0
        else:
            n = usage
        try:
            self._tokens += int(n)
        except (TypeError, ValueError):
            logger.warning("BudgetGuard.add_tokens ignored non-numeric usage: %r", usage)

    @property
    def tokens(self) -> int:
        """Total tokens accumulated so far."""
        return self._tokens

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since :meth:`start`; ``0.0`` before the timer starts."""
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0

    def exceeded(self) -> bool:
        """Return ``True`` when either ceiling has been crossed.

        The wall-clock ceiling is only consulted once :meth:`start` has been
        called; before then elapsed time is treated as ``0``. The token ceiling is
        always checked. This method never raises.
        """
        if self._tokens > self.max_total_tokens:
            return True
        if self._t0 is not None and self.elapsed_seconds() > self.max_wall_seconds:
            return True
        return False

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "BudgetGuard":
        """Build a guard from the ``query_generation`` section of a workflow config.

        Reads ``max_pool_time`` (wall-clock ceiling, seconds) and
        ``max_total_tokens`` (token ceiling) from the ``query_generation`` section,
        falling back to :data:`DEFAULT_MAX_WALL_SECONDS` / :data:`DEFAULT_MAX_TOTAL_TOKENS`
        when a key (or the whole section, or the file) is absent or unreadable.
        """
        section = cls._load_query_generation_section(config_path)

        max_wall_seconds = cls._coerce_positive(
            section.get(_WALL_SECONDS_KEY), DEFAULT_MAX_WALL_SECONDS, _WALL_SECONDS_KEY
        )
        max_total_tokens = int(
            cls._coerce_positive(
                section.get(_TOTAL_TOKENS_KEY), DEFAULT_MAX_TOTAL_TOKENS, _TOTAL_TOKENS_KEY
            )
        )
        return cls(max_wall_seconds=max_wall_seconds, max_total_tokens=max_total_tokens)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _load_query_generation_section(config_path: Union[str, Path]) -> dict:
        """Return the ``query_generation`` mapping from a YAML config, or ``{}``.

        The section lives under ``tasks -> query_generation`` in the workflow
        configs (``conf/workflow/*.yaml``); a top-level ``query_generation`` key is
        also accepted for flexibility. Any read/parse error is logged and an empty
        mapping is returned so defaults apply.
        """
        try:
            with open(config_path, "r") as file:
                parsed = yaml.safe_load(file) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "BudgetGuard could not read config %s (%s); using default budget",
                config_path,
                exc,
            )
            return {}

        if not isinstance(parsed, dict):
            return {}

        tasks = parsed.get("tasks")
        if isinstance(tasks, dict) and isinstance(tasks.get("query_generation"), dict):
            return tasks["query_generation"]
        if isinstance(parsed.get("query_generation"), dict):
            return parsed["query_generation"]
        return {}

    @staticmethod
    def _coerce_positive(value, default: float, key: str) -> float:
        """Coerce ``value`` to a positive number, falling back to ``default``."""
        if value is None:
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "BudgetGuard ignoring non-numeric %s=%r; using default %s",
                key,
                value,
                default,
            )
            return default
        if number <= 0:
            logger.warning(
                "BudgetGuard ignoring non-positive %s=%r; using default %s",
                key,
                value,
                default,
            )
            return default
        return number
