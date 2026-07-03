"""Skill registry: declarative skill cards loaded from markdown + front-matter.

A *skill card* is a markdown file with a YAML front-matter header describing a
capability (the only card today is TabPFN). :class:`SkillRegistry` loads every
``*.md`` card under :data:`SKILLS_DIR`, splitting the YAML front-matter from the
markdown body and building a :class:`SkillCard` per file.

Cards with missing or invalid front-matter are logged as warnings and skipped so
a single malformed card never prevents the remaining cards from loading.
"""

import os
from pathlib import Path
from typing import Callable, List, Literal, Optional

import yaml
from pydantic import BaseModel, ValidationError

import logging

from .models import StructuredQueryPlan, TableStats

logger = logging.getLogger(__name__)

# Default location of the skill cards shipped with the package.
SKILLS_DIR = Path(__file__).parent / "skills"

# Environment variable holding the hosted TabPFN API key. The value is used only
# to authenticate the model-limits/inference calls; it is NEVER logged nor
# injected into prompts or outputs.
TABPFN_API_KEY_ENV = "TABPFN_API_KEY"


class SkillCard(BaseModel):
    """A declarative skill capability parsed from a markdown card."""

    name: str
    version: int
    provider: Literal["api", "local"] = "api"
    applies_to: List[str]  # generation kinds this skill is valid for, e.g. ["PANDAS"]
    task_types: List[str]  # PlanStep.op values that trigger this skill
    data_constraints: dict  # rows/features/classes/context-budget gates
    requires: List[str]  # modules added to the executor/validator import allowlist
    keywords: List[str]
    body: str  # markdown body injected into the generation prompt


class SkillGateContext(BaseModel):
    """Runtime context consulted by the skill gate."""

    allow_tabpfn: bool  # from conf/workflow/*.yaml
    tabpfn_api_key_present: bool  # os.environ.get("TABPFN_API_KEY") is set
    api_limits: Optional[dict] = None  # live limits from tabpfn-client model-limits


class SkillSelection(BaseModel):
    """The set of skill cards selected for a generation run."""

    cards: List[SkillCard] = []


# Front-matter delimiter used at the top of every skill card.
_FRONT_MATTER_DELIMITER = "---"

# Pandas numeric dtype string prefixes. This mirrors the numeric convention used
# by ``ColumnStatistics`` (statistics.py): integer/unsigned/floating dtypes are
# numeric while ``bool`` (and object/category/datetime) are categorical. The
# skill gate only has the ``dtype`` string on each ``ColumnStat`` to work from,
# so numeric-ness is inferred from that string here.
_NUMERIC_DTYPE_PREFIXES = ("int", "uint", "float")


def _is_numeric_dtype(dtype: str) -> bool:
    """Return ``True`` when a pandas dtype string denotes a numeric column.

    Numeric dtypes are ``int*``, ``uint*``, and ``float*`` (case-insensitive, so
    nullable ``Int64``/``Float64`` are included). ``bool`` is NOT numeric, matching
    the ``ColumnStatistics`` convention; ``object``/``category``/``datetime`` are
    likewise treated as categorical.
    """
    normalized = str(dtype).strip().lower()
    if normalized.startswith("bool"):
        return False
    return normalized.startswith(_NUMERIC_DTYPE_PREFIXES)


def _stat_cardinality(stats: List[TableStats], column: str) -> int:
    """Return the cardinality recorded for ``column`` across ``stats``.

    When a column name appears in more than one table, the largest cardinality is
    used (the most conservative bound for the class-count gate). Returns 0 when the
    column is not present in any table.
    """
    values = [
        col.cardinality
        for stat in stats
        for col in stat.columns
        if col.column == column
    ]
    return max(values, default=0)


def _split_front_matter(text: str) -> tuple[Optional[dict], str]:
    """Split ``text`` into its YAML front-matter mapping and markdown body.

    Returns ``(front_matter, body)`` where ``front_matter`` is ``None`` when the
    document has no valid front-matter block. The body is everything after the
    closing delimiter (or the whole document when no front-matter is present).
    """
    stripped = text.lstrip()
    if not stripped.startswith(_FRONT_MATTER_DELIMITER):
        return None, text

    # Work from the first delimiter. Split into: "", front-matter, body...
    # Using maxsplit=2 keeps any "---" occurrences inside the body intact.
    parts = stripped.split(_FRONT_MATTER_DELIMITER, 2)
    if len(parts) < 3:
        # No closing delimiter -> malformed front-matter.
        return None, text

    front_matter_text = parts[1]
    body = parts[2].lstrip("\n")

    try:
        parsed = yaml.safe_load(front_matter_text)
    except yaml.YAMLError:
        return None, body

    if not isinstance(parsed, dict):
        return None, body

    return parsed, body


def _map_api_limits(api_limits: dict) -> dict:
    """Map raw TabPFN model-limits fields onto our constraint keys.

    The hosted model-limits endpoint returns fields such as ``max_cols``,
    ``max_classes``, cell/context budgets, and a per-version ``model_limits``
    mapping. This translates them into the constraint vocabulary used by the
    skill gate:

    - ``max_cols`` -> ``max_features``
    - ``max_classes`` -> ``max_classes``
    - ``test_set_max_cells`` / ``max_cells`` / ``context`` -> ``context_budget``
    - ``max_rows`` / ``max_samples`` -> ``max_rows`` (when reported)

    Only fields actually present in the API payload are returned, so callers can
    layer this on top of the front-matter defaults without erasing them.
    """
    if not isinstance(api_limits, dict) or not api_limits:
        return {}

    flat: dict = dict(api_limits)

    # The endpoint may nest per-model-version limits under "model_limits".
    # Prefer the default version, else the first dict entry, and let top-level
    # fields fill in anything the version block omits.
    nested = api_limits.get("model_limits")
    if isinstance(nested, dict) and nested:
        chosen = None
        default_version = api_limits.get("default_model_version")
        if default_version is not None and isinstance(nested.get(default_version), dict):
            chosen = nested[default_version]
        if chosen is None:
            chosen = next((v for v in nested.values() if isinstance(v, dict)), None)
        if isinstance(chosen, dict):
            merged = dict(chosen)
            for key, value in flat.items():
                if key != "model_limits":
                    merged.setdefault(key, value)
            flat = merged

    def _pick(*keys):
        for key in keys:
            value = flat.get(key)
            if value is not None:
                return value
        return None

    mapped: dict = {}
    max_features = _pick("max_cols", "max_features", "max_columns")
    if max_features is not None:
        mapped["max_features"] = max_features
    max_classes = _pick("max_classes", "max_num_classes")
    if max_classes is not None:
        mapped["max_classes"] = max_classes
    context_budget = _pick(
        "test_set_max_cells", "max_cells", "context", "context_budget",
        "train_set_max_cells",
    )
    if context_budget is not None:
        mapped["context_budget"] = context_budget
    max_rows = _pick("max_rows", "max_samples", "test_set_max_rows")
    if max_rows is not None:
        mapped["max_rows"] = max_rows
    return mapped


def _fetch_tabpfn_api_limits(api_key: str) -> Optional[dict]:
    """Best-effort single query of the tabpfn-client model-limits endpoint.

    Returns the raw limits mapping on success, or ``None`` when ``tabpfn-client``
    is not installed or the endpoint is unreachable. ``tabpfn-client`` is an
    OPTIONAL dependency, so the import is guarded: the module still imports
    cleanly without it.

    The ``api_key`` is used only to authenticate the call; it is never logged.
    """
    try:
        import tabpfn_client  # type: ignore  # noqa: F401
    except Exception:  # ImportError or any import-time failure
        logger.warning(
            "tabpfn-client is not installed; falling back to front-matter "
            "TabPFN data constraints"
        )
        return None

    try:
        # Authenticate with the hosted API. Older/newer client versions expose
        # this under slightly different names, so try the known entry points.
        for auth_name in ("set_access_token", "set_token"):
            auth = getattr(tabpfn_client, auth_name, None)
            if callable(auth):
                auth(api_key)
                break

        limits = _query_model_limits(tabpfn_client)
        if not limits:
            logger.warning(
                "TabPFN model-limits endpoint returned no data; falling back to "
                "front-matter data constraints"
            )
            return None
        return dict(limits)
    except Exception as exc:  # network error, auth error, API change, etc.
        # NOTE: never include api_key in the log record.
        logger.warning(
            "Could not query TabPFN model-limits endpoint (%s); falling back to "
            "front-matter data constraints",
            type(exc).__name__,
        )
        return None


def _query_model_limits(tabpfn_client) -> Optional[dict]:
    """Try the known tabpfn-client surfaces that expose model limits."""
    getters: List[Callable[[], object]] = []

    # Service wrapper (user-facing) surface.
    try:
        from tabpfn_client.service_wrapper import UserDataClient  # type: ignore

        for name in ("get_model_limits", "get_data_size_limits"):
            fn = getattr(UserDataClient, name, None)
            if callable(fn):
                getters.append(fn)
    except Exception:
        pass

    # Low-level service client surface.
    try:
        from tabpfn_client.client import ServiceClient  # type: ignore

        for name in ("get_model_limits", "get_data_size_limits", "model_limits"):
            fn = getattr(ServiceClient, name, None)
            if callable(fn):
                getters.append(fn)
    except Exception:
        pass

    # Any top-level convenience function.
    for name in ("get_model_limits", "model_limits"):
        fn = getattr(tabpfn_client, name, None)
        if callable(fn):
            getters.append(fn)

    for getter in getters:
        try:
            result = getter()
        except Exception:
            continue
        if isinstance(result, dict) and result:
            return result
    return None


class SkillRegistry:
    """Loads skill cards and (in later phases) selects/gates them."""

    def __init__(self, cards: List[SkillCard]):
        self.cards = cards

    @classmethod
    def load(
        cls,
        skills_dir: Path = SKILLS_DIR,
        gate_ctx: Optional["SkillGateContext"] = None,
        limits_fetcher: Optional[Callable[[str], Optional[dict]]] = None,
    ) -> "SkillRegistry":
        """Parse every ``*.md`` card under ``skills_dir`` into a ``SkillCard``.

        Cards whose front-matter is missing or invalid are logged as a warning
        and skipped; the remaining cards continue to load.

        When ``gate_ctx`` is provided and the TabPFN API key is present, the
        registry queries the ``tabpfn-client`` model-limits endpoint **once** and
        stores the raw limits in ``gate_ctx.api_limits``. If the client is
        unavailable or the endpoint is unreachable, ``api_limits`` is left as-is
        and the front-matter ``data_constraints`` act as the fallback. The API
        key is never logged.
        """
        cards: List[SkillCard] = []

        skills_dir = Path(skills_dir)
        if not skills_dir.is_dir():
            logger.warning("Skills directory not found: %s", skills_dir)
            return cls(cards)

        for path in sorted(skills_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read skill card %s: %s", path.name, exc)
                continue

            front_matter, body = _split_front_matter(text)
            if front_matter is None:
                logger.warning(
                    "Skipping skill card %s: missing or invalid front-matter",
                    path.name,
                )
                continue

            try:
                card = SkillCard(body=body, **front_matter)
            except (ValidationError, TypeError) as exc:
                logger.warning(
                    "Skipping skill card %s: invalid front-matter fields: %s",
                    path.name,
                    exc,
                )
                continue

            cards.append(card)

        registry = cls(cards)
        registry._load_api_limits(gate_ctx, limits_fetcher)
        return registry

    @staticmethod
    def _load_api_limits(
        gate_ctx: Optional["SkillGateContext"],
        limits_fetcher: Optional[Callable[[str], Optional[dict]]],
    ) -> None:
        """Populate ``gate_ctx.api_limits`` from the live endpoint when possible.

        No-op unless a gate context is supplied, the API key is present, and the
        limits have not already been fetched. Never raises and never logs the key.
        """
        if gate_ctx is None or not gate_ctx.tabpfn_api_key_present:
            return
        if gate_ctx.api_limits is not None:
            return  # already fetched; keep the single-call contract

        api_key = os.environ.get(TABPFN_API_KEY_ENV)
        if not api_key:
            # The gate context claims a key is present but the environment does
            # not hold one; nothing to query against.
            return

        fetcher = limits_fetcher or _fetch_tabpfn_api_limits
        try:
            limits = fetcher(api_key)
        except Exception as exc:  # defensive: fetchers should not raise
            logger.warning(
                "TabPFN model-limits lookup failed (%s); using front-matter "
                "constraints",
                type(exc).__name__,
            )
            limits = None

        if limits:
            gate_ctx.api_limits = dict(limits)

    def effective_constraints(
        self, card: SkillCard, gate_ctx: SkillGateContext
    ) -> dict:
        """Merge a card's front-matter constraints with live API limits.

        Live API limits win when present. API fields are mapped onto our
        constraint vocabulary (``max_cols`` -> ``max_features``, ``max_classes``
        -> ``max_classes``, cell/context budget -> ``context_budget``). When no
        live limits are available, the front-matter ``data_constraints`` are
        returned unchanged as a conservative fallback.
        """
        constraints = dict(card.data_constraints)
        api_limits = gate_ctx.api_limits if gate_ctx is not None else None
        if not api_limits:
            return constraints

        mapped = _map_api_limits(api_limits)
        for key, value in mapped.items():
            if value is not None:
                constraints[key] = value
        return constraints

    @staticmethod
    def effective_feature_count(stats: List[TableStats]) -> int:
        """Estimate the post-encoding feature count across all tables.

        Numeric columns count as a single feature; categorical columns contribute
        their cardinality (the width of a one-hot / ``get_dummies`` expansion), with
        a floor of 1 so a constant categorical column still counts as one feature.
        """
        total = 0
        for stat in stats:
            for column in stat.columns:
                if _is_numeric_dtype(column.dtype):
                    total += 1
                else:
                    total += max(1, column.cardinality)
        return total

    @staticmethod
    def target_class_count(
        plan: StructuredQueryPlan, stats: List[TableStats]
    ) -> int:
        """Return the largest classification-target cardinality in the plan.

        Collects the target columns of every ``classification`` step (from each
        step's ``columns_role["target"]``) and returns the maximum cardinality
        recorded for those columns, or 0 when there are no classification targets.
        """
        targets = [
            col
            for step in plan.steps
            if step.op == "classification"
            for col in step.columns_role.get("target", [])
        ]
        return max(
            (_stat_cardinality(stats, target) for target in targets),
            default=0,
        )

    def passes_data_constraints(
        self,
        card: SkillCard,
        limits: dict,
        plan: StructuredQueryPlan,
        stats: List[TableStats],
    ) -> bool:
        """Return whether the data respects a card's effective data constraints.

        Checks (each applied only when the corresponding limit is present):
        minimum/maximum training rows, effective feature count, the combined
        row x feature ``context_budget``, and — for classification plans — the
        target-column class count.
        """
        total_rows = max((stat.num_rows for stat in stats), default=0)
        eff_features = self.effective_feature_count(stats)

        if "min_rows" in limits and total_rows < limits["min_rows"]:
            return False
        if "max_rows" in limits and total_rows > limits["max_rows"]:
            return False
        if "max_features" in limits and eff_features > limits["max_features"]:
            return False
        if (
            "context_budget" in limits
            and total_rows * eff_features > limits["context_budget"]
        ):
            return False
        if "classification" in plan.task_types and "max_classes" in limits:
            if self.target_class_count(plan, stats) > limits["max_classes"]:
                return False
        return True

    def select(
        self,
        plan: StructuredQueryPlan,
        kind: str,
        stats: List[TableStats],
        gate_ctx: SkillGateContext,
    ) -> SkillSelection:
        """Two-stage skill selection with programmatic gating.

        Stage 1 (declared applicability): a card is dropped when the requested
        generation ``kind`` is not in its ``applies_to`` (so SQL excludes ML skills)
        or when its ``task_types`` are disjoint from the plan's task types.

        Stage 2 (programmatic gates): a card is dropped when the data violates its
        effective data constraints. TabPFN is additionally withheld when the
        ``allow_tabpfn`` configuration gate is disabled or the API key is absent.
        """
        candidates: List[SkillCard] = []
        plan_task_types = set(plan.task_types)

        for card in self.cards:
            # Stage 1 - declared applicability.
            if kind.upper() not in card.applies_to:
                continue  # SQL excludes ML skills
            if not (set(card.task_types) & plan_task_types):
                continue

            # Stage 2 - programmatic gates.
            limits = self.effective_constraints(card, gate_ctx)
            if not self.passes_data_constraints(card, limits, plan, stats):
                continue
            if card.name == "tabpfn":
                if not gate_ctx.allow_tabpfn:
                    continue  # yaml gate
                if not gate_ctx.tabpfn_api_key_present:
                    continue  # .env gate

            candidates.append(card)

        return SkillSelection(cards=candidates)

    def import_allowlist(self, selection: SkillSelection) -> set:
        """Union of ``requires`` across the selected skills.

        Returns the set of module names that the executor/validator import
        allowlist must include so the selected skills' code can run. This is the
        union of every selected card's ``requires`` list.

        When no skill is selected the result is an empty set, so a module such as
        ``tabpfn_client`` is included only when the TabPFN card is in the
        selection (Requirements 12.1, 12.2, 12.3).
        """
        allowlist: set = set()
        for card in selection.cards:
            allowlist.update(card.requires)
        return allowlist
