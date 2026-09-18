"""The two-level retrievability contract: build it for a table group, then
check a candidate question against it.

Replaces ``keyword_searchability.check_keyword_searchability`` — which
verified a plan's ``question_keywords`` against ONE lexical search — with a
check that (a) votes a QUESTION (not a separately-maintained keyword list)
across a panel of independent retrievers, collapsed to the CKAN family
level (Level A), and (b) verifies the question states whatever details
distinguish the gold file from its same-family siblings (Level B). See
``orqa.benchmark.retrieval_panel`` and ``orqa.benchmark.families``.

No-ops to an automatic pass when no panel/contract is available, the same
degradation convention ``check_keyword_searchability`` used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ...benchmark.families import Facet, FamilyIndex, distinguishing_facets, missing_facets
from ...benchmark.retrieval_panel import RetrieverPanel

CONTRACT_VERSION = 1


@dataclass
class TableContract:
    alias: str
    resource_id: str
    family_id: str
    family_size: int
    facets: list[Facet] = field(default_factory=list)
    residual_siblings: list[str] = field(default_factory=list)


@dataclass
class RetrievalContract:
    top_k: int
    min_agreement: int
    tables: list[TableContract] = field(default_factory=list)
    version: int = CONTRACT_VERSION

    @property
    def gold_ids(self) -> list[str]:
        return [t.resource_id for t in self.tables]

    def to_dict(self) -> dict:
        return {
            "contract_version": self.version,
            "top_k": self.top_k,
            "min_agreement": self.min_agreement,
            "tables": {
                t.resource_id: {
                    "family_id": t.family_id,
                    "family_size": t.family_size,
                    "facets": [f.label for f in t.facets],
                    "residual_siblings": t.residual_siblings,
                }
                for t in self.tables
            },
        }


def build_contract(
    plan_tables: list[dict],
    family_index: FamilyIndex,
    record_lookup: Callable[[str], Optional[dict]],
    top_k_per_table: int,
    max_top_k: int,
    min_agreement: int,
    max_residual_siblings: int,
    scope_loader: Optional[Callable[[str, Optional[list[str]]], Optional[dict]]] = None,
    single_table_top_k: int = 1,
) -> RetrievalContract:
    """Build the retrieval contract for one fixed table group.

    Args:
        plan_tables: ``[{"alias": str, "resource_id": str}, ...]`` — the
            exact tables this generation run is fixed to.
        family_index: Maps each resource id to its CKAN family/siblings.
        record_lookup: ``resource_id -> metadata record | None`` (e.g. the
            reverse index's own ``.get``), used for both the gold table and
            every sibling considered for distinguishing facets.
        top_k_per_table / max_top_k: for a MULTI-table group (more than one
            plan table), ``K = min(max_top_k, top_k_per_table *
            len(plan_tables))`` — Level A's target window. Two or more
            tables cannot all occupy rank 1 of the same search at once, so
            the window widens with table count instead.
        single_table_top_k: Level A's target for a SINGLE-table group
            (``len(plan_tables) <= 1``) — literal rank 1 by default: with
            only one target, a greedy search over the table's own real
            vocabulary reaching rank 1 costs no more than reaching a looser
            rank (see ``orqa.agent.utility.keyword_suggestion``'s module
            docstring), so there is no reason to settle for less.
        min_agreement: How many retrievers must pass for Level A.
        max_residual_siblings: Caps how many still-unresolved siblings get a
            (potentially expensive) data-scope lookup at all.
        scope_loader: ``(resource_id, columns) -> {column: value} | None``,
            called ONLY for residual siblings (and the gold table) when
            metadata facets alone don't separate them — ``columns=None`` on
            the first (gold) call means "every single-valued column";
            subsequent sibling calls are projected to the gold table's own
            scope columns. ``None`` skips the data-scope tier entirely.
    """
    n = max(len(plan_tables), 1)
    top_k = single_table_top_k if n == 1 else min(max_top_k, top_k_per_table * n)

    tables: list[TableContract] = []
    for t in plan_tables:
        alias, resource_id = t["alias"], t["resource_id"]
        family_id = family_index.family_id(resource_id)
        sibling_ids = family_index.siblings(resource_id)

        gold_record = record_lookup(resource_id) or {}
        sibling_records = {sid: (record_lookup(sid) or {}) for sid in sibling_ids}

        result = distinguishing_facets(gold_record, sibling_records)
        facets, residual = result["facets"], result["residual_siblings"]

        if residual and scope_loader is not None:
            capped, overflow = residual[:max_residual_siblings], residual[max_residual_siblings:]
            try:
                gold_scope = scope_loader(resource_id, None) or {}
            except Exception:
                gold_scope = {}
            if gold_scope:
                sibling_scopes: dict[str, dict] = {}
                for sid in capped:
                    try:
                        scope = scope_loader(sid, list(gold_scope))
                    except Exception:
                        scope = None
                    if scope:
                        sibling_scopes[sid] = scope
                if sibling_scopes:
                    rerun = distinguishing_facets(
                        gold_record,
                        {sid: sibling_records[sid] for sid in capped},
                        gold_scope=gold_scope,
                        sibling_scopes=sibling_scopes,
                    )
                    facets = facets + [f for f in rerun["facets"] if f.kind == "scope"]
                    residual = rerun["residual_siblings"] + overflow

        tables.append(
            TableContract(
                alias=alias,
                resource_id=resource_id,
                family_id=family_id,
                family_size=len(sibling_ids) + 1,
                facets=facets,
                residual_siblings=residual,
            )
        )

    return RetrievalContract(top_k=top_k, min_agreement=min_agreement, tables=tables)


def _level_a_feedback(
    contract: RetrievalContract,
    vote: dict,
    missing_tables: list[str],
    question: str = "",
    retrievable_keywords: Optional[list[str]] = None,
) -> str:
    if not missing_tables:
        return ""
    per_retriever = vote["per_retriever"]
    failing = [name for name, r in per_retriever.items() if not r["pass"]]
    lines = [
        f"Level A (family reachable): searching this question surfaces the "
        f"family of {', '.join(missing_tables)} on only {vote['passes']} of "
        f"{len(per_retriever)} retrievers within top {contract.top_k} "
        f"(need {min(contract.min_agreement, len(per_retriever))}); "
        f"{', '.join(failing)} did not."
    ]
    for table in contract.tables:
        if table.alias not in missing_tables:
            continue
        ranks = {name: r["family_ranks"].get(table.resource_id) for name, r in per_retriever.items()}
        lines.append(f"  {table.alias}'s family ranks per retriever: {ranks}.")
    # The diagnostic above says THAT and BY HOW MUCH the question missed,
    # but names no actual words — unlike Level B's facet labels, it left
    # the LLM nothing concrete to write instead. `retrievable_keywords` is
    # the SAME pre-verified anchor (see
    # orqa.agent.utility.keyword_suggestion.suggest_retrievable_keywords)
    # already rendered into the planning prompt as a hint; naming it again
    # HERE, against the question that just failed, turns it into a
    # pointed diagnosis: either the anchor was never used (say so, and
    # name the exact missing terms) or it was used and still isn't enough
    # (say that instead, so the planner doesn't just re-paste the same
    # anchor and expect a different result).
    if retrievable_keywords:
        question_lower = question.lower()
        unused = [kw for kw in retrievable_keywords if kw.lower() not in question_lower]
        if unused:
            lines.append(
                "  A pre-verified keyword anchor exists for this table group "
                f"but is missing from the question: {', '.join(unused)}. Weave "
                "these exact terms into the question's own prose."
            )
        else:
            lines.append(
                "  The pre-verified keyword anchor "
                f"({', '.join(retrievable_keywords)}) is already present in "
                "the question, but the combination still misses the required "
                "rank — add more of the table's own title/tag/column "
                "vocabulary alongside it rather than repeating the same terms."
            )
    return "\n".join(lines)


def _level_b_feedback(missing_by_alias: dict[str, list[Facet]]) -> str:
    if not missing_by_alias:
        return ""
    lines = ["Level B (distinguishing details): the question is missing —"]
    for alias, facets in missing_by_alias.items():
        wanted = "; ".join(f.label for f in facets)
        lines.append(f"  {alias}: {wanted}.")
    lines.append(
        "State these plainly in the question's own prose (not only in "
        "question_keywords) — they are what rules this table's file out "
        "from its same-dataset siblings."
    )
    return "\n".join(lines)


def check_question_retrievability(
    question: str,
    contract: Optional[RetrievalContract],
    panel: Optional[RetrieverPanel],
    retrievable_keywords: Optional[list[str]] = None,
) -> dict:
    """Vote ``question`` against ``contract`` (Level A + Level B).

    ``retrievable_keywords``: the SAME pre-verified anchor already handed
    to the planner as a prompt hint (see
    ``orqa.agent.utility.keyword_suggestion.suggest_retrievable_keywords``)
    — optional, purely to make a Level A miss's feedback name concrete
    vocabulary instead of only diagnostic numbers (see
    ``_level_a_feedback``). Never itself re-verified against the index
    here; passing ``None`` just drops that extra line.

    Returns ``{"approved": bool, "missing_tables": [alias, ...],
    "missing_facets": [{"table": alias, "facet": label}, ...], "votes":
    dict, "feedback": str}``. No-ops to an automatic pass when ``contract``/
    ``panel`` is unavailable or empty, or the question is blank.
    """
    if panel is None or contract is None or not contract.tables or not question:
        return {
            "approved": True,
            "missing_tables": [],
            "missing_facets": [],
            "votes": {},
            "feedback": "",
        }

    vote = panel.vote(question, contract.gold_ids, contract.top_k, contract.min_agreement)

    missing_tables: list[str] = []
    if not vote["approved"]:
        for table in contract.tables:
            exceeds = any(
                r["family_ranks"].get(table.resource_id, contract.top_k + 1) > contract.top_k
                for r in vote["per_retriever"].values()
            )
            if exceeds:
                missing_tables.append(table.alias)

    missing_by_alias: dict[str, list[Facet]] = {}
    missing_flat: list[dict] = []
    for table in contract.tables:
        missed = missing_facets(question, table.facets)
        if missed:
            missing_by_alias[table.alias] = missed
            missing_flat.extend({"table": table.alias, "facet": f.label} for f in missed)

    approved = vote["approved"] and not missing_by_alias

    feedback_parts = [
        _level_a_feedback(contract, vote, missing_tables, question, retrievable_keywords),
        _level_b_feedback(missing_by_alias),
    ]
    feedback = "\n".join(part for part in feedback_parts if part)

    return {
        "approved": approved,
        "missing_tables": missing_tables,
        "missing_facets": missing_flat,
        "votes": vote,
        "feedback": feedback,
    }
