"""The retrievability contract: build it for a table group, then check a
candidate question against it.

A QUESTION (not a separately-maintained keyword list) is retrievable when a
panel of independent retrievers — semantic, hybrid (RRF) and the lexical gate
— MAJORITY-agree that it finds every gold table within the top K. See
``orqa.benchmark.retrieval_panel``. Nothing else is checked here: no
same-dataset sibling files, no details the question must state.

No-ops to an automatic pass when no panel/contract is available, the same
degradation convention as the pre-planning keyword search
(``keyword_suggestion.suggest_retrievable_keywords``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from ...benchmark.index import tokenize
from ...benchmark.retrieval_panel import RANK_CEILING, RetrieverPanel

CONTRACT_VERSION = 2


@dataclass
class TableContract:
    alias: str
    resource_id: str


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
            "tables": {t.resource_id: t.alias for t in self.tables},
        }


def build_contract(
    plan_tables: list[dict],
    top_k_per_table: int,
    max_top_k: int,
    min_agreement: int,
    single_table_top_k: int = 1,
) -> RetrievalContract:
    """Build the retrieval contract for one fixed table group.

    Args:
        plan_tables: ``[{"alias": str, "resource_id": str}, ...]`` — the
            exact tables this generation run is fixed to.
        top_k_per_table / max_top_k: for a MULTI-table group (more than one
            plan table), ``K = min(max_top_k, top_k_per_table *
            len(plan_tables))`` — the target window. Two or more tables
            cannot all occupy rank 1 of the same search at once, so the
            window widens with table count instead.
        single_table_top_k: the target for a SINGLE-table group
            (``len(plan_tables) <= 1``) — literal rank 1 by default: with
            only one target, a greedy search over the table's own real
            vocabulary reaching rank 1 costs no more than reaching a looser
            rank (see ``orqa.agent.utility.keyword_suggestion``'s module
            docstring), so there is no reason to settle for less.
        min_agreement: How many retrievers must pass.
    """
    n = max(len(plan_tables), 1)
    top_k = single_table_top_k if n == 1 else min(max_top_k, top_k_per_table * n)
    tables = [TableContract(t["alias"], t["resource_id"]) for t in plan_tables]
    return RetrievalContract(top_k=top_k, min_agreement=min_agreement, tables=tables)


# How each voter is named in the pipeline log. The user-facing trio is
# lexical / semantic / hybrid; the internal retriever name stays alongside it
# because a panel can hold two lexical voters (``lexical_question`` next to
# ``llm_keywords``) and the label alone would not tell them apart.
_VOTER_LABELS = {
    "question_terms": "lexical",
    "llm_keywords": "lexical",
    "lexical_question": "lexical",
    "dense_question": "semantic",
    "hybrid_rrf": "hybrid",
}
_VOTER_ORDER = ("lexical", "semantic", "hybrid")


def describe_votes(votes: Optional[dict], contract: Optional[RetrievalContract]) -> Optional[dict]:
    """The retriever panel's vote as plain data for the pipeline log and the
    saved attempt record — ``None`` when there was no vote (no panel or
    contract, or the question never got as far as retrieval).

    ``{"approved", "passes", "need", "top_k", "voters": [{"name", "label",
    "pass", "tables": [{"alias", "rank", "within", "via"}]}]}``. ``rank`` is
    the table's 1-indexed rank on that retriever (``None`` when the retriever
    never returned it), ``within`` whether it is inside the window, and
    ``via`` the OTHER tables whose lexical check reached it (the lexical
    gate's cross-coverage) — empty when its own check did. The lexical
    voter's rows also carry ``anchor``: ``{"terms": [...], "missing": [...]}``,
    the table's verified keywords and which of them the question lacks.
    """
    per_retriever = (votes or {}).get("per_retriever")
    if not per_retriever or contract is None or not contract.tables:
        return None
    top_k = votes.get("top_k", contract.top_k)
    voters = []
    for name, verdict in per_retriever.items():
        covered_by = verdict.get("covered_by") or {}
        rows = []
        for table in contract.tables:
            rank = verdict.get("ranks", {}).get(table.resource_id)
            found = rank is not None and rank <= RANK_CEILING
            own = table.alias in (covered_by.get(table.alias) or [])
            row = {
                "alias": table.alias,
                "rank": rank if found else None,
                "within": found and rank <= top_k,
                "via": [] if own else [a for a in covered_by.get(table.alias, []) if a != table.alias],
            }
            anchor = (verdict.get("anchors") or {}).get(table.alias)
            if anchor:
                row["anchor"] = {"terms": list(anchor["terms"]), "missing": list(anchor["missing"])}
            rows.append(row)
        voters.append({
            "name": name,
            "label": _VOTER_LABELS.get(name, name),
            "pass": bool(verdict.get("pass")),
            "tables": rows,
        })
    voters.sort(key=lambda v: _VOTER_ORDER.index(v["label"]) if v["label"] in _VOTER_ORDER else len(_VOTER_ORDER))
    required = contract.min_agreement
    return {
        "approved": bool(votes.get("approved")),
        "passes": int(votes.get("passes", sum(v["pass"] for v in voters))),
        "need": min(required, len(voters)) if required is not None else len(voters),
        "top_k": top_k,
        "voters": voters,
    }


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
        f"Retrievability: searching this question surfaces "
        f"{', '.join(missing_tables)} on only {vote['passes']} of "
        f"{len(per_retriever)} retrievers within top {contract.top_k} "
        f"(need {min(contract.min_agreement, len(per_retriever))}); "
        f"{', '.join(failing)} did not."
    ]
    for table in contract.tables:
        if table.alias not in missing_tables:
            continue
        ranks = {name: r["ranks"].get(table.resource_id) for name, r in per_retriever.items()}
        lines.append(f"  {table.alias}'s rank per retriever: {ranks}.")
    hint = _anchor_hint(question, retrievable_keywords)
    if hint:
        lines.append(hint)
    return "\n".join(lines)


def _anchor_hint(question: str, retrievable_keywords: Optional[list[str]]) -> str:
    """A retrieval diagnosis says THAT and BY HOW MUCH the question missed, but
    names no actual words, so it leaves the LLM nothing concrete to write
    instead. ``retrievable_keywords`` is the SAME
    pre-verified anchor (see
    ``orqa.agent.utility.keyword_suggestion.suggest_retrievable_keywords``)
    already rendered into the planning prompt as a hint; naming it again HERE,
    against the question that just failed, turns it into a pointed diagnosis:
    either the anchor was never used (say so, and name the exact missing
    terms) or it was used and still isn't enough (say that instead, so the
    writer doesn't just re-paste the same anchor and expect a different
    result)."""
    if not retrievable_keywords:
        return ""
    question_lower = question.lower()
    unused = [kw for kw in retrievable_keywords if kw.lower() not in question_lower]
    if unused:
        return (
            "  A pre-verified keyword anchor exists for this table group "
            f"but is missing from the question: {', '.join(unused)}. Weave "
            "these exact terms into the question's own prose (a term that is a "
            "raw column header, code or abbreviation is never pasted in — "
            "describe it in everyday words)."
        )
    return (
        "  The pre-verified keyword anchor "
        f"({', '.join(retrievable_keywords)}) is already present in "
        "the question, but the combination still misses the required "
        "rank — add more of the table's own title/tag/column "
        "vocabulary alongside it rather than repeating the same terms."
    )


# The lexical gate's name in ``votes["per_retriever"]``. It is ONE voter: one
# verdict aggregating a keyword check per table (``_question_terms_vote``),
# counted next to the panel's semantic and hybrid retrievers.
QUESTION_TERMS = "question_terms"

# The panel's own LLM-keyword retriever searches ONE keyword list extracted
# from the whole question — under AND matching that mixes every table's
# vocabulary and finds nothing (under BM25 it ranks a blend of them, which
# favours no table in particular), and which words an LLM keeps is not
# something the question controls. In every gate check — one table or a
# group — the lexical gate takes its place (the panel still extracts those
# keywords for ``hybrid_rrf``).
_REPLACED_BY_LEXICAL_GATE = "llm_keywords"


def _question_terms_for(question: str, resource_id: str, index) -> list[str]:
    """The question's OWN words that ``resource_id``'s record can match, in
    question order — the keywords a searcher could take from the question to
    look for this table.

    Under AND matching a term the record lacks removes the table, and a term
    it has can only remove its competitors, so this — every question word the
    record holds — is the strongest keyword query the question offers for the
    table: if no query built from the question's words reaches the table,
    none does. Under additive BM25 (``keyword_match: any``) a missing term
    costs nothing, and it is still the natural query — exactly the words that
    can score the table — but not provably the strongest: a word the record
    shares with many others lifts them too. An index that cannot say what a
    record matches on (``searchable_terms``) gets every question word instead.
    """
    tokens = list(dict.fromkeys(t for t in tokenize(question) if len(t) > 1))
    lookup = getattr(index, "searchable_terms", None)
    vocabulary = lookup(resource_id) if callable(lookup) else None
    if vocabulary is None:
        return tokens
    return [t for t in tokens if t in vocabulary]


def _anchor_terms_for(
    anchors: Optional[Union[list[str], dict[str, list[str]]]], alias: str
) -> list[str]:
    """The verified anchor for ``alias``: a ``{alias: [...]}`` dict gives each
    table its own; a plain list is one group anchor shared by every table."""
    if isinstance(anchors, dict):
        return [str(term) for term in anchors.get(alias) or []]
    return [str(term) for term in anchors or []]


def anchor_presence(question: str, terms: list[str]) -> tuple[list[str], list[str]]:
    """``(present, missing)``: which anchor ``terms`` the question contains.

    A term is present when EVERY one of its tokens is among the question's
    tokens — the same ``benchmark.index.tokenize`` the index applies, so
    "present" means exactly "would be a query term for the index". The index
    is bag-of-words, so adjacency is not required. A term with no token at all
    (punctuation) is ignored."""
    question_tokens = set(tokenize(question))
    present: list[str] = []
    missing: list[str] = []
    for term in terms:
        tokens = tokenize(term)
        if not tokens:
            continue
        (present if all(token in question_tokens for token in tokens) else missing).append(term)
    return present, missing


def _question_terms_vote(
    question: str,
    tables: list[TableContract],
    panel: RetrieverPanel,
    top_k: int,
    anchors: Optional[Union[list[str], dict[str, list[str]]]] = None,
) -> dict:
    """The lexical gate: ONE verdict from the keyword checks of every table.

    Each table has up to two checks, both searches on the panel's keyword
    index (the lexical, portal-faithful one):

    1. **Its verified anchor.** ``anchors`` is the keyword set the pre-planning
       search proved surfaces the table (see ``keyword_suggestion``), e.g.
       ``[belfast, ni]``. When the question contains EVERY anchor term, the
       question supports exactly that query — one that already found the table
       — so it is searched as it is: no LLM call and no dependence on which
       other words the question happens to use. A question that holds the
       anchor is therefore retrievable by the lexical gate.
    2. **The question's own words** the table's record can match, in question
       order (``_question_terms_for``) — the strongest query the question
       offers when it does not hold the anchor, so a question that words the
       table differently but still finds it is not rejected for that.

    A table is covered when ANY check reaches it within ``top_k``. A check that
    also surfaces OTHER gold tables in its top ``top_k`` covers them too:
    searching for table ``n`` and finding ``n - 1`` and ``n + 2`` counts as
    finding all three. The gate passes when every table is covered. (That
    mostly matters for tables sharing vocabulary, or when BM25's scoring lets a
    neighbour's shorter query place a table higher than its own.)

    Payload shape as ``RetrieverPanel.vote``'s per-retriever entry, with
    ``ranks`` the BEST rank across all checks, plus ``terms`` (each table's
    question-words query, by alias), ``covered_by`` (for each table, the
    aliases whose check reached it) and ``anchors`` (for each table that has
    one: ``{"terms": [...], "missing": [...]}``, ``missing`` empty when the
    question holds the whole anchor).
    """
    index = panel.keyword_index
    gold_ids = [t.resource_id for t in tables]
    best = panel.ranks([], gold_ids)
    terms_by_alias: dict[str, list[str]] = {}
    anchors_by_alias: dict[str, dict] = {}
    covered_by: dict[str, list[str]] = {t.alias: [] for t in tables}

    for table in tables:
        question_terms = _question_terms_for(question, table.resource_id, index)
        terms_by_alias[table.alias] = question_terms

        anchor = _anchor_terms_for(anchors, table.alias)
        _present, missing = anchor_presence(question, anchor)
        if anchor:
            anchors_by_alias[table.alias] = {"terms": anchor, "missing": missing}

        checks: list[list[str]] = []
        if anchor and not missing:
            checks.append(anchor)
        if question_terms and not (
            checks and set(tokenize(" ".join(question_terms))) == set(tokenize(" ".join(anchor)))
        ):
            checks.append(question_terms)

        for terms in checks:
            ranking = [r.resource_id for r in index.search(terms, top_k=RANK_CEILING)]
            if panel.universe is not None:
                ranking = [rid for rid in ranking if rid in panel.universe]
            if not ranking:
                continue
            found = panel.ranks(ranking, gold_ids)
            for gold in tables:
                rid = gold.resource_id
                best[rid] = min(best[rid], found[rid])
                if found[rid] <= top_k and table.alias not in covered_by[gold.alias]:
                    covered_by[gold.alias].append(table.alias)

    passed = bool(gold_ids) and all(best[rid] <= top_k for rid in gold_ids)
    return {
        "pass": passed,
        "ranks": best,
        "terms": terms_by_alias,
        "anchors": anchors_by_alias,
        "covered_by": covered_by,
    }


def _lexical_gate_notes(
    tables: list[TableContract], missing: list[str], lexical: dict, top_k: int
) -> list[str]:
    """Why the lexical gate did not cover each missing table."""
    lines = []
    for table in tables:
        if table.alias not in missing:
            continue
        if lexical["ranks"][table.resource_id] <= top_k:
            continue
        terms = lexical["terms"].get(table.alias) or []
        if not terms:
            lines.append(
                f"  {table.alias}: the question shares no word with its record, so "
                f"no keyword search built from the question can find it."
            )
        else:
            lines.append(
                f"  {table.alias}: no keyword check reaches it within the top "
                f"{top_k} — its own words in the question ({', '.join(terms)}) "
                f"are not enough to surface it."
            )
    return lines


def _check_per_table(
    question: str,
    contract: RetrievalContract,
    panel: RetrieverPanel,
    retrievable_keywords: Optional[Union[list[str], dict[str, list[str]]]],
    per_table_top_k: int,
) -> dict:
    """The retrieval vote with the lexical gate — for a single table, and for
    a multi-table group judged per table.

    The whole-contract vote asks that EVERY gold table sit inside ONE
    ranking's window — a single search that surfaces the whole group. That
    is the wrong bar when the searcher refines its search step by step (one
    search per table or sub-topic), and under portal AND matching no single
    keyword query can surface tables that share no vocabulary anyway. So the
    window is ``per_table_top_k`` for every table, and the lexical retriever
    is replaced by the lexical gate (``_question_terms_vote``): a keyword
    check per table — on the table's verified anchor when the question holds
    it — together ONE vote. For a single table the same gate replaces the
    LLM keyword extraction too, so its lexical vote is deterministic.

    That vote sits in the panel's usual majority with the semantic and hybrid
    retrievers (``contract.min_agreement``), each of which passes when every
    table is within ``per_table_top_k`` of its ranking of the whole question.
    The question is ranked once, whatever the table count.
    """
    rankings = {
        name: ranking
        for name, ranking in panel.rank(question).items()
        if name != _REPLACED_BY_LEXICAL_GATE
    }
    lexical = _question_terms_vote(
        question, contract.tables, panel, per_table_top_k, retrievable_keywords
    )
    vote = panel.vote(
        question,
        contract.gold_ids,
        per_table_top_k,
        contract.min_agreement,
        rankings=rankings,
        extra={QUESTION_TERMS: lexical},
    )

    missing_tables: list[str] = []
    level_a_parts: list[str] = []
    if not vote["approved"]:
        for table in contract.tables:
            exceeds = any(
                r["ranks"].get(table.resource_id, per_table_top_k + 1) > per_table_top_k
                for r in vote["per_retriever"].values()
            )
            if exceeds:
                missing_tables.append(table.alias)
        window = RetrievalContract(
            top_k=per_table_top_k, min_agreement=contract.min_agreement, tables=contract.tables
        )
        level_a_parts.append(
            _level_a_feedback(window, vote, missing_tables, question, None)
        )
        level_a_parts.extend(
            _lexical_gate_notes(contract.tables, missing_tables, lexical, per_table_top_k)
        )
        for table in contract.tables:
            if table.alias not in missing_tables:
                continue
            keywords = (
                retrievable_keywords.get(table.alias)
                if isinstance(retrievable_keywords, dict)
                else retrievable_keywords
            )
            hint = _anchor_hint(question, keywords)
            if hint:
                level_a_parts.append(f"  ({table.alias})\n{hint}")

    feedback = "\n".join(part for part in level_a_parts if part)
    return {
        "approved": vote["approved"],
        "missing_tables": missing_tables,
        "votes": {**vote, "top_k": per_table_top_k},
        "feedback": feedback,
    }


def check_question_retrievability(
    question: str,
    contract: Optional[RetrievalContract],
    panel: Optional[RetrieverPanel],
    retrievable_keywords: Optional[Union[list[str], dict[str, list[str]]]] = None,
    per_table_top_k: Optional[int] = None,
) -> dict:
    """Vote ``question`` against ``contract``.

    ``retrievable_keywords``: the SAME pre-verified anchor the question was
    written from — handed to the question writer as the vocabulary to build
    it around (see
    ``orqa.agent.utility.keyword_suggestion.suggest_retrievable_keywords``)
    — optional. It is the first check of the lexical gate: a question that
    holds every term of a table's anchor is searched with exactly that query
    (see ``_question_terms_vote``), which is how a question that carries e.g.
    ``[belfast, ni]`` passes its lexical vote. It also makes a miss's feedback
    name concrete vocabulary instead of only diagnostic numbers (see
    ``_level_a_feedback``). The anchor is never trusted blindly: the rank the
    search actually returns decides. Passing ``None`` drops both uses. A
    ``{alias: [...]}`` dict gives each table its own anchor; a plain list is
    one anchor for the group.

    A single-table contract is always voted with the lexical gate, in the
    contract's own window: the question passes its lexical vote when it holds
    the table's verified keywords (or its own words find the table), counted
    as ONE vote next to the semantic and hybrid retrievers.

    ``per_table_top_k``: when set AND the contract has more than one table,
    every table must be within this window (instead of ``contract.top_k``
    for the group) and the lexical retriever becomes the lexical gate the
    same way (see ``_check_per_table``). ``None`` — the default — keeps the
    whole-contract vote for a multi-table group.

    Returns ``{"approved": bool, "missing_tables": [alias, ...], "votes":
    dict, "feedback": str}``. No-ops to an automatic pass when ``contract``/
    ``panel`` is unavailable or empty, or the question is blank.
    """
    if panel is None or contract is None or not contract.tables or not question:
        return {
            "approved": True,
            "missing_tables": [],
            "votes": {},
            "feedback": "",
        }

    if len(contract.tables) == 1:
        # One table: the contract's own window, and the same deterministic
        # lexical gate a multi-table group gets — the question either holds
        # the table's verified keywords or it does not.
        return _check_per_table(
            question, contract, panel, retrievable_keywords, contract.top_k
        )
    if per_table_top_k is not None:
        return _check_per_table(
            question, contract, panel, retrievable_keywords, per_table_top_k
        )

    if isinstance(retrievable_keywords, dict):
        # Whole-contract vote with an alias-keyed anchor: one flat list.
        retrievable_keywords = [
            kw for kws in retrievable_keywords.values() for kw in kws
        ] or None

    vote = panel.vote(question, contract.gold_ids, contract.top_k, contract.min_agreement)

    missing_tables: list[str] = []
    if not vote["approved"]:
        for table in contract.tables:
            exceeds = any(
                r["ranks"].get(table.resource_id, contract.top_k + 1) > contract.top_k
                for r in vote["per_retriever"].values()
            )
            if exceeds:
                missing_tables.append(table.alias)

    return {
        "approved": vote["approved"],
        "missing_tables": missing_tables,
        "votes": vote,
        "feedback": _level_a_feedback(
            contract, vote, missing_tables, question, retrievable_keywords
        ),
    }
