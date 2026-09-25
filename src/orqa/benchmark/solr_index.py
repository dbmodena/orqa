"""Reverse index backed by a LIVE Solr core — the retriever LakeGen queries.

``orqa.benchmark.index.PortalIndexView`` can only IMITATE the portal's keyword
search, and the imitation drifts: measured against the live UK core, a quarter
of the anchors it called "rank 1" were not, and some matched nothing at all.
The reasons are structural, not tunable:

* the core is built from a different representation of the metadata (15,205
  per-file documents, not the 27,390 normalized records);
* its ``title`` field holds the FILE's name (the normalized ``resource_name``)
  and the dataset title is not indexed at all;
* it searches whatever ``qf`` the running core is configured with — here
  including the column fields — which is not what the repository's
  ``conf/solr/<portal>`` says;
* AND is not a schema default: the client asks for ``q.op=AND`` explicitly.

So this class does not emulate any of that. ``search`` sends the keywords to
the core as an edismax ``q.op=AND`` query (``solr.client.LocalSolrClient``,
the repository's own client) and ranks by what Solr returns. A query costs
about a millisecond, which is why the anchor search can afford to ask Solr
for every trial instead of approximating it.

Everything that is NOT a search is served from the normalized records
(``get``, file paths, families): the retrievability contract's facts about a
file — its period, its qualifiers — live there, not in the Solr document.
Where the anchor search needs the vocabulary a document really offers to a
query, ``candidate_record`` and ``searchable_terms`` read it from the Solr
document itself.

Same duck-typed surface as ``DatasetIndex`` / ``ESDatasetIndex`` (``search``,
``get``), plus ``search_many`` so a batch of trials runs concurrently.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional

from .index import DatasetIndex, SearchResult, tokenize

logger = logging.getLogger(__name__)

DEFAULT_SOLR_URL = "http://localhost:8983/solr"

# What a search hit needs (the embedding vector is deliberately not fetched).
_SEARCH_FL = "resource_id,dataset_id,title,publisher,tags,dataset_url,score"
# The text a query can match on a document, for candidate vocabulary.
_DOC_FL = (
    "resource_id,dataset_id,title,description,publisher,tags,"
    "columns.name,columns.label,columns.description,columns.type"
)


def _first(value: Any, default: Any = "") -> Any:
    """Solr returns a multi-valued field as a list even when it holds one value."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


class SolrDatasetIndex:
    """Keyword search against a live Solr core, plus the normalized records
    for everything that is not a search.

    Args:
        client: ``LocalSolrClient``-shaped: ``select(tokens, q_op=..., **params)``.
        records: The normalized-metadata ``DatasetIndex`` — source of ``get``,
            file paths and ``source``.
        match: ``"all"`` sends ``q.op=AND`` (how the portal is queried);
            ``"any"`` sends ``q.op=OR``.
        max_workers: Concurrency of :meth:`search_many`.
    """

    # keyword_suggestion: candidate terms must be ones this index can match.
    restricts_candidates = True

    def __init__(
        self,
        client: Any,
        records: DatasetIndex,
        match: str = "all",
        max_workers: int = 8,
    ):
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        self._client = client
        self._records = records
        self.match = match
        self._q_op = "AND" if match == "all" else "OR"
        self._max_workers = max(1, int(max_workers))
        self._docs: dict[str, Optional[dict]] = {}
        self._terms: dict[str, Optional[frozenset[str]]] = {}
        self._resource_ids: Optional[frozenset[str]] = None

    @classmethod
    def build(
        cls,
        records: DatasetIndex,
        core: str,
        base_url: Optional[str] = None,
        match: str = "all",
        timeout: float = 30.0,
    ) -> "SolrDatasetIndex":
        """Index over ``core`` at ``base_url`` (else ``$SOLR_URL``, else the
        local default — the convention ``solr.solr.Solr`` already uses)."""
        from solr.client import LocalSolrClient

        url = base_url or os.environ.get("SOLR_URL") or DEFAULT_SOLR_URL
        return cls(LocalSolrClient(core, base_url=url, timeout=timeout), records, match=match)

    def ping(self) -> int:
        """Number of documents in the core; raises if Solr is unreachable."""
        response = self._client.select(["*:*"], q_op="AND", rows=0)["response"]
        return int(response["numFound"])

    # ------------------------------------------------------------------ search

    @staticmethod
    def _terms_of(keywords: str | Iterable[str]) -> list[str]:
        """Distinct query tokens, in order. Tokenizing with the index's own
        tokenizer leaves lowercase alphanumerics only, so no Solr query
        syntax (operators, ``field:`` prefixes, quotes) can be smuggled in by
        a keyword, and the uppercase words Solr reads as operators cannot
        occur."""
        if isinstance(keywords, str):
            tokens = tokenize(keywords)
        else:
            tokens = [t for kw in keywords for t in tokenize(kw)]
        return list(dict.fromkeys(tokens))

    def _result(self, doc: dict) -> SearchResult:
        resource_id = doc["resource_id"]
        filepath = self._records.dataset_filepath(resource_id)
        return SearchResult(
            resource_id=resource_id,
            dataset_id=_first(doc.get("dataset_id"), resource_id),
            title=_first(doc.get("title")),
            publisher=_first(doc.get("publisher"), None),
            tags=doc.get("tags") or [],
            score=float(_first(doc.get("score"), 0.0)),
            matched_terms=[],
            csv_path=str(filepath),
            csv_exists=filepath.exists(),
            dataset_url=_first(doc.get("dataset_url"), None),
            source=self._records.source,
        )

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        """Rank the core's documents against ``keywords`` — Solr's own ranking,
        ``q.op=AND`` unless ``match="any"``. No keywords, no query."""
        terms = self._terms_of(keywords)
        if not terms:
            return []
        response = self._client.select(
            terms, q_op=self._q_op, rows=max(1, int(top_k)), fl=_SEARCH_FL
        )["response"]
        results = [self._result(doc) for doc in response.get("docs", [])]
        if only_available:
            results = [r for r in results if r.csv_exists]
        return results[:top_k]

    def search_many(
        self,
        queries: list[str | Iterable[str]],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[list[SearchResult]]:
        """``search`` for each query, concurrently, in order — what
        ``keyword_suggestion`` uses to evaluate a round of trials."""
        if len(queries) <= 1 or self._max_workers == 1:
            return [self.search(q, top_k, only_available) for q in queries]
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(lambda q: self.search(q, top_k, only_available), queries))

    # ------------------------------------------------- the Solr document itself

    def _doc(self, resource_id: str) -> Optional[dict]:
        if resource_id not in self._docs:
            response = self._client.select(
                ["*:*"], q_op="AND", rows=1, fl=_DOC_FL,
                fq=f'resource_id:"{resource_id}"',
            )["response"]
            docs = response.get("docs") or []
            self._docs[resource_id] = docs[0] if docs else None
        return self._docs[resource_id]

    def candidate_record(self, resource_id: str) -> Optional[dict]:
        """The document as the Solr core holds it, shaped like a normalized
        record so ``keyword_suggestion`` can draw candidate terms from it:
        ``title`` is the FILE's name (as Solr indexes it), ``resource_name``
        is empty, the columns are the core's. ``None`` if the core has no
        such document — the caller then falls back to ``get``."""
        doc = self._doc(resource_id)
        if doc is None:
            return None
        return {
            "resource_id": resource_id,
            "dataset_id": _first(doc.get("dataset_id"), resource_id),
            "title": _first(doc.get("title")),
            "resource_name": "",
            "tags": list(doc.get("tags") or []),
            "publisher": _first(doc.get("publisher")),
            "description": _first(doc.get("description")),
            "columns": doc.get("columns") or [],
        }

    def searchable_terms(self, resource_id: str) -> Optional[frozenset[str]]:
        """The tokens a query term can match on this document: its title,
        description, publisher, tags and columns. The ids, ``source`` and
        ``format`` fields are searched too but are never a useful term (an id
        fragment, or a word every document shares), so they are left out.
        ``None`` if the core has no such document."""
        if resource_id not in self._terms:
            record = self.candidate_record(resource_id)
            if record is None:
                self._terms[resource_id] = None
            else:
                text = [record["title"], record["description"], record["publisher"], *record["tags"]]
                for column in record["columns"]:
                    text.extend(str(column.get(k) or "") for k in ("name", "label", "description", "type"))
                self._terms[resource_id] = frozenset(t for chunk in text for t in tokenize(str(chunk)))
        return self._terms[resource_id]

    def resource_ids(self) -> frozenset[str]:
        """Every resource id the core holds — the universe a search can
        return. Fetched once; a retriever that ranks OTHER records (the local
        embedding cache covers the whole normalized metadata, not just the
        core) can be restricted to it so it competes on the same field."""
        if self._resource_ids is None:
            total = self.ping()
            response = self._client.select(
                ["*:*"], q_op="AND", rows=max(1, total), fl="resource_id"
            )["response"]
            self._resource_ids = frozenset(d["resource_id"] for d in response.get("docs", []))
        return self._resource_ids

    # ------------------------------------------ everything else: the records

    def get(self, resource_id: str) -> Optional[dict]:
        return self._records.get(resource_id)

    def dataset_filepath(self, resource_id: str):
        return self._records.dataset_filepath(resource_id)

    def __len__(self) -> int:
        return len(self._records)

    def __getattr__(self, name: str) -> Any:
        # datasets_path / datasets_format / source: pass-through.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._records, name)
