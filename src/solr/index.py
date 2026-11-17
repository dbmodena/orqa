"""The retrieval index of a portal with a LakeGen Solr core: the live core
LakeGen's benchmark runs query, asked the way LakeGen asks it.

OrQA builds the benchmark LakeGen is scored on, so a question is only fair if
LakeGen's own retrievers can find its tables. Imitating them drifts — the
Elasticsearch imitation this replaces matched Solr's AND result set for about
a tenth of the queries — because the cores' analysis cannot be reproduced
in-process (English Snowball stemming before ASCII folding on French, Italian
and Spanish text, word-delimiter splits, 2-10-gram title substrings, the stock
synonyms, a catch-all ``text`` field). So every decision here is Solr's:

* **Keyword search** (:meth:`SolrDatasetIndex.search`) — LakeGen's
  ``KeywordRetriever``: GET ``/<core>/select``, ``defType=edismax``,
  ``q.op=AND`` (``match="any"``: OR), the core's own ``qf`` (none is sent),
  ``q`` = the keywords as written, joined by spaces. Not accent-folded nor
  re-tokenized: on paris_v2 'écoles élémentaires' finds 4 tables and the
  folded 'ecoles elementaires' 3, with a different top 1. See
  :func:`query_words` for the only rewriting (query syntax becomes literal).
* **Dense search** (:meth:`SolrDatasetIndex.semantic_search_many`) —
  LakeGen's ``SemanticRetriever``: the question embedded like LakeGen's
  (``orqa.agent.llm_client.EmbeddingClient.QueryEmbedder``: NFKC,
  cohere.embed-v4.0, 1024 dimensions, SEARCH_QUERY), then
  ``{!knn f=table_embedding topK=N}`` POSTed to ``/<core>/query`` with
  LakeGen's two ``fq``.
* **Which words a document matches** (:meth:`SolrDatasetIndex.terms_matching`)
  — asked of Solr, term by term, so the lexical gate keeps exactly the
  question words Solr would match ('tree' matches 119 nyc tables through the
  title n-grams of 'street', 'trees' 48).

Nothing else is needed from the normalized metadata: the core's documents
carry title, description, publisher, tags and columns, and a hit's local
file comes from a directory listing (``local_files``). A core document with
no local file is still a competitor (it is in :meth:`resource_ids`); a local
file the core lacks can never be retrieved.

Read-only: the instance is shared with LakeGen (see ``solr.client``).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from orqa.utils import dataset_id_to_resource_id

from .index import SearchResult, tokenize

logger = logging.getLogger(__name__)

DEFAULT_SOLR_URL = "http://localhost:8983/solr"

# LakeGen's SemanticRetriever filters: only vectors of this representation and
# model are comparable with a cohere.embed-v4.0 metadata-v1 question vector.
LAKEGEN_KNN_FILTERS = (
    'representation_version:"metadata-v1"',
    'embedding_model:"cohere.embed-v4.0"',
)

# Words per keyword query. maxClauseCount (1024) is spent per word per qf
# field (19 on every LakeGen core): AND queries failed with HTTP 500 at 55-59
# words, and the anchor search's seed query sends a table's whole vocabulary.
MAX_QUERY_WORDS = 40

# What a hit needs. Never `fl=*`: the stored table_embedding made LakeGen's
# rows=100 response 1.5 MB; these display fields cost no measurable time
# (uk, rows=500: 30.4 ms vs 30.5 ms for resource_id,score alone).
_SEARCH_FL = "resource_id,dataset_id,title,publisher,tags,dataset_url,score"
# The text a query can match on a document, for candidate vocabulary.
_DOC_FL = (
    "resource_id,dataset_id,title,description,publisher,tags,"
    "columns.name,columns.label,columns.description,columns.type"
)
# terms_matching asks this many terms per request (one facet.query each).
_TERMS_PER_REQUEST = 40
# resource_ids() pages through the core with cursorMark (uk: 4 pages).
_ID_PAGE = 5000

# Edismax reads these as query syntax: '?' and '*' are wildcards that bypass
# stemming ('trees?' found 3 nyc tables, 'trees' 48), ':' a field prefix
# ('staff:junior' 0 hits on uk, the words 874), a leading '-' a NOT, quotes a
# phrase. Escaped, each is plain text to the analyzer; inner hyphens and
# slashes ('covid-19', 'built-up', '2019/20') give identical hits either way.
_QUERY_SYNTAX = re.compile(r'([+\-!(){}\[\]^"~*?:\\/&|])')
_OPERATORS = {"AND", "OR", "NOT"}


class SolrUnavailable(RuntimeError):
    """The configured core cannot serve retrieval (down, empty, unknown, or
    its KNN returns nothing) — raised at load time instead of degrading."""


def query_words(keywords: str | Iterable[str]) -> list[str]:
    """The words of a keyword query, as LakeGen sends them: each keyword
    stripped and whitespace-collapsed, blank and case-insensitive repeated
    keywords dropped (LakeGen's intent normalizer), the rest kept verbatim —
    case, accents, hyphens and apostrophes are the core analyzer's business.

    The one rewrite: query syntax is made literal (see ``_QUERY_SYNTAX``;
    an upper-case AND/OR/NOT is lower-cased to the plain word), so a keyword
    or a question can only ever mean its words. LakeGen joins raw keywords,
    which is identical for every query it has logged (inner hyphens only).
    A word with no letter or digit is dropped (no analyzer makes a token of
    it), and the query is capped at :data:`MAX_QUERY_WORDS` words.
    """
    items = [keywords] if isinstance(keywords, str) else list(keywords)
    seen: set[str] = set()
    words: list[str] = []
    for item in items:
        keyword = " ".join(str(item).split())
        if not keyword or keyword.casefold() in seen:
            continue
        seen.add(keyword.casefold())
        for word in keyword.split(" "):
            if not any(c.isalnum() for c in word):
                continue
            word = word.lower() if word in _OPERATORS else word
            words.append(_QUERY_SYNTAX.sub(r"\\\1", word))
    return words[:MAX_QUERY_WORDS]


def local_files_by_resource_id(datasets_path: Path, datasets_format: str) -> dict[str, Path]:
    """``{resource_id: file}`` for the portal's local tables.

    The id rule is ``orqa.utils.dataset_id_to_resource_id`` — the one the
    rest of OrQA applies to a file stem: CKAN files are
    ``<dataset_id>___<resource_id>`` (uk, valencia) and every core's
    ``resource_id`` is the part after the last ``___``; other portals' stems
    are the id itself. Only ``*.<format>`` files count, so a partial download
    (nyc's ``*.csv.tmp``) is not a table.
    """
    files: dict[str, Path] = {}
    clashes: list[str] = []
    for path in sorted(Path(datasets_path).glob(f"*.{datasets_format}")):
        resource_id = dataset_id_to_resource_id(path.stem)
        if resource_id in files:
            clashes.append(path.name)
            continue
        files[resource_id] = path
    if clashes:
        logger.warning(
            "%d local files share a resource id with another file and are ignored (e.g. %s).",
            len(clashes), clashes[0],
        )
    return files


def _first(value: Any, default: Any = "") -> Any:
    """Solr returns a multi-valued field as a list even when it holds one value."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _quoted(field: str, value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'{field}:"{escaped}"'


class _Shared:
    """Caches of one core, shared by every :meth:`SolrDatasetIndex.with_match`
    sibling: the documents and the id universe do not depend on q.op."""

    def __init__(self) -> None:
        self.docs: dict[str, Optional[dict]] = {}
        self.terms: dict[str, Optional[frozenset[str]]] = {}
        self.term_hits: dict[tuple[str, str, str], bool] = {}
        self.resource_ids: Optional[frozenset[str]] = None
        self.count: Optional[int] = None


class SolrDatasetIndex:
    """Keyword and KNN retrieval over one live Solr core (module docstring).

    Args:
        client: ``solr.client.LocalSolrClient``-shaped: ``select(tokens, *,
            core=, q_op=, **params)`` (GET) and ``query_json(core, body)``
            (POST to ``/query``).
        core: The core, e.g. ``uk`` or ``paris_v2``.
        match: ``"all"`` sends ``q.op=AND`` (LakeGen); ``"any"`` ``q.op=OR``.
        local_files: ``{resource_id: local file}``
            (:func:`local_files_by_resource_id`).
        source: The portal's source (``ckan``/``socrata``/``ods``), copied
            onto every result.
        datasets_path, datasets_format: Where the local tables live.
        vector_field: The core's ``DenseVectorField``.
        knn_filters: ``fq`` of every KNN query (LakeGen's two).
        embedder: ``texts -> (n, d) array`` of query vectors
            (``QueryEmbedder``); only :meth:`semantic_search_many` needs it.
    """

    # keyword_suggestion: candidate terms must be ones this index can match.
    restricts_candidates = True
    # Threads of search_many / semantic_search_many; Solr answers in ~1 ms.
    max_workers = 8

    def __init__(
        self,
        client: Any,
        core: str,
        *,
        match: str = "all",
        local_files: Mapping[str, Path],
        source: str,
        datasets_path: Path,
        datasets_format: str,
        vector_field: str = "table_embedding",
        knn_filters: Sequence[str] = LAKEGEN_KNN_FILTERS,
        embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
    ):
        if not vector_field or not vector_field.replace("_", "").isalnum():
            raise ValueError(f"vector_field must be a plain Solr field name, got {vector_field!r}")
        self._client = client
        self.core = core.strip()
        self._set_match(match)
        self._local_files = dict(local_files)
        self.source = source
        self.datasets_path = Path(datasets_path)
        self.datasets_format = datasets_format
        self.vector_field = vector_field
        self.knn_filters = tuple(knn_filters)
        self._embedder = embedder
        self._check_embedding_model()
        self._shared = _Shared()

    def _set_match(self, match: str) -> None:
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        self.match = match
        self._q_op = "AND" if match == "all" else "OR"

    def _check_embedding_model(self) -> None:
        """A query embedded by another model than the stored vectors' ranks
        noise without any error, so a named model must agree with the
        ``embedding_model`` filter."""
        model = getattr(self._embedder, "model", None)
        if not model:
            return
        for spec in self.knn_filters:
            field, _, value = spec.partition(":")
            if field.strip() == "embedding_model":
                expected = value.strip().strip('"')
                if str(model).rsplit("/", 1)[-1] != expected:
                    raise ValueError(
                        f"the query embedder uses {model!r}, but the core's vectors are "
                        f"filtered to {expected!r}: set litellm.yaml embedding_model to it"
                    )

    @classmethod
    def from_config(cls, cfg: Any, embedder: Optional[Callable[[List[str]], np.ndarray]] = None) -> "SolrDatasetIndex":
        """The index ``tasks.search.solr`` describes. The URL is
        ``search.solr.url``, else ``$SOLR_URL``, else the local default;
        ``match`` is ``retrieval_contract.keyword_match``. Without an
        ``embedder``, a ``QueryEmbedder`` over litellm.yaml's
        ``embedding_model`` is built (no provider call until a question is
        ranked). Makes no Solr request: call :meth:`ping` and
        :meth:`check_knn` to fail early."""
        from solr.client import LocalSolrClient

        solr = cfg.search.solr
        url = solr.url or os.environ.get("SOLR_URL") or DEFAULT_SOLR_URL
        if embedder is None:
            from orqa.agent.llm_client.EmbeddingClient import EmbeddingClient, QueryEmbedder

            embedder = QueryEmbedder(
                EmbeddingClient(
                    cfg.llm_config_path / "litellm.yaml",
                    batch_size=cfg.candidates_discovery.embedding_batch_size,
                ),
                dimensions=solr.embedding_dimensions,
                cache_size=cfg.search.query_embedding_cache_size,
            )
        return cls(
            LocalSolrClient(base_url=url, timeout=solr.timeout, retries=solr.retries),
            solr.core,
            match=cfg.search.retrieval_contract.keyword_match,
            local_files=local_files_by_resource_id(cfg.datasets_path, cfg.datasets_format),
            source=cfg.source,
            datasets_path=cfg.datasets_path,
            datasets_format=cfg.datasets_format,
            vector_field=solr.vector_field,
            knn_filters=solr.knn_filters,
            embedder=embedder,
        )

    def with_match(self, match: str) -> "SolrDatasetIndex":
        """The same core under another q.op (``"any"`` for the panel's
        ``lexical_question``), sharing the client, the embedder and every
        cache."""
        if match == self.match:
            return self
        sibling = copy.copy(self)
        sibling._set_match(match)
        return sibling

    @property
    def lexical_index(self) -> "SolrDatasetIndex":
        """``RetrieverPanel`` reads its question-text index from here."""
        return self

    # ------------------------------------------------------------- health

    def ping(self) -> int:
        """The core's document count; :class:`SolrUnavailable` when Solr is
        down, the core unknown, or empty."""
        try:
            response = self._client.select(["*:*"], core=self.core, indent=False, rows=0)
            count = int(response["response"]["numFound"])
        except Exception as exc:
            raise SolrUnavailable(
                f"Solr core {self.core!r} is not answering ({exc}). Start Solr, or fix "
                f"tasks.search.solr.core / url ($SOLR_URL)."
            ) from exc
        if count == 0:
            raise SolrUnavailable(f"Solr core {self.core!r} holds no documents.")
        self._shared.count = count
        return count

    def check_knn(self) -> None:
        """KNN with one stored vector must return hits. Through the wrong
        handler, or with filters no document carries, Solr answers KNN with
        0 hits and no error, which would make the dense retriever silently
        rank nothing (and the gate silently stricter)."""
        try:
            docs = self._client.select(
                ["*:*"], core=self.core, indent=False, rows=1, sort="resource_id asc",
                fl=f"resource_id,{self.vector_field}", fq=list(self.knn_filters),
            )["response"]["docs"]
        except Exception as exc:
            raise SolrUnavailable(f"Solr core {self.core!r}: cannot read a stored vector ({exc}).") from exc
        vector = docs[0].get(self.vector_field) if docs else None
        if not vector:
            raise SolrUnavailable(
                f"Solr core {self.core!r} has no stored {self.vector_field!r} under the KNN "
                f"filters {list(self.knn_filters)}: dense retrieval would rank nothing."
            )
        dimensions = getattr(self._embedder, "dimensions", None)
        if dimensions is not None and len(vector) != dimensions:
            raise SolrUnavailable(
                f"Solr core {self.core!r} stores {len(vector)}-d {self.vector_field!r}, but the "
                f"query embedder makes {dimensions}-d vectors (tasks.search.solr.embedding_dimensions)."
            )
        try:
            hits = self._knn(vector, 10)
        except Exception as exc:
            raise SolrUnavailable(f"Solr core {self.core!r}: KNN query failed ({exc}).") from exc
        if not hits:
            raise SolrUnavailable(
                f"Solr core {self.core!r}: KNN with the stored vector of "
                f"{docs[0].get('resource_id')!r} returned no hits."
            )

    # ------------------------------------------------------------- search

    def _result(self, doc: dict) -> SearchResult:
        resource_id = doc["resource_id"]
        path = self._local_files.get(resource_id)
        return SearchResult(
            resource_id=resource_id,
            dataset_id=_first(doc.get("dataset_id"), resource_id),
            title=_first(doc.get("title")),
            publisher=_first(doc.get("publisher"), None),
            tags=doc.get("tags") or [],
            score=float(_first(doc.get("score"), 0.0)),
            matched_terms=[],
            csv_path=str(path) if path is not None else "",
            csv_exists=path is not None,
            dataset_url=_first(doc.get("dataset_url"), None),
            source=self.source,
        )

    def _map(self, fn: Callable[[Any], Any], items: list) -> list:
        if len(items) <= 1 or self.max_workers <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            return list(pool.map(fn, items))

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        """The core's ranking of ``keywords`` (:func:`query_words`), top
        ``top_k``; ``only_available`` then drops hits with no local file.
        No words, no query."""
        words = query_words(keywords)
        if not words:
            return []
        response = self._client.select(
            words, core=self.core, q_op=self._q_op, indent=False,
            rows=max(1, int(top_k)), fl=_SEARCH_FL,
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
        """:meth:`search` for each query, concurrently, in order — how
        ``keyword_suggestion`` evaluates a round of trials."""
        return self._map(lambda q: self.search(q, top_k, only_available), list(queries))

    def _knn(self, vector: Sequence[float], top_k: int) -> list[SearchResult]:
        values = [float(v) for v in vector]
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError("a KNN query vector must be non-empty and finite")
        k = max(1, int(top_k))
        body = {
            # Compact JSON, as LakeGen serializes it.
            "query": f"{{!knn f={self.vector_field} topK={k}}}"
            + json.dumps(values, separators=(",", ":")),
            "filter": list(self.knn_filters),
            "fields": _SEARCH_FL,
            "limit": k,
        }
        docs = self._client.query_json(self.core, body)["response"].get("docs", [])
        return [self._result(doc) for doc in docs]

    def semantic_search_many(self, questions: List[str], top_k: int) -> List[List[SearchResult]]:
        """LakeGen's dense branch for each question, in order: the
        question's query vector, then Solr KNN (``topK = rows = top_k``,
        cosine — Solr scores ``(1 + cos) / 2``)."""
        questions = list(questions)
        if not questions:
            return []
        if self._embedder is None:
            raise RuntimeError(f"SolrDatasetIndex({self.core!r}) has no query embedder for KNN")
        vectors = self._embedder(questions)
        return self._map(lambda vector: self._knn(vector, top_k), list(vectors))

    # ----------------------------------------- the Solr documents themselves

    def _doc(self, resource_id: str) -> Optional[dict]:
        docs = self._shared.docs
        if resource_id not in docs:
            response = self._client.select(
                ["*:*"], core=self.core, indent=False, rows=1, fl=_DOC_FL,
                fq=_quoted("resource_id", resource_id),
            )["response"]
            found = response.get("docs") or []
            docs[resource_id] = found[0] if found else None
        return docs[resource_id]

    def terms_matching(self, resource_id: str, terms: Iterable[str]) -> frozenset[str]:
        """Which of ``terms``, each searched ALONE exactly as :meth:`search`
        would send it (same words, same q.op, the core's qf), match this
        document — Solr's answer, not a tokenizer's guess. Stemming, word
        splits and title n-grams make the two differ: 2.2% of paris_v2's
        locally matchable candidate terms never match their own document, and
        valencia_v2 matches 17 terms the local tokens miss.

        One request per :data:`_TERMS_PER_REQUEST` terms: ``q=*:*``,
        ``fq=resource_id``, one ``facet.query={!edismax q.op=.. v=$tN}`` per
        term (count 1 = match). Answers are cached per (q.op, document, term).
        A document the core lacks matches nothing.
        """
        asked = list(dict.fromkeys(str(t) for t in terms))
        if not asked or self._doc(resource_id) is None:
            return frozenset()
        hits = self._shared.term_hits
        todo = [t for t in asked if (self._q_op, resource_id, t) not in hits]
        for start in range(0, len(todo), _TERMS_PER_REQUEST):
            queries: dict[str, str] = {}
            params: dict[str, str] = {}
            for i, term in enumerate(todo[start : start + _TERMS_PER_REQUEST]):
                words = query_words([term])
                if not words:
                    hits[(self._q_op, resource_id, term)] = False
                    continue
                queries[f"{{!edismax q.op={self._q_op} v=$t{i}}}"] = term
                params[f"t{i}"] = " ".join(words)
            if not queries:
                continue
            counts = self._client.select(
                ["*:*"], core=self.core, q_op=self._q_op, indent=False, rows=0,
                fq=_quoted("resource_id", resource_id), facet="true",
                **{"facet.query": list(queries)}, **params,
            )["facet_counts"]["facet_queries"]
            for key, term in queries.items():
                hits[(self._q_op, resource_id, term)] = int(counts.get(key, 0)) > 0
        return frozenset(t for t in asked if hits[(self._q_op, resource_id, t)])

    def candidate_record(self, resource_id: str) -> Optional[dict]:
        """The document as the core holds it, shaped like a normalized
        record so ``keyword_suggestion`` can draw candidate terms from it.
        ``title`` is what the core indexes as title (on ``uk`` the FILE's
        name; the dataset title is not indexed there), ``resource_name`` is
        empty, the columns are the core's. ``None`` if the core has no such
        document."""
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
        """Candidate vocabulary: the local tokens (``index.tokenize``) of the
        document's stored title, description, publisher, tags and columns.
        The ids, ``source`` and ``format`` are searched too but are never a
        useful term (an id fragment, or a word every document shares), so
        they are left out. Whether Solr matches a token is
        :meth:`terms_matching`'s question. ``None`` if the core has no such
        document."""
        cache = self._shared.terms
        if resource_id not in cache:
            record = self.candidate_record(resource_id)
            if record is None:
                cache[resource_id] = None
            else:
                text = [record["title"], record["description"], record["publisher"], *record["tags"]]
                for column in record["columns"]:
                    text.extend(str(column.get(k) or "") for k in ("name", "label", "description", "type"))
                cache[resource_id] = frozenset(t for chunk in text for t in tokenize(str(chunk)))
        return cache[resource_id]

    def resource_ids(self) -> frozenset[str]:
        """Every document id of the core — the universe LakeGen retrieves
        from, so a retriever's ranking is restricted to it. Paged with
        cursorMark and fetched once."""
        if self._shared.resource_ids is None:
            ids: list[str] = []
            cursor = "*"
            while True:
                payload = self._client.select(
                    ["*:*"], core=self.core, indent=False, rows=_ID_PAGE, fl="resource_id",
                    sort="resource_id asc", cursorMark=cursor,
                )
                docs = payload["response"].get("docs", [])
                ids.extend(d["resource_id"] for d in docs)
                following = payload.get("nextCursorMark")
                if not docs or not following or following == cursor:
                    break
                cursor = following
            self._shared.resource_ids = frozenset(ids)
            self._shared.count = len(ids)
        return self._shared.resource_ids

    def get(self, resource_id: str) -> Optional[dict]:
        """:meth:`candidate_record` plus the local file; ``None`` if the
        core has no such document."""
        record = self.candidate_record(resource_id)
        if record is None:
            return None
        path = self.dataset_filepath(resource_id)
        return {
            **record,
            "csv_path": str(path) if path is not None else "",
            "csv_exists": path is not None,
            "source": self.source,
        }

    def dataset_filepath(self, resource_id: str) -> Optional[Path]:
        """The local table of a core document; ``None`` when there is none."""
        return self._local_files.get(resource_id)

    def __len__(self) -> int:
        """The core's document count (not the local files')."""
        return self._shared.count if self._shared.count is not None else self.ping()
