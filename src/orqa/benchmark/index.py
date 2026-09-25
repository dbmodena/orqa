"""
Keyword-based reverse index over the normalized datasets metadata.

Each open data source (e.g. socrata/nyc, ckan/valencia, ods/paris) stores
its metadata in <data_path>/metadata/normalized_metadata.json, a list of
records with a common schema (title, description, publisher, tags, columns,
download info). This module builds an in-memory inverted index over those
textual fields and ranks datasets with BM25, so that given a bunch of
keywords extrapolated from a question we can find the CSV files needed
to answer it.

The index is materialized under <data_path>/index/ so that the MCP
server starts from a ready-to-use artifact; it is transparently rebuilt
whenever the normalized metadata file changes. No external search
infrastructure is required.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Relative weights of each metadata field when scoring a match.
# A keyword hitting the title matters more than one buried in
# the description.
FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    # Names one file of a dataset ("2021-12-31 Organogram (Junior)"): as
    # identifying as the title it refines.
    "resource_name": 3.0,
    "tags": 2.5,
    "columns": 2.0,
    "publisher": 1.5,
    "description": 1.0,
}

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Bumped whenever the on-disk index layout or the tokenization/weighting
# scheme changes, to force a rebuild of stale index files.
INDEX_FORMAT_VERSION = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase, strip accents/diacritics and split on any
    non-alphanumeric character.

    Accent folding keeps the index usable across the languages of the
    crawled portals (English, French, Italian, Spanish, Catalan), e.g.
    both "credito" and "crédito" map to the same token.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _TOKEN_RE.findall(text)


def _record_field_texts(record: dict) -> dict[str, str]:
    """Extract the indexable text of each weighted field from a record."""
    columns_parts = []
    for col in record.get("columns") or []:
        for key in ("name", "label", "description"):
            value = col.get(key)
            if value:
                columns_parts.append(str(value))

    return {
        "title": record.get("title") or "",
        "resource_name": record.get("resource_name") or "",
        "tags": " ".join(record.get("tags") or []),
        "columns": " ".join(columns_parts),
        "publisher": record.get("publisher") or "",
        "description": " ".join(
            part
            for part in (record.get("resource_description"), record.get("description"))
            if part
        ),
    }


@dataclass
class SearchResult:
    resource_id: str
    dataset_id: str
    title: str
    publisher: Optional[str]
    tags: list[str]
    score: float
    matched_terms: list[str]
    csv_path: str
    csv_exists: bool
    dataset_url: Optional[str]
    source: Optional[str] = None
    # Populated by HybridDatasetIndex. Kept optional so lexical-only indexes
    # retain the same result contract and callers can inspect score fusion
    # without needing a second diagnostics API.
    lexical_score: Optional[float] = None
    semantic_score: Optional[float] = None

    def to_dict(self) -> dict:
        result = {
            "resource_id": self.resource_id,
            "dataset_id": self.dataset_id,
            "title": self.title,
            "publisher": self.publisher,
            "tags": self.tags,
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
            "csv_path": self.csv_path,
            "csv_exists": self.csv_exists,
            "dataset_url": self.dataset_url,
            "source": self.source,
        }
        if self.lexical_score is not None:
            result["lexical_score"] = round(self.lexical_score, 4)
        if self.semantic_score is not None:
            result["semantic_score"] = round(self.semantic_score, 4)
        return result


class DatasetIndex:
    """
    BM25 inverted index over the normalized metadata of a single source.

    Documents are metadata records keyed by resource_id; the on-disk
    dataset for a record is <datasets_path>/<resource_id>.<fmt>, the
    same convention used by the rest of the pipeline.
    """

    # portal_view: search() accepts match="all" and a field subset.
    supports_match = True

    def __init__(
        self,
        records: Iterable[dict],
        datasets_path: Path,
        datasets_format: str = "csv",
        source: Optional[str] = None,
    ):
        self.datasets_path = Path(datasets_path)
        self.datasets_format = datasets_format
        self.source = source

        self._records: dict[str, dict] = {}
        # term -> {resource_id -> weighted term frequency}
        self._postings: dict[str, dict[str, float]] = defaultdict(dict)
        # resource_id -> weighted document length
        self._doc_len: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        # (resource_id, fields) -> distinct tokens of that record's text in
        # those fields; filled lazily by ``searchable_terms``.
        self._terms_cache: dict[tuple, frozenset[str]] = {}

        self._build(records)

    @classmethod
    def from_metadata_file(
        cls,
        normalized_metadata_filepath: Path,
        datasets_path: Path,
        datasets_format: str = "csv",
        source: Optional[str] = None,
    ) -> "DatasetIndex":
        with open(normalized_metadata_filepath, "r") as file:
            records = json.load(file)
        return cls(records, datasets_path, datasets_format, source)

    @staticmethod
    def _metadata_fingerprint(normalized_metadata_filepath: Path) -> dict:
        stat = Path(normalized_metadata_filepath).stat()
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}

    def save(self, index_filepath: Path, normalized_metadata_filepath: Path) -> None:
        """
        Materialize the index (postings + records) as a single
        self-contained JSON file, remembering a fingerprint of the
        metadata file it was built from for staleness detection.
        """
        index_filepath = Path(index_filepath)
        index_filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_FORMAT_VERSION,
            "source": self.source,
            "datasets_format": self.datasets_format,
            "metadata_fingerprint": self._metadata_fingerprint(
                normalized_metadata_filepath
            ),
            "avg_doc_len": self._avg_doc_len,
            "doc_len": self._doc_len,
            "postings": dict(self._postings),
            "records": self._records,
        }
        with open(index_filepath, "w") as file:
            json.dump(payload, file, ensure_ascii=False)

    @classmethod
    def load(cls, index_filepath: Path, datasets_path: Path) -> "DatasetIndex":
        with open(index_filepath, "r") as file:
            payload = json.load(file)
        if payload.get("version") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"Index file {index_filepath} has version "
                f"{payload.get('version')}, expected {INDEX_FORMAT_VERSION}"
            )
        return cls._from_payload(payload, datasets_path)

    @classmethod
    def _from_payload(cls, payload: dict, datasets_path: Path) -> "DatasetIndex":
        index = cls.__new__(cls)
        index.datasets_path = Path(datasets_path)
        index.datasets_format = payload["datasets_format"]
        index.source = payload.get("source")
        index._records = payload["records"]
        index._postings = defaultdict(dict, payload["postings"])
        index._doc_len = payload["doc_len"]
        index._avg_doc_len = payload["avg_doc_len"]
        index._terms_cache = {}
        return index

    @classmethod
    def build_or_load(
        cls,
        normalized_metadata_filepath: Path,
        index_filepath: Path,
        datasets_path: Path,
        datasets_format: str = "csv",
        source: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> tuple["DatasetIndex", bool]:
        """
        Load the materialized index when it is up to date with the
        normalized metadata file, otherwise (re)build and save it.

        Returns the index and whether it was rebuilt.
        """
        index_filepath = Path(index_filepath)
        if not force_rebuild and index_filepath.exists():
            try:
                with open(index_filepath, "r") as file:
                    payload = json.load(file)
                fresh = payload.get(
                    "version"
                ) == INDEX_FORMAT_VERSION and payload.get(
                    "metadata_fingerprint"
                ) == cls._metadata_fingerprint(normalized_metadata_filepath)
            except (json.JSONDecodeError, OSError):
                fresh = False
            if fresh:
                return cls._from_payload(payload, datasets_path), False

        index = cls.from_metadata_file(
            normalized_metadata_filepath, datasets_path, datasets_format, source
        )
        index.save(index_filepath, normalized_metadata_filepath)
        return index, True

    def _build(self, records: Iterable[dict]) -> None:
        for record in records:
            resource_id = record.get("resource_id") or record.get("dataset_id")
            if not resource_id:
                continue
            self._records[resource_id] = record

            weighted_tf: Counter[str] = Counter()
            doc_len = 0.0
            for field_name, text in _record_field_texts(record).items():
                weight = FIELD_WEIGHTS[field_name]
                tokens = tokenize(text)
                doc_len += weight * len(tokens)
                for token in tokens:
                    weighted_tf[token] += weight

            self._doc_len[resource_id] = doc_len
            for token, tf in weighted_tf.items():
                self._postings[token][resource_id] = tf

        if self._doc_len:
            self._avg_doc_len = sum(self._doc_len.values()) / len(self._doc_len)

    def __len__(self) -> int:
        return len(self._records)

    def _idf(self, term: str) -> float:
        n = len(self._records)
        df = len(self._postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def dataset_filepath(self, resource_id: str) -> Path:
        return self.datasets_path / f"{resource_id}.{self.datasets_format}"

    def get(self, resource_id: str) -> Optional[dict]:
        record = self._records.get(resource_id)
        if record is None:
            return None
        filepath = self.dataset_filepath(resource_id)
        return {
            **record,
            "csv_path": str(filepath),
            "csv_exists": filepath.exists(),
            "source_key": self.source,
        }

    def searchable_terms(
        self, resource_id: str, fields: Optional[Iterable[str]] = None
    ) -> Optional[frozenset[str]]:
        """The distinct index tokens of a record's text in ``fields`` (every
        indexed field when ``None``); ``None`` for an unknown record.

        This is what a search restricted to ``fields`` can match on that
        record — e.g. a portal whose Solr ``qf`` has no column fields cannot
        find a table by one of its column names, however well that name
        scores in the merged posting lists.
        """
        record = self._records.get(resource_id)
        if record is None:
            return None
        wanted = None if fields is None else frozenset(fields)
        key = (resource_id, wanted)
        cached = self._terms_cache.get(key)
        if cached is not None:
            return cached
        tokens: set[str] = set()
        for field_name, text in _record_field_texts(record).items():
            if wanted is None or field_name in wanted:
                tokens.update(tokenize(text))
        result = frozenset(tokens)
        self._terms_cache[key] = result
        return result

    def _score(
        self,
        terms: list[str],
        match: str,
        fields: Optional[frozenset[str]],
    ) -> tuple[dict[str, float], dict[str, set[str]]]:
        """BM25 scores (and matched terms) per record.

        ``match="any"`` (the default everywhere): a record scores on every
        term it contains, so more matching terms only add score — the
        additive behavior this index has always had.

        ``match="all"``: a record qualifies only if it contains EVERY term
        (an unknown term therefore matches nothing), then ranks by the same
        BM25 among those — the semantics of a Solr edismax query with
        ``q.op=AND``.

        ``fields`` restricts what counts as a match to those record fields.
        Scores still use the merged, weighted term frequencies (a close
        approximation of a per-field score, exact for membership).
        """
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, set[str]] = defaultdict(set)
        distinct = sorted(set(terms))
        known = [t for t in distinct if self._idf(t) != 0.0]

        candidates: Optional[set[str]] = None
        if match == "all":
            if not distinct or len(known) != len(distinct):
                return {}, {}
            by_size = sorted(known, key=lambda t: len(self._postings[t]))
            candidates = set(self._postings[by_size[0]])
            for term in by_size[1:]:
                candidates.intersection_update(self._postings[term])
                if not candidates:
                    return {}, {}

        for term in known:
            idf = self._idf(term)
            for resource_id, tf in self._postings[term].items():
                if candidates is not None and resource_id not in candidates:
                    continue
                if fields is not None and term not in self.searchable_terms(resource_id, fields):
                    continue
                dl = self._doc_len[resource_id]
                norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avg_doc_len)
                scores[resource_id] += idf * (tf * (BM25_K1 + 1)) / (tf + norm)
                matched[resource_id].add(term)

        if match == "all":
            complete = {rid for rid, hit in matched.items() if len(hit) == len(known)}
            scores = {rid: sc for rid, sc in scores.items() if rid in complete}
        return scores, matched

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
        match: str = "any",
        fields: Optional[Iterable[str]] = None,
    ) -> list[SearchResult]:
        """
        Rank datasets against a set of keywords.

        `keywords` can be a free-text string or a list of keywords;
        either way it is normalized with the same tokenizer used at
        indexing time. Set `only_available` to drop results whose CSV
        file is not present on disk.

        `match="all"` requires every keyword to be present (Solr's
        ``q.op=AND``); the default ``"any"`` is additive. `fields` limits
        matching to the named record fields (see ``FIELD_WEIGHTS``).
        """
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        allowed: Optional[frozenset[str]] = None
        if fields is not None:
            allowed = frozenset(fields)
            unknown = allowed - set(FIELD_WEIGHTS)
            if unknown:
                raise ValueError(
                    f"unknown index field(s) {sorted(unknown)}; "
                    f"valid fields: {sorted(FIELD_WEIGHTS)}"
                )
        if isinstance(keywords, str):
            terms = tokenize(keywords)
        else:
            terms = [t for kw in keywords for t in tokenize(kw)]

        scores, matched = self._score(terms, match, allowed)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

        results = []
        for resource_id, score in ranked:
            record = self._records[resource_id]
            filepath = self.dataset_filepath(resource_id)
            exists = filepath.exists()
            if only_available and not exists:
                continue
            results.append(
                SearchResult(
                    resource_id=resource_id,
                    dataset_id=record.get("dataset_id", resource_id),
                    title=record.get("title", ""),
                    publisher=record.get("publisher"),
                    tags=record.get("tags") or [],
                    score=score,
                    matched_terms=sorted(matched[resource_id]),
                    csv_path=str(filepath),
                    csv_exists=exists,
                    dataset_url=record.get("dataset_url"),
                    source=self.source,
                )
            )
            if len(results) >= top_k:
                break
        return results


class PortalIndexView:
    """A lexical view of a reverse index — the built-in :class:`DatasetIndex`
    or ``orqa.benchmark.es_index.ESDatasetIndex``, anything that sets
    ``supports_match`` — that answers the way the portal's own search does:
    every keyword must match (Solr ``q.op=AND``), and only in the record
    fields that portal actually searches.

    The retrievability gate and the keyword-anchor search used to run on the
    additive ``match="any"`` index, which is MORE lenient than the portal — a
    keyword the gold table lacks merely adds nothing there, but under AND it
    removes the table. Judging retrievability through this view makes "found"
    mean what it means to the portal. The underlying index is not modified,
    so consumers that want additive matching (the benchmark solver) are
    unaffected.

    This is an IMITATION, for when no Solr is reachable: it does not know the
    core's document universe, its field names or its ``qf``, only what the
    caller tells it in ``fields``. Prefer ``orqa.benchmark.solr_index.
    SolrDatasetIndex``, which asks the real core.
    """

    # keyword_suggestion: candidate terms must be ones this view can match.
    restricts_candidates = True

    def __init__(
        self,
        index: DatasetIndex,
        match: str = "all",
        fields: Optional[Iterable[str]] = None,
    ):
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        self._index = index
        self.match = match
        self.searchable_fields: Optional[frozenset[str]] = (
            frozenset(fields) if fields else None
        )
        unknown = (self.searchable_fields or frozenset()) - set(FIELD_WEIGHTS)
        if unknown:
            raise ValueError(
                f"unknown index field(s) {sorted(unknown)}; "
                f"valid fields: {sorted(FIELD_WEIGHTS)}"
            )

    def _search_options(self) -> dict:
        options: dict = {"match": self.match, "fields": self.searchable_fields}
        if getattr(self._index, "supports_highlight", False):
            # The gate runs hundreds of searches per table group and never
            # reads matched_terms; highlighting is the slow part of an
            # Elasticsearch search.
            options["highlight"] = False
        return options

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        return self._index.search(
            keywords, top_k=top_k, only_available=only_available, **self._search_options()
        )

    def search_many(
        self,
        queries: list[str | Iterable[str]],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[list[SearchResult]]:
        """``search`` for each query, in order — batched by the underlying
        index when it can (Elasticsearch ``_msearch``), else one by one."""
        batched = getattr(self._index, "search_many", None)
        if callable(batched):
            options = {"match": self.match, "fields": self.searchable_fields}
            return batched(queries, top_k=top_k, only_available=only_available, **options)
        return [self.search(q, top_k, only_available) for q in queries]

    def searchable_terms(self, resource_id: str) -> Optional[frozenset[str]]:
        """What this view can match on ``resource_id`` (see
        ``DatasetIndex.searchable_terms``)."""
        return self._index.searchable_terms(resource_id, self.searchable_fields)

    def get(self, resource_id: str) -> Optional[dict]:
        return self._index.get(resource_id)

    def dataset_filepath(self, resource_id: str) -> Path:
        return self._index.dataset_filepath(resource_id)

    def __len__(self) -> int:
        return len(self._index)

    def __getattr__(self, name: str) -> Any:
        # datasets_path / datasets_format / source: plain pass-through.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._index, name)


def portal_view(
    index: Any, match: str = "any", fields: Iterable[str] = ()
) -> Any:
    """``index`` as the portal searches it — or ``index`` itself, unchanged.

    Returns a :class:`PortalIndexView` over the lexical index (the
    lexical half of a hybrid index) when ``match`` is ``"all"`` or ``fields``
    restricts the searched fields. Works over any index that ``supports_match``
    — the built-in one and the Elasticsearch one. Returns ``index`` untouched
    when the defaults are requested, and — with a warning — when the index
    cannot match that way, so a run degrades to the additive behavior it had
    before rather than failing.
    """
    fields = tuple(fields or ())
    if match == "any" and not fields:
        return index
    lexical = getattr(index, "lexical_index", index)
    if not getattr(lexical, "supports_match", False):
        logger.warning(
            "Portal-faithful matching (match=%r, fields=%s) needs an index that "
            "supports it (builtin or elasticsearch); %s does not — falling back "
            "to additive matching.",
            match, list(fields), type(index).__name__,
        )
        return index
    return PortalIndexView(lexical, match=match, fields=fields)


def load_index(cfg) -> Optional[Any]:
    """Build/load the reverse index for the backend ``cfg.mcp_search.backend``
    selects, creating it from the normalized metadata when missing or stale.

    Shared by the MCP retrieval-benchmark server and any other caller (e.g.
    the plan judge's keyword-searchability check) that needs the SAME index
    a portal's ``mcp_search`` config points at. Returns ``None`` instead of
    raising when the index can't be built (metadata not yet produced,
    Elasticsearch unreachable, ...) — callers should treat that as "no
    index available" rather than a fatal error.
    """
    try:
        if cfg.mcp_search.backend == "elasticsearch":
            from orqa.benchmark import es_index

            es_url = (
                os.environ.get("ELASTICSEARCH_URL", "").strip()
                or cfg.mcp_search.elasticsearch_url
            )
            # tasks.mcp_search.elasticsearch_managed: start (installing first,
            # once) a LOCAL Elasticsearch for this run instead of expecting one.
            managed = getattr(cfg.mcp_search, "elasticsearch_managed", None)
            if managed is not None and managed.enabled:
                from orqa.benchmark import es_local

                es_local.ensure_running(cfg, es_url)
            es = es_index.connect(es_url)
            index, rebuilt = es_index.ESDatasetIndex.build_or_load(
                es,
                cfg.mcp_search.es_index_name,
                cfg.normalized_metadata_filepath,
                cfg.datasets_path,
                cfg.datasets_format,
                source=cfg.source,
            )
            location = f"Elasticsearch index {cfg.mcp_search.es_index_name!r} at {es_url}"
        else:
            index, rebuilt = DatasetIndex.build_or_load(
                cfg.normalized_metadata_filepath,
                cfg.mcp_search.index_filepath,
                cfg.datasets_path,
                cfg.datasets_format,
                source=cfg.source,
            )
            location = str(cfg.mcp_search.index_filepath)
    except Exception:
        logger.warning("Could not build/load the reverse index for %r.", cfg.source, exc_info=True)
        return None

    action = "Created" if rebuilt else "Reusing"
    logger.info("%s %s (%d datasets)", action, location, len(index))
    # main.py leaves logging unconfigured, so the line above is never shown; this
    # one says the index is up and which one it is.
    print(f"[index] {action} {location} ({len(index)} datasets) — index ready.", flush=True)

    if cfg.mcp_search.hybrid_search_enabled:
        try:
            from orqa.agent.llm_client.EmbeddingClient import EmbeddingClient
            from orqa.benchmark.hybrid_index import HybridDatasetIndex

            embedding_client = EmbeddingClient(
                cfg.llm_config_path / "litellm.yaml",
                batch_size=cfg.candidates_discovery.embedding_batch_size,
            )
            # The search-metadata cache (orqa.embedding_discovery.pipeline.
            # embed_search_metadata) covers EVERY normalized record, not just
            # the ones discovery indexes — prefer it, falling back to
            # discovery's own cache for a portal that hasn't run the
            # embed-search-metadata step yet.
            search_cache_path = getattr(
                cfg.candidates_discovery, "search_embeddings_path", None
            )
            cache_path = (
                search_cache_path
                if search_cache_path is not None and Path(search_cache_path).exists()
                else cfg.candidates_discovery.embeddings_cache_path
            )
            index = HybridDatasetIndex.from_cache(
                index,
                cache_path,
                embedding_client,
                lexical_weight=cfg.mcp_search.lexical_weight,
                semantic_weight=cfg.mcp_search.semantic_weight,
                query_input_type=cfg.mcp_search.query_embedding_input_type,
                query_cache_size=cfg.mcp_search.query_embedding_cache_size,
                fusion_method=cfg.mcp_search.fusion_method,
                rrf_k=cfg.mcp_search.rrf_k,
            )
            coverage = len(index._embedding_ids) / len(index) if len(index) else 0.0
            logger.info(
                "Hybrid retrieval enabled (fusion=%s, lexical=%.3f, semantic=%.3f%s) "
                "using %s — vector coverage %d/%d (%.1f%%).",
                index.fusion_method,
                index.lexical_weight,
                index.semantic_weight,
                f", rrf_k={index.rrf_k}" if index.fusion_method == "rrf" else "",
                cache_path,
                len(index._embedding_ids),
                len(index),
                100.0 * coverage,
            )
            if coverage < 1.0:
                logger.warning(
                    "Metadata vector coverage is below 100%% (%d/%d) — records "
                    "without a vector score 0 on the dense/semantic half of "
                    "every hybrid search. Run the embed-search-metadata step "
                    "to cover every record.",
                    len(index._embedding_ids),
                    len(index),
                )
        except Exception:
            # Keep the reverse index usable when semantic artifacts or the
            # provider are unavailable. Every consumer still receives this
            # same lexical fallback from the shared factory.
            logger.warning(
                "Could not enable hybrid retrieval for %r; using lexical search.",
                cfg.source,
                exc_info=True,
            )
    return index


@dataclass
class Catalog:
    """
    Discovers and lazily indexes every open data source available under
    the OrQA data directory.

    The expected layout is the one produced by the crawling pipeline:
    <data_dir>/<group>/<provider>/<city>/metadata/normalized_metadata.json
    with datasets in <data_dir>/<group>/<provider>/<city>/datasets/<fmt>/.
    Sources are addressed by "<provider>/<city>" (e.g. "socrata/nyc").
    """

    data_dir: Path
    datasets_format: str = "csv"
    _paths: dict[str, Path] = field(init=False, default_factory=dict)
    _indexes: dict[str, DatasetIndex] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Resolve so that the csv_path values returned to MCP clients
        # remain valid regardless of their working directory.
        self.data_dir = Path(self.data_dir).resolve()
        pattern = "*/*/*/metadata/normalized_metadata.json"
        for metadata_file in sorted(self.data_dir.glob(pattern)):
            city_path = metadata_file.parent.parent
            source_key = f"{city_path.parent.name}/{city_path.name}"
            self._paths[source_key] = city_path

    @property
    def sources(self) -> list[str]:
        return list(self._paths)

    def index(self, source: str) -> DatasetIndex:
        if source not in self._paths:
            available = ", ".join(self.sources) or "none"
            raise KeyError(f"Unknown source {source!r}. Available: {available}")
        if source not in self._indexes:
            city_path = self._paths[source]
            self._indexes[source] = DatasetIndex.from_metadata_file(
                city_path / "metadata" / "normalized_metadata.json",
                city_path / "datasets" / self.datasets_format,
                self.datasets_format,
                source=source,
            )
        return self._indexes[source]

    def search(
        self,
        keywords: str | Iterable[str],
        source: Optional[str] = None,
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        """
        Search one source, or every discovered source when `source` is None.

        Note: BM25 scores from different corpora are not strictly
        comparable, so cross-source rankings are indicative.
        """
        sources = [source] if source else self.sources
        results: list[SearchResult] = []
        for src in sources:
            results.extend(self.index(src).search(keywords, top_k, only_available))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, resource_id: str, source: Optional[str] = None) -> Optional[dict]:
        sources = [source] if source else self.sources
        for src in sources:
            record = self.index(src).get(resource_id)
            if record is not None:
                return record
        return None
