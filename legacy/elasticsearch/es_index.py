"""
Elasticsearch-backed reverse index over the normalized datasets metadata.

Same role and interface as orqa.benchmark.index.DatasetIndex, but the
inverted index lives in an Elasticsearch index (one per city, named
"orqa-<provider>-<city>"). When the MCP server starts, the index is
created from <data_path>/metadata/normalized_metadata.json if it does not
exist, and recreated if the metadata file changed since it was built
(a fingerprint of the metadata file is stored in the index _meta).

Ranking is Elasticsearch's BM25 with per-field boosts mirroring the
built-in backend (title = resource name > tags > columns > publisher >
description) and
an accent-folding analyzer for the multilingual portals.

Matching is additive by default (``match="any"``: a record scores on every
term it contains, summed across its fields — a ``cross_fields`` query, not
the per-field maximum of ``best_fields``). ``match="all"`` requires EVERY
term to be present — in any of the searched fields — which is how the
portal's keyword search behaves (Solr edismax with ``q.op=AND``); ``fields``
limits which record fields may match. ``orqa.benchmark.index.portal_view``
binds those two to an index for the retrievability gate.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from orqa.benchmark.es_local import announce
from orqa.benchmark.index import (
    FIELD_WEIGHTS,
    SearchResult,
    _record_field_texts,
    tokenize,
)

# Bumped whenever the mapping or the indexing scheme changes, to force
# a rebuild of indexes created by older versions of this module.
ES_INDEX_FORMAT_VERSION = 4

# Query-time field boosts, mirroring index.FIELD_WEIGHTS
SEARCH_FIELDS = [
    "title^3",
    "resource_name^3",
    "tags^2.5",
    "columns_text^2",
    "publisher^1.5",
    "description",
]

# The normalized-record field names (index.FIELD_WEIGHTS) -> the boosted
# Elasticsearch field each one is searched as, so a caller restricts a search
# in the vocabulary the rest of the pipeline uses.
_ES_FIELD = {
    "title": "title^3",
    "resource_name": "resource_name^3",
    "tags": "tags^2.5",
    "columns": "columns_text^2",
    "publisher": "publisher^1.5",
    "description": "description",
}

# Just what a ranking needs from each hit (no highlights, no other record
# fields) — the retrievability gate runs hundreds of searches per table group.
_LEAN_SOURCE = [
    "resource_id",
    "record.dataset_id",
    "record.title",
    "record.publisher",
    "record.tags",
    "record.dataset_url",
]

# `msearch` batch size: bounds one request's body, not the number of trials.
_MSEARCH_CHUNK = 200

_HIGHLIGHT_RE = re.compile(r"<em>(.*?)</em>")

_INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    # HybridDatasetIndex fuses over the WHOLE lexical ranking (it asks for
    # `len(index)` results), and Elasticsearch refuses a window past this
    # setting's 10,000 default — the UK corpus is 27,390 records.
    "max_result_window": 100000,
    "analysis": {
        "analyzer": {
            # lowercase + accent folding, so "crédito" matches "credito"
            # across the English/French/Italian/Spanish/Catalan portals.
            # Tokens split on EVERY non-alphanumeric ("foo_bar" -> foo, bar;
            # "3.5" -> 3, 5), exactly as index.tokenize does — the standard
            # tokenizer keeps those whole, so a term the anchor search took from
            # a record's text would not match it here. Solr's text_general
            # splits them too (WordDelimiterGraphFilter).
            "folded": {
                "type": "custom",
                "tokenizer": "alnum",
                "filter": ["lowercase", "asciifolding"],
            }
        },
        "tokenizer": {
            "alnum": {"type": "pattern", "pattern": r"[^\p{L}\p{N}]+"},
        },
    },
}

_TEXT = {"type": "text", "analyzer": "folded"}

_MAPPINGS_PROPERTIES = {
    "resource_id": {"type": "keyword"},
    "dataset_id": {"type": "keyword"},
    "title": _TEXT,
    "resource_name": _TEXT,
    "tags": _TEXT,
    "columns_text": _TEXT,
    "publisher": _TEXT,
    "description": _TEXT,
    # The full normalized metadata record, stored as-is for retrieval
    # but not indexed.
    "record": {"type": "object", "enabled": False},
}


def connect(es_url: str) -> Elasticsearch:
    """
    Connect to Elasticsearch and fail fast with an actionable message
    when the cluster is unreachable.
    """
    es = Elasticsearch(es_url, request_timeout=30)
    if not es.ping():
        raise RuntimeError(
            f"Cannot reach Elasticsearch at {es_url}. Start it (e.g. "
            "docker run -d -p 9200:9200 -e discovery.type=single-node "
            "-e xpack.security.enabled=false "
            "docker.elastic.co/elasticsearch/elasticsearch:8.17.0) or fix "
            "tasks.mcp_search.elasticsearch_url in the workflow yaml, or "
            'switch tasks.mcp_search.backend to "builtin".'
        )
    return es


class ESDatasetIndex:
    """
    Elasticsearch counterpart of DatasetIndex; duck-type compatible with
    the tools in server.py (search / get / dataset_filepath / len).
    """

    # portal_view: this index can search with match="all" / a field subset,
    # and its search can skip the (slow) highlighting.
    supports_match = True
    supports_highlight = True

    def __init__(
        self,
        es: Elasticsearch,
        index_name: str,
        datasets_path: Path,
        datasets_format: str = "csv",
        source: Optional[str] = None,
    ):
        self.es = es
        self.index_name = index_name
        self.datasets_path = Path(datasets_path)
        self.datasets_format = datasets_format
        self.source = source
        # (resource_id, fields) -> distinct tokens of that record's text
        self._terms_cache: dict[tuple, Optional[frozenset[str]]] = {}

    # ------------------------------------------------------------------
    # index lifecycle

    @staticmethod
    def _metadata_fingerprint(normalized_metadata_filepath: Path) -> dict:
        stat = Path(normalized_metadata_filepath).stat()
        return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}

    @classmethod
    def build_or_load(
        cls,
        es: Elasticsearch,
        index_name: str,
        normalized_metadata_filepath: Path,
        datasets_path: Path,
        datasets_format: str = "csv",
        source: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> tuple["ESDatasetIndex", bool]:
        """
        Reuse the Elasticsearch index when it exists and is up to date
        with the normalized metadata file, otherwise (re)create it.

        Returns the index and whether it was rebuilt.
        """
        index = cls(es, index_name, datasets_path, datasets_format, source)

        if not force_rebuild and es.indices.exists(index=index_name):
            mapping = es.indices.get_mapping(index=index_name)
            meta = mapping[index_name]["mappings"].get("_meta", {})
            if meta.get("format_version") == ES_INDEX_FORMAT_VERSION and meta.get(
                "metadata_fingerprint"
            ) == cls._metadata_fingerprint(normalized_metadata_filepath):
                return index, False

        index._create(normalized_metadata_filepath)
        return index, True

    def _create(self, normalized_metadata_filepath: Path) -> None:
        import json

        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        self.es.indices.create(
            index=self.index_name,
            settings=_INDEX_SETTINGS,
            mappings={
                "_meta": {
                    "format_version": ES_INDEX_FORMAT_VERSION,
                    "metadata_fingerprint": self._metadata_fingerprint(
                        normalized_metadata_filepath
                    ),
                    "source": self.source,
                },
                "properties": _MAPPINGS_PROPERTIES,
            },
        )

        with open(normalized_metadata_filepath, "r") as file:
            records = json.load(file)

        announce(
            f"building index '{self.index_name}' from {Path(normalized_metadata_filepath).name} "
            f"({len(records)} records)…"
        )
        bulk(self.es, self._actions(records))
        self.es.indices.refresh(index=self.index_name)
        announce(f"index '{self.index_name}' built.")

    def _actions(self, records: Iterable[dict]) -> Iterable[dict]:
        for record in records:
            resource_id = record.get("resource_id") or record.get("dataset_id")
            if not resource_id:
                continue
            fields = _record_field_texts(record)
            yield {
                "_index": self.index_name,
                "_id": resource_id,
                "resource_id": resource_id,
                "dataset_id": record.get("dataset_id", resource_id),
                "title": fields["title"],
                "resource_name": fields["resource_name"],
                "tags": fields["tags"],
                "columns_text": fields["columns"],
                "publisher": fields["publisher"],
                "description": fields["description"],
                "record": record,
            }

    # ------------------------------------------------------------------
    # DatasetIndex-compatible interface

    def __len__(self) -> int:
        return int(self.es.count(index=self.index_name)["count"])

    def dataset_filepath(self, resource_id: str) -> Path:
        return self.datasets_path / f"{resource_id}.{self.datasets_format}"

    def get(self, resource_id: str) -> Optional[dict]:
        if not self.es.exists(index=self.index_name, id=resource_id):
            return None
        record = self.es.get(index=self.index_name, id=resource_id)["_source"][
            "record"
        ]
        filepath = self.dataset_filepath(resource_id)
        return {
            **record,
            "csv_path": str(filepath),
            "csv_exists": filepath.exists(),
            "source_key": self.source,
        }

    # ------------------------------------------------------------------
    # search

    @staticmethod
    def _keyword_text(keywords: str | Iterable[str]) -> str:
        return keywords if isinstance(keywords, str) else " ".join(keywords)

    @staticmethod
    def _check_match(match: str) -> None:
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")

    @staticmethod
    def _es_fields(fields: Optional[Iterable[str]]) -> list[str]:
        """The boosted Elasticsearch fields for a set of normalized-record
        field names (every field when ``None``)."""
        if fields is None:
            return list(SEARCH_FIELDS)
        wanted = set(fields)
        unknown = wanted - set(FIELD_WEIGHTS)
        if unknown:
            raise ValueError(
                f"unknown index field(s) {sorted(unknown)}; "
                f"valid fields: {sorted(FIELD_WEIGHTS)}"
            )
        if not wanted:
            raise ValueError("fields must name at least one index field")
        return [es for name, es in _ES_FIELD.items() if name in wanted]

    def _query(
        self, text: str, match: str, fields: Optional[Iterable[str]]
    ) -> dict:
        # `cross_fields` for BOTH operators, because the default `best_fields`
        # is wrong for each:
        #   and -> it would need every term in ONE field — "belfast lough"
        #     (title) with "2007" (description) would not match — which is not
        #     how the portal's keyword search behaves.
        #   or  -> a record's score is its single best FIELD, not a sum, so a
        #     term that only the tags hold ("ni") never adds to a title match
        #     and cannot tell apart records that differ only in their tags.
        #     `cross_fields` blends the fields into one, so every keyword a
        #     record contains adds to its score, as in the built-in index.
        return {
            "multi_match": {
                "query": text,
                "fields": self._es_fields(fields),
                "operator": "or" if match == "any" else "and",
                "type": "cross_fields",
            }
        }

    def _hits_to_results(
        self, hits: list[dict], top_k: int, only_available: bool
    ) -> list[SearchResult]:
        results = []
        for hit in hits:
            record = hit["_source"]["record"]
            resource_id = hit["_source"]["resource_id"]
            filepath = self.dataset_filepath(resource_id)
            exists = filepath.exists()
            if only_available and not exists:
                continue

            matched = {
                term.lower()
                for fragments in hit.get("highlight", {}).values()
                for fragment in fragments
                for term in _HIGHLIGHT_RE.findall(fragment)
            }

            results.append(
                SearchResult(
                    resource_id=resource_id,
                    dataset_id=record.get("dataset_id", resource_id),
                    title=record.get("title", ""),
                    publisher=record.get("publisher"),
                    tags=record.get("tags") or [],
                    score=float(hit["_score"]),
                    matched_terms=sorted(matched),
                    csv_path=str(filepath),
                    csv_exists=exists,
                    dataset_url=record.get("dataset_url"),
                    source=self.source,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
        match: str = "any",
        fields: Optional[Iterable[str]] = None,
        highlight: bool = True,
    ) -> list[SearchResult]:
        """
        Rank datasets against a set of keywords with BM25 and return the
        top_k matches.

        ``match="all"`` requires every keyword to be present in some searched
        field (the portal's AND); the default ``"any"`` is additive across
        fields (a keyword only the tags hold still adds to a record's score).
        ``fields`` limits matching to the named record fields
        (``index.FIELD_WEIGHTS``).
        ``highlight=False`` skips highlighting — and with it ``matched_terms``
        — and fetches only what a ranking needs, for callers running many
        searches.
        """
        self._check_match(match)
        text = self._keyword_text(keywords)
        if match == "all" and not tokenize(text):
            return []

        # CSV availability is filesystem knowledge Elasticsearch does not
        # have, so over-fetch and post-filter when only_available is set.
        size = top_k * 5 if only_available else top_k

        request: dict = {
            "index": self.index_name,
            "query": self._query(text, match, fields),
            "size": size,
        }
        if highlight:
            request["highlight"] = {
                "fields": {
                    field.split("^")[0]: {"number_of_fragments": 3}
                    for field in SEARCH_FIELDS
                }
            }
        else:
            request["source_includes"] = _LEAN_SOURCE

        response = self.es.search(**request)
        return self._hits_to_results(response["hits"]["hits"], top_k, only_available)

    def search_many(
        self,
        queries: list[str | Iterable[str]],
        top_k: int = 10,
        only_available: bool = False,
        match: str = "any",
        fields: Optional[Iterable[str]] = None,
    ) -> list[list[SearchResult]]:
        """``search`` for each query, batched through ``_msearch`` — one round
        trip per ``_MSEARCH_CHUNK`` queries instead of one per query. That is
        what ``keyword_suggestion`` uses to score a round of trials. Same
        semantics as ``search(..., highlight=False)``, in the same order."""
        self._check_match(match)
        size = top_k * 5 if only_available else top_k
        results: list[list[SearchResult]] = [[] for _ in queries]
        positions: list[int] = []
        bodies: list[dict] = []
        for position, keywords in enumerate(queries):
            text = self._keyword_text(keywords)
            if match == "all" and not tokenize(text):
                continue  # no terms, no query: nothing can match
            positions.append(position)
            bodies.append(
                {
                    "query": self._query(text, match, fields),
                    "size": size,
                    "_source": _LEAN_SOURCE,
                }
            )

        for start in range(0, len(bodies), _MSEARCH_CHUNK):
            searches: list[dict] = []
            for body in bodies[start:start + _MSEARCH_CHUNK]:
                searches.extend([{"index": self.index_name}, body])
            responses = self.es.msearch(searches=searches)["responses"]
            for offset, response in enumerate(responses):
                if "error" in response:
                    raise RuntimeError(f"Elasticsearch _msearch failed: {response['error']}")
                results[positions[start + offset]] = self._hits_to_results(
                    response["hits"]["hits"], top_k, only_available
                )
        return results

    def searchable_terms(
        self, resource_id: str, fields: Optional[Iterable[str]] = None
    ) -> Optional[frozenset[str]]:
        """The distinct tokens a search restricted to ``fields`` can match on
        this record (every field when ``None``); ``None`` for an unknown
        record. Same meaning as ``DatasetIndex.searchable_terms`` — computed
        from the stored record, so it costs one ``get`` per record."""
        wanted = None if fields is None else frozenset(fields)
        key = (resource_id, wanted)
        if key not in self._terms_cache:
            record = self.get(resource_id)
            if record is None:
                self._terms_cache[key] = None
            else:
                tokens: set[str] = set()
                for name, text in _record_field_texts(record).items():
                    if wanted is None or name in wanted:
                        tokens.update(tokenize(text))
                self._terms_cache[key] = frozenset(tokens)
        return self._terms_cache[key]
