"""
Elasticsearch-backed reverse index over the normalized datasets metadata.

Same role and interface as orqa.benchmark.index.DatasetIndex, but the
inverted index lives in an Elasticsearch index (one per city, named
"orqa-<provider>-<city>"). When the MCP server starts, the index is
created from <data_path>/metadata/normalized_metadata.json if it does not
exist, and recreated if the metadata file changed since it was built
(a fingerprint of the metadata file is stored in the index _meta).

Ranking is Elasticsearch's BM25 with per-field boosts mirroring the
built-in backend (title > tags > columns > publisher > description) and
an accent-folding analyzer for the multilingual portals.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from orqa.benchmark.index import SearchResult, _record_field_texts

# Bumped whenever the mapping or the indexing scheme changes, to force
# a rebuild of indexes created by older versions of this module.
ES_INDEX_FORMAT_VERSION = 1

# Query-time field boosts, mirroring index.FIELD_WEIGHTS
SEARCH_FIELDS = [
    "title^3",
    "tags^2.5",
    "columns_text^2",
    "publisher^1.5",
    "description",
]

_HIGHLIGHT_RE = re.compile(r"<em>(.*?)</em>")

_INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
        "analyzer": {
            # lowercase + accent folding, so "crédito" matches "credito"
            # across the English/French/Italian/Spanish/Catalan portals
            "folded": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding"],
            }
        }
    },
}

_TEXT = {"type": "text", "analyzer": "folded"}

_MAPPINGS_PROPERTIES = {
    "resource_id": {"type": "keyword"},
    "dataset_id": {"type": "keyword"},
    "title": _TEXT,
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

        bulk(self.es, self._actions(records))
        self.es.indices.refresh(index=self.index_name)

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

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        """
        Rank datasets against a set of keywords with BM25 and return the
        top_k matches.
        """
        if not isinstance(keywords, str):
            keywords = " ".join(keywords)

        # CSV availability is filesystem knowledge Elasticsearch does not
        # have, so over-fetch and post-filter when only_available is set.
        size = top_k * 5 if only_available else top_k

        response = self.es.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": keywords,
                    "fields": SEARCH_FIELDS,
                    "operator": "or",
                }
            },
            highlight={
                "fields": {
                    field.split("^")[0]: {"number_of_fragments": 3}
                    for field in SEARCH_FIELDS
                }
            },
            size=size,
        )

        results = []
        for hit in response["hits"]["hits"]:
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
