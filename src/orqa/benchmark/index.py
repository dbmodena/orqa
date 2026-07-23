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
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Relative weights of each metadata field when scoring a match.
# A keyword hitting the title matters more than one buried in
# the description.
FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
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
INDEX_FORMAT_VERSION = 1

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
        "tags": " ".join(record.get("tags") or []),
        "columns": " ".join(columns_parts),
        "publisher": record.get("publisher") or "",
        "description": record.get("description") or "",
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

    def to_dict(self) -> dict:
        return {
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


class DatasetIndex:
    """
    BM25 inverted index over the normalized metadata of a single source.

    Documents are metadata records keyed by resource_id; the on-disk
    dataset for a record is <datasets_path>/<resource_id>.<fmt>, the
    same convention used by the rest of the pipeline.
    """

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

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        """
        Rank datasets against a set of keywords.

        `keywords` can be a free-text string or a list of keywords;
        either way it is normalized with the same tokenizer used at
        indexing time. Set `only_available` to drop results whose CSV
        file is not present on disk.
        """
        if isinstance(keywords, str):
            terms = tokenize(keywords)
        else:
            terms = [t for kw in keywords for t in tokenize(kw)]

        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, set[str]] = defaultdict(set)

        for term in set(terms):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for resource_id, tf in self._postings[term].items():
                dl = self._doc_len[resource_id]
                norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avg_doc_len)
                scores[resource_id] += idf * (tf * (BM25_K1 + 1)) / (tf + norm)
                matched[resource_id].add(term)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

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
