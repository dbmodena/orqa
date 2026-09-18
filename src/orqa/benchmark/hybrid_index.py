"""Hybrid lexical/semantic ranking over a reverse-index backend.

The decorator keeps one retrieval contract for statement-generation keyword
suggestion, the plan gate, benchmark solving, and web replay: every caller sees
the same fusion of the configured lexical backend's BM25 score and cosine
similarity against the metadata vectors produced by semantic candidate
discovery.

Two fusion methods, selected by ``fusion_method`` (default unchanged):

- ``"weighted_score"`` (the original, still the default): each side's score
  is normalized to [0, 1] and combined as ``lexical_weight * lexical +
  semantic_weight * semantic`` — an average, so a candidate weak on ONE side
  still needs to be strong on the other to rank well overall.
- ``"rrf"`` (Reciprocal Rank Fusion, matching LakeGen's own Solr setup —
  see ``orqa.benchmark.retrieval_panel`` module docstring for the
  comparison): each side contributes ``1 / (rrf_k + rank)`` from its OWN
  ranked list, summed. Rank-based rather than score-based, which sidesteps
  BM25 and cosine similarity being on incomparable scales to begin with —
  and, unlike the weighted average, a candidate that ranks #1 on EITHER
  side alone already scores well, since "the mechanism underneath is the
  same" (both sides are ultimately just producing a relevance ranking; a
  document doesn't need to double-prove itself on both to be genuinely
  relevant).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from orqa.agent.llm_client.EmbeddingClient import EmbeddingClient
from orqa.utils import dataset_id_to_resource_id

from .index import SearchResult

logger = logging.getLogger(__name__)


class HybridDatasetIndex:
    """Backend-neutral fusion around ``DatasetIndex``-compatible APIs."""

    def __init__(
        self,
        lexical_index: Any,
        embedding_ids: list[str],
        embedding_vectors: np.ndarray,
        embedding_client: EmbeddingClient,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
        query_input_type: Optional[str] = "search_query",
        query_cache_size: int = 2048,
        fusion_method: Literal["weighted_score", "rrf"] = "weighted_score",
        rrf_k: int = 60,
    ):
        if lexical_weight < 0 or semantic_weight < 0:
            raise ValueError("Hybrid retrieval weights must be non-negative")
        weight_sum = lexical_weight + semantic_weight
        if weight_sum <= 0:
            raise ValueError("At least one hybrid retrieval weight must be positive")
        if len(embedding_ids) != len(embedding_vectors) or not embedding_ids:
            raise ValueError(
                "Hybrid retrieval requires at least one aligned metadata vector"
            )
        if fusion_method not in ("weighted_score", "rrf"):
            raise ValueError(
                f"fusion_method must be 'weighted_score' or 'rrf', got {fusion_method!r}"
            )
        if rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")

        self.lexical_index = lexical_index
        self.datasets_path = lexical_index.datasets_path
        self.datasets_format = lexical_index.datasets_format
        self.source = lexical_index.source
        self.embedding_client = embedding_client
        self.lexical_weight = lexical_weight / weight_sum
        self.semantic_weight = semantic_weight / weight_sum
        self.query_input_type = query_input_type
        self.query_cache_size = max(1, int(query_cache_size))
        self.fusion_method = fusion_method
        self.rrf_k = int(rrf_k)
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        vectors = np.asarray(embedding_vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"Expected a 2-D metadata embedding matrix, got {vectors.shape}"
            )
        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 0)
        if not np.any(valid):
            raise ValueError(
                "Metadata embedding cache contains no finite, non-zero vectors"
            )
        self._embedding_ids = [
            rid for rid, keep in zip(embedding_ids, valid) if keep
        ]
        self._embedding_vectors = vectors[valid] / norms[valid, None]

    @classmethod
    def from_cache(
        cls,
        lexical_index: Any,
        cache_path: Path,
        embedding_client: EmbeddingClient,
        **kwargs: Any,
    ) -> "HybridDatasetIndex":
        """Load and canonicalize the semantic-discovery metadata vector cache.

        Cache IDs are dataset filename stems. CKAN stems often include a
        human-readable prefix, so they are converted to the resource IDs used
        by both lexical backends. Duplicate canonical IDs are combined by a
        deterministic mean before normalization.
        """
        cache_path = Path(cache_path)
        manifest_path = cache_path.with_name("embeddings_manifest.json")
        if not cache_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Hybrid metadata embeddings require {cache_path} and {manifest_path}"
            )

        with open(manifest_path, encoding="utf-8") as file:
            manifest = json.load(file)
        with np.load(cache_path, allow_pickle=False) as data:
            raw_ids = [str(value) for value in data["ids"]]
            raw_vectors = np.asarray(data["vectors"], dtype=np.float32)

        if raw_vectors.ndim != 2 or len(raw_ids) != len(raw_vectors):
            raise ValueError(
                "Embedding cache IDs and vectors are not a 2-D aligned matrix"
            )

        grouped: dict[str, list[np.ndarray]] = {}
        stale = 0
        invalid = 0
        aligned = sorted(zip(raw_ids, raw_vectors), key=lambda item: item[0])
        for raw_id, vector in aligned:
            entry = manifest.get(raw_id) or {}
            if entry.get("model") != embedding_client.model:
                stale += 1
                continue
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
                invalid += 1
                continue
            resource_id = dataset_id_to_resource_id(raw_id)
            grouped.setdefault(resource_id, []).append(vector)

        embedding_ids = sorted(grouped)
        vectors = np.asarray(
            [np.mean(grouped[resource_id], axis=0) for resource_id in embedding_ids],
            dtype=np.float32,
        )
        duplicates = sum(len(items) - 1 for items in grouped.values())
        logger.info(
            "Loaded %d hybrid metadata vectors from %s "
            "(%d stale model, %d invalid, %d duplicate IDs merged).",
            len(embedding_ids), cache_path, stale, invalid, duplicates,
        )
        return cls(
            lexical_index,
            embedding_ids,
            vectors,
            embedding_client,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.lexical_index)

    def dataset_filepath(self, resource_id: str) -> Path:
        return self.lexical_index.dataset_filepath(resource_id)

    def get(self, resource_id: str) -> Optional[dict]:
        return self.lexical_index.get(resource_id)

    @staticmethod
    def _query_text(keywords: str | Iterable[str]) -> str:
        if isinstance(keywords, str):
            return " ".join(keywords.split())
        # Keyword lists represent a set for both BM25 and semantic retrieval.
        # Sorting makes query embeddings stable even when callers pass a set.
        cleaned = {
            str(keyword).strip()
            for keyword in keywords
            if keyword is not None and str(keyword).strip()
        }
        return " ".join(sorted(cleaned))

    def _cache_query_vector(self, text: str, vector: np.ndarray) -> None:
        self._query_cache[text] = vector
        self._query_cache.move_to_end(text)
        while len(self._query_cache) > self.query_cache_size:
            self._query_cache.popitem(last=False)

    def _embed_queries(self, texts: list[str]) -> dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(text for text in texts if text))
        missing = [text for text in unique_texts if text not in self._query_cache]
        if missing:
            vectors = self.embedding_client.embed(
                missing, input_type=self.query_input_type
            )
            if len(vectors) != len(missing):
                raise ValueError(
                    f"Embedding provider returned {len(vectors)} vectors "
                    f"for {len(missing)} queries"
                )
            expected_dim = self._embedding_vectors.shape[1]
            for text, raw_vector in zip(missing, vectors):
                vector = np.asarray(raw_vector, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if (
                    vector.ndim != 1
                    or len(vector) != expected_dim
                    or not np.isfinite(vector).all()
                    or not np.isfinite(norm)
                    or norm <= 0
                ):
                    raise ValueError(
                        f"Invalid query embedding for {text!r}: "
                        f"shape={vector.shape}, expected ({expected_dim},)"
                    )
                self._cache_query_vector(text, vector / norm)
        return {text: self._query_cache[text] for text in unique_texts}

    def _lexical_results(
        self, keywords: str | Iterable[str]
    ) -> list[SearchResult]:
        # Fusion needs each positive lexical score, not only the caller's
        # requested window: a moderate BM25 result can still win after its
        # semantic score is added.
        return self.lexical_index.search(
            keywords, top_k=max(1, len(self.lexical_index))
        )

    def _fuse(
        self,
        lexical_results: list[SearchResult],
        semantic_values: np.ndarray,
        top_k: int,
        only_available: bool,
    ) -> list[SearchResult]:
        lexical_by_id = {result.resource_id: result for result in lexical_results}
        semantic_by_id = {
            resource_id: float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
            for resource_id, cosine in zip(self._embedding_ids, semantic_values)
        }

        candidate_ids = set(lexical_by_id) | set(semantic_by_id)
        if only_available:
            candidate_ids = {
                resource_id
                for resource_id in candidate_ids
                if self.dataset_filepath(resource_id).exists()
            }

        max_lexical = max(
            (lexical_by_id[rid].score for rid in candidate_ids if rid in lexical_by_id),
            default=0.0,
        )

        # RRF (see the module docstring) needs each side's RANK, not its
        # raw score — ``lexical_results`` is already ranked (its list
        # order IS the rank); ``semantic_by_id`` isn't, so it's ranked
        # here once, up front, rather than per candidate below.
        lexical_rank_by_id: dict[str, int] = {}
        semantic_rank_by_id: dict[str, int] = {}
        if self.fusion_method == "rrf":
            lexical_rank_by_id = {rid: i + 1 for i, rid in enumerate(lexical_by_id)}
            semantic_rank_by_id = {
                rid: i + 1
                for i, rid in enumerate(
                    sorted(semantic_by_id, key=lambda r: -semantic_by_id[r])
                )
            }

        scored: list[tuple[str, float, float, float]] = []
        for resource_id in candidate_ids:
            lexical = (
                lexical_by_id[resource_id].score / max_lexical
                if max_lexical > 0 and resource_id in lexical_by_id
                else 0.0
            )
            semantic = semantic_by_id.get(resource_id, 0.0)
            if self.fusion_method == "rrf":
                # Each side contributes independently — a candidate
                # missing from one side gets 0 from it, no cross-penalty,
                # so ranking #1 on EITHER side alone already scores well.
                hybrid = 0.0
                if resource_id in lexical_rank_by_id:
                    hybrid += 1.0 / (self.rrf_k + lexical_rank_by_id[resource_id])
                if resource_id in semantic_rank_by_id:
                    hybrid += 1.0 / (self.rrf_k + semantic_rank_by_id[resource_id])
            else:
                hybrid = self.lexical_weight * lexical + self.semantic_weight * semantic
            scored.append((resource_id, hybrid, lexical, semantic))

        scored.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        results: list[SearchResult] = []
        for resource_id, hybrid, lexical, semantic in scored:
            lexical_result = lexical_by_id.get(resource_id)
            if lexical_result is not None:
                result = replace(
                    lexical_result,
                    score=hybrid,
                    lexical_score=lexical,
                    semantic_score=semantic,
                )
            else:
                record = self.get(resource_id)
                if record is None:
                    continue
                filepath = self.dataset_filepath(resource_id)
                result = SearchResult(
                    resource_id=resource_id,
                    dataset_id=record.get("dataset_id", resource_id),
                    title=record.get("title", ""),
                    publisher=record.get("publisher"),
                    tags=record.get("tags") or [],
                    score=hybrid,
                    matched_terms=[],
                    csv_path=str(filepath),
                    csv_exists=filepath.exists(),
                    dataset_url=record.get("dataset_url"),
                    source=self.source,
                    lexical_score=lexical,
                    semantic_score=semantic,
                )
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    def search_many(
        self,
        queries: list[str | Iterable[str]],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[list[SearchResult]]:
        """Search multiple keyword sets, batching uncached query embeddings."""
        texts = [self._query_text(query) for query in queries]
        lexical_batches = [self._lexical_results(text) for text in texts]
        try:
            vectors = self._embed_queries(texts)
        except Exception:
            logger.warning(
                "Query embedding failed; using lexical-only retrieval for this batch.",
                exc_info=True,
            )
            return [
                self.lexical_index.search(text, top_k=top_k, only_available=only_available)
                for text in texts
            ]

        semantic_columns: list[Optional[np.ndarray]] = [None] * len(texts)
        nonempty_positions = [i for i, text in enumerate(texts) if text]
        if nonempty_positions:
            # One matrix multiplication for the whole greedy-search round is
            # substantially cheaper than one corpus scan per candidate term.
            query_matrix = np.asarray(
                [vectors[texts[i]] for i in nonempty_positions], dtype=np.float32
            )
            semantic_matrix = self._embedding_vectors @ query_matrix.T
            for column, position in enumerate(nonempty_positions):
                semantic_columns[position] = semantic_matrix[:, column]

        results = []
        for text, lexical_results, semantic_values in zip(
            texts, lexical_batches, semantic_columns
        ):
            if not text:
                results.append(
                    self.lexical_index.search(
                        text, top_k=top_k, only_available=only_available
                    )
                )
                continue
            assert semantic_values is not None
            results.append(
                self._fuse(
                    lexical_results, semantic_values, top_k, only_available
                )
            )
        return results

    def search(
        self,
        keywords: str | Iterable[str],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[SearchResult]:
        return self.search_many([keywords], top_k, only_available)[0]

    def semantic_search_many(
        self,
        queries: list[str | Iterable[str]],
        top_k: int = 10,
        only_available: bool = False,
    ) -> list[list[SearchResult]]:
        """Pure cosine ranking over the metadata embeddings — no lexical
        fusion. The dense retriever of the retrieval panel (see
        ``orqa.benchmark.retrieval_panel.RetrieverPanel``) needs a ranking
        driven ONLY by the embedded question, independent of ``search_many``'s
        BM25 half, so it is a genuinely separate signal from
        ``lexical_question`` rather than a re-weighting of the same one.
        """
        texts = [self._query_text(query) for query in queries]
        try:
            vectors = self._embed_queries(texts)
        except Exception:
            logger.warning(
                "Query embedding failed; semantic-only retrieval unavailable "
                "for this batch.",
                exc_info=True,
            )
            return [[] for _ in texts]

        results: list[list[SearchResult]] = []
        for text in texts:
            if not text:
                results.append([])
                continue
            query_vector = vectors[text]
            cosine = self._embedding_vectors @ query_vector
            order = np.argsort(-cosine)

            batch: list[SearchResult] = []
            for idx in order:
                resource_id = self._embedding_ids[idx]
                if only_available and not self.dataset_filepath(resource_id).exists():
                    continue
                record = self.get(resource_id)
                if record is None:
                    continue
                filepath = self.dataset_filepath(resource_id)
                score = float(np.clip((cosine[idx] + 1.0) / 2.0, 0.0, 1.0))
                batch.append(
                    SearchResult(
                        resource_id=resource_id,
                        dataset_id=record.get("dataset_id", resource_id),
                        title=record.get("title", ""),
                        publisher=record.get("publisher"),
                        tags=record.get("tags") or [],
                        score=score,
                        matched_terms=[],
                        csv_path=str(filepath),
                        csv_exists=filepath.exists(),
                        dataset_url=record.get("dataset_url"),
                        source=self.source,
                        semantic_score=score,
                    )
                )
                if len(batch) >= top_k:
                    break
            results.append(batch)
        return results
