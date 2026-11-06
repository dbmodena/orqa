"""LiteLLM embedding client, YAML-configured like :class:`LLMClient`, and the
question embedder of Solr KNN retrieval (:class:`QueryEmbedder`)."""

import logging
import os
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml
from litellm import embedding as litellm_embedding

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Thin LiteLLM embedding wrapper configured from litellm.yaml."""

    def __init__(self, config_path: Path, batch_size: int = 64):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.model: str = config["embedding_model"]
        self.batch_size = batch_size
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 10.0)

        provider = self.model.split("/")[0] if "/" in self.model else ""
        provider_params = (config.get("provider_params") or {}).get(provider) or {}
        # Router-only keys like drop_params are fine to pass to litellm.embedding;
        # resolve os.environ/ indirections the same way litellm does for chat.
        self.provider_params = {
            k: (
                os.environ.get(v.split("/", 1)[1], "")
                if isinstance(v, str) and v.startswith("os.environ/")
                else v
            )
            for k, v in provider_params.items()
        }

    def embed(
        self,
        texts: list[str],
        input_type: str | None = None,
        *,
        output_dimensions: int | None = None,
        truncate: str | None = None,
    ) -> list[list[float]]:
        """Embed texts, optionally selecting the provider's retrieval role.

        Existing document-embedding callers omit ``input_type`` and retain
        their current cache semantics. Hybrid search uses ``search_query``
        for query vectors when configured.

        ``output_dimensions`` (LiteLLM's ``dimensions``, OCI's
        ``outputDimensions``) and ``truncate`` exist for Solr KNN: LakeGen's
        stored vectors are cohere.embed-v4.0 at 1024 dimensions, while the
        model's default is 1536, so a query vector must ask for 1024
        explicitly to live in the same space. A vector of another length is
        rejected at once — it is a wrong request, not a transient failure,
        so it is not retried.
        """
        request_params = dict(self.provider_params)
        if input_type:
            request_params["input_type"] = input_type
        if output_dimensions is not None:
            request_params["dimensions"] = int(output_dimensions)
        if truncate:
            request_params["truncate"] = truncate
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = litellm_embedding(
                        model=self.model, input=batch, **request_params
                    )
                    data = sorted(response["data"], key=lambda d: d["index"])
                    if len(data) != len(batch):
                        raise ValueError(
                            f"Embedding provider returned {len(data)} vectors "
                            f"for a batch of {len(batch)} texts"
                        )
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Embedding batch %d failed (attempt %d/%d): %s",
                        start // self.batch_size,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                    if attempt + 1 < self.max_retries:
                        time.sleep(self.retry_delay)
            else:
                raise RuntimeError(
                    f"Embedding batch starting at {start} failed after "
                    f"{self.max_retries} attempts"
                ) from last_error
            if output_dimensions is not None:
                wrong = {len(d["embedding"]) for d in data} - {int(output_dimensions)}
                if wrong:
                    raise ValueError(
                        f"{self.model} returned {sorted(wrong)}-dimensional vectors; "
                        f"{output_dimensions} were requested"
                    )
            vectors.extend(d["embedding"] for d in data)
        return vectors


def normalize_query_text(text: str) -> str:
    """A question as LakeGen embeds it (``lakegen.llm.embeddings.
    _normalize_query``): NFKC, every Unicode control/format character
    (category ``C*``: zero-width spaces, soft hyphens, newlines) turned into a
    space, whitespace collapsed. Any other spelling of the same question gives
    a different vector, so OrQA's dense ranking would no longer be LakeGen's.
    """
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    normalized = " ".join(normalized.split())
    if not normalized:
        raise ValueError("a blank question has no query embedding")
    return normalized


class QueryEmbedder:
    """Questions -> query vectors in the space of a Solr core's stored
    ``table_embedding``, embedded exactly as LakeGen embeds a question:
    :func:`normalize_query_text`, cohere.embed-v4.0 (the litellm.yaml
    ``embedding_model``), ``input_type`` SEARCH_QUERY, ``truncate`` END and
    ``output_dimensions`` 1024 (``lakegen.llm.embeddings.OCIEmbedding``).
    OrQA's own metadata caches are 1536-d and cannot be reused.

    Called as ``embedder(texts) -> np.ndarray`` of shape ``(len(texts),
    dimensions)`` — the ``embedder`` callable ``SolrDatasetIndex`` takes.
    Each question is embedded once per process: an LRU cache of
    ``cache_size`` normalized texts sits in front of the provider, because
    the retrieval panel ranks the same question for the dense and the hybrid
    retriever, and a question is re-gated after each rewrite.

    Args:
        client: The LiteLLM embedding client (OCI cohere.embed-v4.0).
        dimensions: Vector length of the core's ``table_embedding``.
        input_type: Provider role (LiteLLM maps ``search_query`` to OCI's
            ``SEARCH_QUERY``).
        truncate: Where an over-long input is cut (LakeGen: ``END``).
        cache_size: Questions kept in the LRU cache.
    """

    def __init__(
        self,
        client: EmbeddingClient,
        *,
        dimensions: int = 1024,
        input_type: str = "search_query",
        truncate: str = "END",
        cache_size: int = 2048,
    ):
        self.client = client
        self.dimensions = int(dimensions)
        self.input_type = input_type
        self.truncate = truncate
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # The panel may rank questions from several worker threads.
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.client.model

    def __call__(self, texts: list[str]) -> np.ndarray:
        normalized = [normalize_query_text(text) for text in texts]
        with self._lock:
            found = {t: self._cache[t] for t in normalized if t in self._cache}
            for text in found:
                self._cache.move_to_end(text)
        missing = [t for t in dict.fromkeys(normalized) if t not in found]
        if missing:
            vectors = np.asarray(
                self.client.embed(
                    missing,
                    input_type=self.input_type,
                    output_dimensions=self.dimensions,
                    truncate=self.truncate,
                ),
                # float64 as the provider sent it: Solr parses the JSON vector
                # itself, so OrQA's request carries LakeGen's exact numbers.
                dtype=np.float64,
            )
            if not np.isfinite(vectors).all():
                raise ValueError(f"{self.model} returned a non-finite query vector")
            found.update(zip(missing, vectors))
            with self._lock:
                for text, vector in zip(missing, vectors):
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        if not normalized:
            return np.zeros((0, self.dimensions))
        return np.stack([found[text] for text in normalized])
