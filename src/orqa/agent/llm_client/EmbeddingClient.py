"""LiteLLM embedding client, YAML-configured like :class:`LLMClient`, the
Ollama client of the Solr cores' local model (:class:`OllamaEmbeddingClient`),
and the question embedder of Solr KNN retrieval (:class:`QueryEmbedder`)."""

import logging
import os
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests
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
        # Keys like drop_params are fine to pass to litellm.embedding;
        # os.environ/ indirections are resolved as LLMClient does for chat.
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


# LakeGen embeds a question behind this prefix when the plain question makes
# the model return NaN (lakegen.llm.embeddings.QUERY_NON_FINITE_FALLBACK_PREFIX).
QUERY_NON_FINITE_FALLBACK_PREFIX = "Represent this question for dataset retrieval. "
_NAN_MARKERS = ("unsupported value: nan", "non-finite", "zero-norm")


class EmbeddingError(RuntimeError):
    """The embedding server gave no usable vector."""


class OllamaEmbeddingClient:
    """An Ollama embedding model (``bge-m3``) called exactly as LakeGen
    calls it (``lakegen.llm.embeddings.OllamaMultilingualEmbedding``, through
    llama-index's ``OllamaEmbedding``): ``POST /api/embed`` with each text
    stripped and no options. Documents go ``batch_size`` (LakeGen: 16) per
    request; a question goes alone and, when the model returns NaN for it,
    once more behind LakeGen's retrieval prefix.

    Same ``embed`` signature as :class:`EmbeddingClient`, so
    :class:`QueryEmbedder` and the Solr build take either: ``input_type``
    ``search_query`` selects the question path, ``output_dimensions`` is
    checked (the model has one size), ``truncate`` is Ollama's own (it cuts
    an over-long text to the model's context).

    Args:
        model: The Ollama model, also the vectors' ``embedding_model`` label.
        base_url: The Ollama server.
        batch_size: Documents per request.
        options: Ollama options of document requests only (``num_thread`` to
            re-embed faster); questions always go without, like LakeGen's.
        timeout: Seconds per request.
        retries: Attempts per request on a connection error or HTTP 5xx.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        batch_size: int = 16,
        options: dict | None = None,
        timeout: float = 600.0,
        retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = max(1, int(batch_size))
        self.options = dict(options or {})
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self._session = requests.Session()

    def _request(self, texts: list[str], options: dict) -> list[list[float]]:
        body = {"model": self.model, "input": [text.strip() for text in texts]}
        if options:
            body["options"] = options
        last = ""
        for attempt in range(self.retries):
            if attempt:
                time.sleep(2 ** attempt)
            try:
                response = self._session.post(f"{self.base_url}/api/embed", json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code == 200:
                vectors = response.json().get("embeddings") or []
                if len(vectors) != len(texts):
                    raise EmbeddingError(f"{self.model} returned {len(vectors)} vectors for {len(texts)} texts")
                return vectors
            last = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code < 500 or any(m in last.casefold() for m in _NAN_MARKERS):
                break
        raise EmbeddingError(f"Ollama {self.base_url} {self.model}: {last}")

    @staticmethod
    def _check(vector: list[float], dimensions: int | None) -> list[float]:
        values = [float(v) for v in vector]
        if not values or not all(np.isfinite(values)):
            raise EmbeddingError("non-finite vector")
        if dimensions is not None and len(values) != int(dimensions):
            raise EmbeddingError(f"{len(values)}-dimensional vector, {dimensions} expected")
        if not np.linalg.norm(values) > 1e-12:
            raise EmbeddingError("zero-norm vector")
        return values

    def _query(self, text: str, dimensions: int | None) -> list[float]:
        try:
            return self._check(self._request([text], {})[0], dimensions)
        except EmbeddingError as exc:
            if not any(m in str(exc).casefold() for m in _NAN_MARKERS):
                raise
        return self._check(self._request([QUERY_NON_FINITE_FALLBACK_PREFIX + text], {})[0], dimensions)

    def embed(
        self,
        texts: list[str],
        input_type: str | None = None,
        *,
        output_dimensions: int | None = None,
        truncate: str | None = None,
    ) -> list[list[float]]:
        if input_type == "search_query":
            return [self._query(text, output_dimensions) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._check(v, output_dimensions) for v in self._request(batch, self.options))
        return vectors


def solr_embedding_client(
    model: str,
    url: str | None,
    llm_config_path: Path,
    *,
    batch_size: int = 16,
    options: dict | None = None,
):
    """The client of the model a Solr core's vectors come from, by LakeGen's
    rule (``lakegen.llm.embeddings.get_embedding_model``): a ``cohere.`` or
    ``ocid1.`` model through OCI (litellm.yaml's ``embedding_model``, which
    must be it), anything else through Ollama at ``url`` (None: OrQA's own
    embedding server, ``solr.embedder``)."""
    if model.startswith(("cohere.", "ocid1.")):
        client = EmbeddingClient(llm_config_path / "litellm.yaml", batch_size=batch_size)
        if client.model.rsplit("/", 1)[-1] != model:
            raise ValueError(
                f"litellm.yaml embeds with {client.model!r}, but the Solr vectors are {model!r}: "
                "set its embedding_model to it"
            )
        return client
    if url is None:
        from solr.embedder import default_url

        url = default_url()
    return OllamaEmbeddingClient(model, url, batch_size=batch_size, options=options)
