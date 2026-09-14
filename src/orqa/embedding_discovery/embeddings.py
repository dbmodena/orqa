"""
Metadata embeddings for candidates discovery.

Builds one text document per dataset from the raw normalized metadata
(title, publisher, responsible entity, temporal coverage, tags, per-column
name/label/type, description), fitted to the embedding model's token limit;
embeds them through :class:`EmbeddingClient` (the LiteLLM embedding API wrapper —
see ``agent.llm_client.EmbeddingClient``) and caches the vectors on disk so
reruns never re-pay the API cost for unchanged metadata.
"""

import functools
import hashlib
import json
import logging
import os
from pathlib import Path

import litellm
import numpy as np
import tiktoken

from ..agent.llm_client.EmbeddingClient import EmbeddingClient
from ..utils import pl_scan_dataset
from ..utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)


def embedding_max_input_tokens(model: str, default: int = 512) -> int:
    """Per-input token limit of ``model``, from litellm's model map.

    ``default`` (Cohere embed v3's limit) covers models litellm doesn't know.
    """
    try:
        return int(litellm.get_model_info(model)["max_input_tokens"])
    except Exception:
        return default


@functools.lru_cache(maxsize=1)
def _tokenizer() -> tiktoken.Encoding:
    # Stand-in for the provider's tokenizer, which isn't available locally:
    # close enough to fit a document to the limit.
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, disallowed_special=()))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to at most ``max_tokens`` tokens, at a word boundary."""
    if max_tokens <= 0:
        return ""
    tokens = _tokenizer().encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    cut = _tokenizer().decode(tokens[:max_tokens]).rstrip("\ufffd")
    return cut[: cut.rfind(" ")] if " " in cut else cut


def load_raw_normalized_metadata(metadata_path: Path) -> dict[str, dict]:
    """Load normalized_metadata.json keeping the full records (incl. columns).

    The prompt-oriented loader in utils strips the per-column metadata, which
    is exactly what the embedding text needs — so read the raw list here.
    """
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    rv = {}
    for record in metadata:
        if not isinstance(record, dict):
            continue
        resource_id = record.get("resource_id") or record.get("dataset_id")
        if resource_id:
            rv[resource_id] = record
    return rv


def build_embedding_text(
    record: dict,
    dataset_path: Path | None = None,
    scan_opts: dict | None = None,
    max_tokens: int = 512,
) -> str:
    """Build the text document embedded for one dataset.

    The embedding API cuts inputs past the model's token limit from the end,
    so the document is fitted to ``max_tokens`` here instead: the identifying
    fields and the column list come first and are kept whole, and the
    description — usually the longest field — fills the remaining budget.

    Falls back to the CSV header (via a zero-row polars scan) when the
    metadata carries no column information (CKAN portals).
    """
    lines = [
        f"Title: {record.get('title') or ''}",
        f"Publisher: {record.get('publisher') or ''}",
    ]
    if record.get("responsible_entity"):
        lines.append(f"Responsible entity: {record['responsible_entity']}")
    if record.get("temporal_coverage"):
        lines.append(f"Temporal coverage: {record['temporal_coverage']}")
    lines.append(f"Tags: {', '.join(record.get('tags') or [])}")
    lines.extend(_column_lines(record, dataset_path, scan_opts))

    text = "\n".join(lines)
    description = record.get("description")
    if description:
        remaining = max_tokens - _count_tokens(f"{text}\nDescription: ")
        description = _truncate_to_tokens(description, remaining)
        if description:
            text = f"{text}\nDescription: {description}"
    return _truncate_to_tokens(text, max_tokens)


def _column_lines(
    record: dict, dataset_path: Path | None, scan_opts: dict | None
) -> list[str]:
    columns = record.get("columns") or []
    if columns:
        lines = ["Columns:"]
        for col in columns:
            name = col.get("name", "")
            label = col.get("label", "")
            ctype = col.get("type", "")
            label_part = f" ({label})" if label and label != name else ""
            type_part = f" [{ctype}]" if ctype else ""
            lines.append(f"- {name}{label_part}{type_part}")
        return lines
    if dataset_path is None:
        return []
    try:
        schema = pl_scan_dataset(dataset_path, scan_opts or {}).collect_schema()
    except Exception as exc:
        logger.warning(
            "Could not read CSV header for %s (%s); embedding metadata only.",
            dataset_path.name,
            exc,
        )
        return []
    return ["Columns:", *(f"- {name} [{dtype}]" for name, dtype in schema.items())]


class EmbeddingCache:
    """Disk cache: vectors in an .npz, (model, text hash) manifest as JSON."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.manifest_path = cache_path.with_name("embeddings_manifest.json")

    def _load(self) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        vectors: dict[str, np.ndarray] = {}
        manifest: dict[str, dict] = {}
        if self.cache_path.exists() and self.manifest_path.exists():
            try:
                with np.load(self.cache_path, allow_pickle=False) as data:
                    ids = data["ids"]
                    vecs = data["vectors"]
                vectors = {str(i): v for i, v in zip(ids, vecs)}
                with open(self.manifest_path) as f:
                    manifest = json.load(f)
            except Exception as exc:
                logger.warning("Embedding cache unreadable (%s); rebuilding.", exc)
                vectors, manifest = {}, {}
        return vectors, manifest

    def _save(self, vectors: dict[str, np.ndarray], manifest: dict[str, dict]):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        ids = list(vectors)
        matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
        # numpy appends ".npz" to names lacking it, so the temp file must
        # already carry the extension for os.replace to find it.
        tmp = self.cache_path.with_name(f"{self.cache_path.stem}.tmp.npz")
        np.savez_compressed(tmp, ids=np.asarray(ids), vectors=matrix)
        os.replace(tmp, self.cache_path)
        tmp_manifest = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f)
        os.replace(tmp_manifest, self.manifest_path)

    def get_or_compute(
        self, texts_by_id: dict[str, str], client: EmbeddingClient
    ) -> tuple[list[str], np.ndarray]:
        """Return (ids, vectors) for every dataset, embedding only cache misses.

        The cache is persisted after every API batch so a mid-run failure
        loses at most one batch worth of embeddings.
        """
        vectors, manifest = self._load()

        def is_hit(dataset_id: str, digest: str) -> bool:
            entry = manifest.get(dataset_id)
            return (
                entry is not None
                and entry.get("model") == client.model
                and entry.get("text_sha256") == digest
                and dataset_id in vectors
            )

        digests = {
            dataset_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for dataset_id, text in texts_by_id.items()
        }
        misses = [
            dataset_id
            for dataset_id in texts_by_id
            if not is_hit(dataset_id, digests[dataset_id])
        ]
        n_cached = len(texts_by_id) - len(misses)
        log = PipelineLogger()
        log.embedding_start(len(texts_by_id), n_cached, len(misses))

        total_batches = -(-len(misses) // client.batch_size) if misses else 0
        for batch_idx, start in enumerate(range(0, len(misses), client.batch_size), 1):
            batch_ids = misses[start : start + client.batch_size]
            log.embedding_batch(batch_idx, total_batches, len(batch_ids))
            batch_vectors = client.embed([texts_by_id[i] for i in batch_ids])
            for dataset_id, vector in zip(batch_ids, batch_vectors):
                vectors[dataset_id] = np.asarray(vector, dtype=np.float32)
                manifest[dataset_id] = {
                    "model": client.model,
                    "text_sha256": digests[dataset_id],
                }
            self._save(vectors, manifest)

        log.embedding_done(len(texts_by_id), n_cached, len(misses))

        ids = [i for i in texts_by_id if i in vectors]
        matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
        return ids, matrix
