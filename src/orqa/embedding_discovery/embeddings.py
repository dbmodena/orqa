"""
Metadata embeddings for candidates discovery.

Builds one text document per dataset from the raw normalized metadata
(title, publisher, tags, description, per-column name/label/type), embeds
them through :class:`EmbeddingClient` (the LiteLLM embedding API wrapper —
see ``agent.llm_client.EmbeddingClient``) and caches the vectors on disk so
reruns never re-pay the API cost for unchanged metadata.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from ..agent.llm_client.EmbeddingClient import EmbeddingClient
from ..utils import pl_scan_dataset
from ..utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)

SEP = "__"


def dataset_id_to_resource_id(dataset_id: str) -> str:
    """Map a dataset filename stem to its metadata resource id.

    Stems are either ``<resource_id>`` or ``<name>__<resource_id>``.
    """
    return dataset_id.split(SEP)[-1] if SEP in dataset_id else dataset_id


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
    max_chars: int = 4000,
) -> str:
    """Build the text document embedded for one dataset.

    Falls back to the CSV header (via a zero-row polars scan) when the
    metadata carries no column information (CKAN portals).
    """
    lines = [
        f"Title: {record.get('title', '')}",
        f"Publisher: {record.get('publisher', '')}",
        f"Tags: {', '.join(record.get('tags') or [])}",
        f"Description: {(record.get('description') or '')[:1500]}",
    ]

    columns = record.get("columns") or []
    if columns:
        lines.append("Columns:")
        for col in columns:
            name = col.get("name", "")
            label = col.get("label", "")
            ctype = col.get("type", "")
            label_part = f" ({label})" if label and label != name else ""
            type_part = f" [{ctype}]" if ctype else ""
            lines.append(f"- {name}{label_part}{type_part}")
    elif dataset_path is not None:
        try:
            schema = pl_scan_dataset(dataset_path, scan_opts or {}).collect_schema()
            lines.append("Columns:")
            lines.extend(f"- {name} [{dtype}]" for name, dtype in schema.items())
        except Exception as exc:
            logger.warning(
                "Could not read CSV header for %s (%s); embedding metadata only.",
                dataset_path.name,
                exc,
            )

    return "\n".join(lines)[:max_chars]


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
