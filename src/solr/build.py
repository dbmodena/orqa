"""The ``build-solr-core`` step: a portal's Solr core from OrQA's own data.

What LakeGen's ``solr_builder`` did (``build_portal_core`` + ``index``), done
by OrQA on its own Solr (``solr.instance``), so that the documents LakeGen
searches come from the metadata OrQA generates its questions from:

1. Documents — one per normalized-metadata record whose table OrQA serves (a
   non-empty ``*.<format>`` file under ``datasets_path``, matched by
   resource id the way retrieval matches it), shaped as LakeGen shaped them
   (:func:`solr_document`): the columns flattened into aligned ``columns.*``
   arrays, dates in Solr form, empty values left out.
2. Vectors — each document's ``metadata-v1`` text (:func:`metadata_v1`)
   embedded by ``search.solr.embedding_model`` as a document: bge-m3 on
   OrQA's embedding server (``solr.embedder``), or cohere.embed-v4.0 as a
   SEARCH_DOCUMENT through OCI. A text already embedded by that model (the
   serving core's vectors, the on-disk :class:`VectorCache`) is not
   embedded again, so a rebuild pays only for new or changed tables.
3. The build goes into ``<core>_build``, is checked (every document indexed,
   a stored vector's KNN finds hits) and compared with the serving core.
4. A portal with no core yet is served at once. An existing core is replaced
   only by ``python src/solr/cli.py publish <core>``, after the comparison
   has been read: a rebuilt corpus changes what LakeGen is scored on.

:func:`reembed_core` (``python src/solr/cli.py reembed``) re-makes only the
vectors of a core's stored documents, for a change of embedding model.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from solr.index import SolrDatasetIndex, local_files_by_resource_id

METADATA_V1 = "metadata-v1"
DATE_FIELDS = ("created_at", "modified_at")
# Texts per embedding request (LakeGen's indexer: 16).
EMBED_BATCH = 16
# Stored fields a document's metadata-v1 text is made of, plus its vector.
_REUSE_FL = (
    "resource_id,title,description,tags,columns.name,columns.description,"
    "representation_version,embedding_model,{vector}"
)


# ------------------------------------------------------------ documents

def solr_timestamp(value: Any) -> Optional[str]:
    """An ISO timestamp as Solr stores it (UTC, milliseconds, ``Z``);
    unparsable text as given, None when empty."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def solr_document(record: dict[str, Any], indexed_ts: str) -> dict[str, Any]:
    """One normalized record as a flat Solr document, its columns aligned.

    Every ``columns.*`` array has one entry per column: Solr drops an empty
    string, which would shift the arrays the client zips back together. So
    a missing name, label or type falls back, and descriptions are kept
    only when every column has one.
    """
    document = {
        key: value
        for key, value in record.items()
        if key != "columns" and value not in (None, "", [])
    }
    for field in DATE_FIELDS:
        if field in document:
            stamp = solr_timestamp(document[field])
            if stamp:
                document[field] = stamp
            else:
                document.pop(field)
    columns = [column for column in record.get("columns") or [] if isinstance(column, dict)]
    if columns:
        names = [str(column.get("name") or f"column_{i}") for i, column in enumerate(columns)]
        document["columns.name"] = names
        document["columns.label"] = [
            str(column.get("label") or name) for column, name in zip(columns, names, strict=True)
        ]
        document["columns.type"] = [str(column.get("type") or "text") for column in columns]
        if all(str(column.get("description") or "").strip() for column in columns):
            document["columns.description"] = [str(column["description"]) for column in columns]
    document["indexed_ts"] = indexed_ts
    return document


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def metadata_v1(document: dict[str, Any]) -> str:
    """The text a table is embedded as (LakeGen's ``metadata-v1``): title,
    description, tags, column names and descriptions, no rows. Accepts a
    record (``columns`` as dicts) or a flat Solr document (``columns.*``);
    "" when the document has none of those fields."""
    columns = document.get("columns")
    if isinstance(columns, list):
        names = _strings([c.get("name") for c in columns if isinstance(c, dict)])
        descriptions = _strings([c.get("description") for c in columns if isinstance(c, dict)])
    else:
        names = _strings(document.get("columns.name") or document.get("columns_name"))
        descriptions = _strings(document.get("columns.description") or document.get("columns_description"))
    sections = [
        ("Title", _strings(document.get("title"))),
        ("Description", _strings(document.get("description"))),
        ("Tags", _strings(document.get("tags"))),
        ("Column names", names),
        ("Column descriptions", descriptions),
    ]
    rendered = [f"{label}: {' | '.join(values)}" for label, values in sections if values]
    if not rendered:
        return ""
    return "\n".join(("Represent this table metadata for information retrieval.", *rendered))


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def served_documents(
    records: Iterable[dict[str, Any]], local_files: dict[str, Path], indexed_ts: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The documents of the records whose table OrQA serves, and a report
    of what was left out."""
    served = {rid: path for rid, path in local_files.items() if path.stat().st_size > 0}
    documents, without_table, duplicates = [], 0, []
    claimed: set[str] = set()
    for record in records:
        resource_id = str(record.get("resource_id") or "").strip()
        if resource_id not in served:
            without_table += 1
            continue
        if resource_id in claimed:
            duplicates.append(resource_id)
            continue
        claimed.add(resource_id)
        documents.append(solr_document(record, indexed_ts))
    report = {
        "records_without_a_served_table": without_table,
        "duplicate_records": duplicates[:20],
        "empty_tables": len(local_files) - len(served),
        "tables_without_a_record": len(set(served) - claimed),
    }
    return documents, report


# ------------------------------------------------------------- vectors

def _filter_value(filters: Sequence[str], field: str) -> str:
    """The value a KNN filter requires of ``field`` (``field:"value"``)."""
    for spec in filters:
        name, _, value = spec.partition(":")
        if name.strip() == field:
            return value.strip().strip('"')
    raise ValueError(f"tasks.search.solr.knn_filters has no {field!r} filter to label the vectors with")


def stored_vectors(client: Any, core: str, vector_field: str, version: str, model: str) -> dict[str, list[float]]:
    """``{sha256(metadata-v1 text): vector}`` of a core's documents embedded
    with this representation and model."""
    vectors: dict[str, list[float]] = {}
    cursor = "*"
    while True:
        payload = client.select(
            ["*:*"], core=core, indent=False, restore_columns=False, rows=500,
            fl=_REUSE_FL.format(vector=vector_field), sort="resource_id asc", cursorMark=cursor,
        )
        docs = payload["response"].get("docs", [])
        for doc in docs:
            vector = doc.get(vector_field)
            if vector and doc.get("representation_version") == version and doc.get("embedding_model") == model:
                text = metadata_v1(doc)
                if text:
                    vectors[_text_key(text)] = vector
        following = payload.get("nextCursorMark")
        if not docs or not following or following == cursor:
            return vectors
        cursor = following


def embed_documents(
    documents: list[dict[str, Any]],
    embed,
    known: dict[str, list[float]],
    *,
    vector_field: str,
    version: str,
    model: str,
) -> dict[str, int]:
    """Give every document its vector and provenance in place: reused from
    ``known`` when its text is there, else from ``embed(texts)``. A document
    with an empty text gets no vector (LakeGen's indexer skips it too)."""
    pending: list[tuple[dict[str, Any], str]] = []
    counts = Counter()
    for document in documents:
        text = metadata_v1(document)
        if not text:
            counts["without_text"] += 1
            continue
        vector = known.get(_text_key(text))
        if vector is None:
            pending.append((document, text))
            continue
        document.update({vector_field: vector, "representation_version": version, "embedding_model": model})
        counts["reused"] += 1
    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start : start + EMBED_BATCH]
        vectors = embed([text for _, text in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(f"the embedder returned {len(vectors)} vectors for {len(batch)} texts")
        for (document, _), vector in zip(batch, vectors, strict=True):
            document.update({vector_field: [float(v) for v in vector], "representation_version": version, "embedding_model": model})
        counts["embedded"] += len(batch)
        print(f"[embed] embedded {counts['embedded']}/{len(pending)}", flush=True)
    return dict(counts)


# -------------------------------------------------------------- compare

def compare_cores(client: Any, serving: str, built: str, vector_field: str) -> dict[str, Any]:
    """Ids added and removed, and per field how many kept documents changed."""
    def stored(core: str) -> dict[str, dict[str, Any]]:
        docs, cursor = {}, "*"
        while True:
            payload = client.select(
                ["*:*"], core=core, indent=False, restore_columns=False, rows=500,
                fl="*", sort="resource_id asc", cursorMark=cursor,
            )
            page = payload["response"].get("docs", [])
            for doc in page:
                docs[doc["resource_id"]] = doc
            following = payload.get("nextCursorMark")
            if not page or not following or following == cursor:
                return docs
            cursor = following

    old, new = stored(serving), stored(built)
    ignore = {"_version_", "indexed_ts", vector_field}
    changed = Counter()
    vector_changed = 0
    for resource_id in set(old) & set(new):
        a, b = old[resource_id], new[resource_id]
        for field in (set(a) | set(b)) - ignore:
            if a.get(field) != b.get(field):
                changed[field] += 1
        vector_changed += a.get(vector_field) != b.get(vector_field)
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    return {
        "serving": serving, "built": built,
        "documents": {"serving": len(old), "built": len(new), "kept": len(set(old) & set(new))},
        "added": len(added), "removed": len(removed),
        "added_examples": added[:10], "removed_examples": removed[:10],
        "kept_with_field_changes": dict(changed.most_common()),
        "kept_with_a_new_vector": vector_changed,
    }


# ---------------------------------------------------------------- cache

class VectorCache:
    """Document vectors of one model, by metadata-v1 text: a JSON-lines file
    appended as vectors arrive (float32, base64), so a long embedding run
    that stops resumes where it was, and a rebuild re-embeds nothing it
    already embedded."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.vectors: dict[str, list[float]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as lines:
                for line in lines:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:  # a line cut by a crash
                        continue
                    raw = np.frombuffer(base64.b64decode(item["v"]), dtype="<f4")
                    self.vectors[item["k"]] = [float(x) for x in raw]

    def add(self, texts: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as out:
            for text, vector in zip(texts, vectors, strict=True):
                key = _text_key(text)
                self.vectors[key] = [float(x) for x in np.asarray(vector, dtype="<f4")]
                encoded = base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()
                out.write(json.dumps({"k": key, "v": encoded}) + "\n")


def vector_cache(instance: Any, model: str, dimensions: int) -> VectorCache:
    return VectorCache(instance.home / "embeddings" / f"{model.replace('/', '_')}-{dimensions}.jsonl")


# ---------------------------------------------------------- build cores

def load_build(
    instance: Any, client: Any, core: str, documents: list[dict[str, Any]],
    *, vector_field: str, knn_filters: Sequence[str],
) -> str:
    """Post ``documents`` into a fresh ``<core>_build`` and check it: every
    document indexed, a stored vector's KNN (under ``knn_filters``) finds
    hits. Returns the build core's name."""
    from solr.instance import build_core_name

    build = build_core_name(core)
    if build in instance.cores():
        instance.unload_core(build)
    instance.create_core(build)
    instance.add_documents(build, documents)
    built = SolrDatasetIndex(
        client, build, local_files={}, source="", datasets_path=Path("."),
        datasets_format="parquet", vector_field=vector_field, knn_filters=knn_filters,
    )
    count = built.ping()
    if count != len(documents):
        raise RuntimeError(f"{build} holds {count} documents, {len(documents)} were posted")
    built.check_knn()
    return build


def write_report(instance: Any, name: str, report: dict[str, Any]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    path = instance.home / "builds" / f"{name}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return path


def reembed_core(
    instance: Any, client: Any, core: str, embedding_client: Any,
    *, vector_field: str = "table_embedding", dimensions: int = 1024,
) -> dict[str, Any]:
    """``core``'s documents, unchanged, with every vector re-made from their
    metadata-v1 text by ``embedding_client`` (labelled with its model), in
    ``<core>_build``. Works from the stored documents alone, so it serves
    cores OrQA has no metadata for (LakeGen's benchmark lakes)."""
    model = embedding_client.model
    documents = []
    cursor = "*"
    while True:
        payload = client.select(
            ["*:*"], core=core, indent=False, restore_columns=False, rows=500,
            fl="*", sort="resource_id asc", cursorMark=cursor,
        )
        page = payload["response"].get("docs", [])
        for doc in page:
            for field in ("_version_", vector_field, "representation_version", "embedding_model"):
                doc.pop(field, None)
            documents.append(doc)
        following = payload.get("nextCursorMark")
        if not page or not following or following == cursor:
            break
        cursor = following
    cache = vector_cache(instance, model, dimensions)
    print(f"[reembed] {core}: {len(documents)} documents, {len(cache.vectors)} vectors cached for {model}", flush=True)

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = embedding_client.embed(texts, input_type="search_document", output_dimensions=dimensions)
        cache.add(texts, vectors)
        return vectors

    counts = embed_documents(
        documents, embed, cache.vectors, vector_field=vector_field, version=METADATA_V1, model=model,
    )
    filters = (f'representation_version:"{METADATA_V1}"', f'embedding_model:"{model}"')
    build = load_build(instance, client, core, documents, vector_field=vector_field, knn_filters=filters)
    report = {
        "core": core, "build": build, "model": model, "documents": len(documents), "vectors": counts,
        "comparison": compare_cores(client, core, build, vector_field),
    }
    report["path"] = str(write_report(instance, f"{core}-reembed", report))
    return report


# ----------------------------------------------------------------- step

def build_solr_core(cfg) -> dict[str, Any]:
    """The workflow step (``python src/main.py ... --steps build-solr-core``)."""
    from orqa.agent.llm_client.EmbeddingClient import solr_embedding_client
    from solr.client import LocalSolrClient
    from solr.embedder import OllamaServer
    from solr.instance import SolrInstance

    solr = cfg.search.solr
    core = solr.core
    if not core:
        raise ValueError("build-solr-core needs tasks.search.solr.core: the core's name on OrQA's Solr")
    version = _filter_value(solr.knn_filters, "representation_version")
    model = solr.embedding_model
    if version != METADATA_V1:
        raise ValueError(f"build-solr-core writes {METADATA_V1} vectors; the KNN filters ask for {version!r}")
    embedding_client = solr_embedding_client(model, solr.embedding_url, cfg.llm_config_path, batch_size=EMBED_BATCH)
    if solr.embedding_url is None and not model.startswith(("cohere.", "ocid1.")):
        OllamaServer.from_env().ensure(model)

    instance = SolrInstance.from_env()
    if instance.start():
        print(f"[build-solr-core] started OrQA's Solr at {instance.url}", flush=True)
    client = LocalSolrClient(base_url=instance.url, timeout=solr.timeout, retries=solr.retries)

    records = json.loads(Path(cfg.normalized_metadata_filepath).read_text(encoding="utf-8"))
    indexed_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    local_files = local_files_by_resource_id(cfg.datasets_path, cfg.datasets_format)
    documents, report = served_documents(records, local_files, indexed_ts)
    print(
        f"[build-solr-core] {len(records)} records -> {len(documents)} documents "
        f"({report['records_without_a_served_table']} records without a served table, "
        f"{report['tables_without_a_record']} tables without a record, {report['empty_tables']} empty tables)",
        flush=True,
    )
    if not documents:
        raise RuntimeError(f"no record of {cfg.normalized_metadata_filepath} has a table under {cfg.datasets_path}")

    cores = instance.cores()
    cache = vector_cache(instance, model, solr.embedding_dimensions)
    known = dict(cache.vectors)
    if core in cores:
        known.update(stored_vectors(client, core, solr.vector_field, version, model))

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = embedding_client.embed(
            texts, input_type="search_document",
            output_dimensions=solr.embedding_dimensions, truncate="END",
        )
        cache.add(texts, vectors)
        return vectors

    report["vectors"] = embed_documents(
        documents, embed, known, vector_field=solr.vector_field, version=version, model=model
    )
    build = load_build(
        instance, client, core, documents, vector_field=solr.vector_field, knn_filters=solr.knn_filters,
    )
    count = len(documents)

    report.update(core=core, build=build, documents=count, built_at=indexed_ts)
    if core in cores:
        report["comparison"] = compare_cores(client, core, build, solr.vector_field)
    else:
        instance.publish(core)
        report["published"] = True
    path = write_report(instance, core, report)

    if report.get("published"):
        print(f"[build-solr-core] core {core!r} created with {count} documents (report {path})", flush=True)
    else:
        comparison = report["comparison"]
        print(
            f"[build-solr-core] built {build!r}: {count} documents, {comparison['added']} added, "
            f"{comparison['removed']} removed, {comparison['documents']['kept']} kept "
            f"(field changes {comparison['kept_with_field_changes']}). {core!r} still serves the "
            f"old documents: read {path}, then `python src/solr/cli.py publish {core}`.",
            flush=True,
        )
    return report
