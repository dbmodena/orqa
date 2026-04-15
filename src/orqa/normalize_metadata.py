"""Normalize raw metadata payloads from CKAN, ODS, and Socrata.

This module provides a small, standalone preprocessing layer that turns
source-specific metadata payloads into a flat list of homogeneous dataset
records. It is designed to be callable from higher-level scripts such as
`main.py` without requiring changes to the current ORQA pipeline.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable, Literal

from conf import OrQAConfig

MetadataSource = Literal["ckan", "ods", "socrata"]

REQUIRED_SCHEMA_KEYS = [
    "dataset_id",
    "resource_id",
    "source",
    "title",
    "description",
    "publisher",
    "tags",
    "created_at",
    "modified_at",
    "dataset_url",
    "download_url",
    "format",
    "columns",
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_metadata_records(records: list[dict], source: str) -> list[dict]:
    """Normalize a list of raw metadata records into a flat unified schema."""
    normalized_records: list[dict] = []
    source_normalized = _normalize_source(source)

    for record in records:
        if source_normalized == "ckan":
            normalized_records.extend(_normalize_ckan_record(record))
        elif source_normalized == "ods":
            item = _normalize_ods_record(record)
            if item is not None:
                normalized_records.append(item)
        elif source_normalized == "socrata":
            item = _normalize_socrata_record(record)
            if item is not None:
                normalized_records.append(item)
        else:
            raise ValueError(f"Unsupported source: {source}")

    return normalized_records


def normalize_file(
    input_path: Path | str,
    source: str,
    output_path: Path | str | None = None,
) -> list[dict]:
    """Load a raw metadata file, normalize it, and optionally write JSON output."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    normalized = normalize_metadata_records(records, source)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(normalized, file, indent=2, ensure_ascii=False)

    return normalized


def normalize_metadata(cfg: OrQAConfig) -> list[dict]:
    """Normalize the metadata file referenced by an OrQA config."""
    input_path = cfg.original_metadata_filepath
    output_path = cfg.normalized_metadata_filepath
    return normalize_file(input_path, cfg.source, output_path)


def _normalize_source(source: str) -> MetadataSource:
    source_normalized = source.strip().lower()
    if source_normalized not in {"ckan", "ods", "socrata"}:
        raise ValueError(
            "source must be one of: 'ckan', 'ods', 'socrata'"
        )
    return source_normalized  # type: ignore[return-value]


def _normalize_ckan_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = _clean_text(record.get("id")) or _clean_text(record.get("name"))
    if dataset_id is None:
        return []

    extras = _extras_to_dict(record.get("extras", []))
    title = _clean_text(record.get("title")) or _clean_text(record.get("name"))
    description = _clean_html(record.get("notes"))
    publisher = (
        _clean_text(_safe_get(record, "organization", "title"))
        or _clean_text(_safe_get(record, "organization", "name"))
        or _clean_text(extras.get("dcat_publisher_name"))
    )
    tags = _unique_strings(tag.get("display_name") or tag.get("name") for tag in record.get("tags", []))
    created_at = _clean_text(record.get("metadata_created")) or _clean_text(extras.get("dcat_issued"))
    modified_at = _clean_text(record.get("metadata_modified")) or _clean_text(extras.get("dcat_modified"))
    dataset_url = (
        _clean_text(record.get("url"))
        or _clean_text(extras.get("guid"))
    )

    normalized: list[dict[str, Any]] = []
    for resource in record.get("resources", []):
        if not isinstance(resource, dict):
            continue

        resource_id = _clean_text(resource.get("id")) or dataset_id
        download_url = _clean_text(resource.get("url"))
        state = (_clean_text(resource.get("state")) or "").lower()

        if resource_id is None:
            continue
        if state and state != "active":
            continue
        if download_url is None:
            continue

        normalized.append(
            _post_process_record(
                {
                    "dataset_id": dataset_id,
                    "resource_id": resource_id,
                    "source": "ckan",
                    "title": title,
                    "description": description,
                    "publisher": publisher,
                    "tags": tags,
                    "created_at": created_at,
                    "modified_at": _clean_text(resource.get("metadata_modified")) or modified_at,
                    "dataset_url": dataset_url,
                    "download_url": download_url,
                    "format": _normalize_format(resource.get("format")),
                    "columns": [],
                }
            )
        )

    return normalized


def _normalize_ods_record(record: dict[str, Any]) -> dict[str, Any] | None:
    dataset_id = _clean_text(record.get("dataset_id")) or _clean_text(record.get("dataset_uid"))
    if dataset_id is None:
        return None

    default_meta = _safe_get(record, "metas", "default", default={})
    dcat_meta = _safe_get(record, "metas", "dcat", default={})
    fields = record.get("fields", [])

    dataset_url = _build_ods_dataset_url(dataset_id)

    return _post_process_record(
        {
            "dataset_id": dataset_id,
            "resource_id": dataset_id,
            "source": "ods",
            "title": _clean_text(default_meta.get("title")) or dataset_id,
            "description": _clean_html(default_meta.get("description")),
            "publisher": _clean_text(default_meta.get("publisher")) or _clean_text(dcat_meta.get("creator")),
            "tags": _unique_strings(
                list(default_meta.get("keyword", []) or [])
                + list(default_meta.get("theme", []) or [])
            ),
            "created_at": _clean_text(dcat_meta.get("created")) or _clean_text(dcat_meta.get("issued")),
            "modified_at": _clean_text(default_meta.get("modified")),
            "dataset_url": dataset_url,
            "download_url": None,
            "format": None,
            "columns": [_normalize_ods_field(field) for field in fields if isinstance(field, dict)],
        }
    )


def _normalize_socrata_record(record: dict[str, Any]) -> dict[str, Any] | None:
    resource = record.get("resource", {})
    classification = record.get("classification", {})

    dataset_id = _clean_text(resource.get("id"))
    if dataset_id is None:
        return None

    domain_metadata = _domain_metadata_to_dict(classification.get("domain_metadata", []))
    dataset_url = _clean_text(resource.get("permalink")) or _build_socrata_dataset_url(dataset_id)
    download_url = _build_socrata_download_url(resource)

    return _post_process_record(
        {
            "dataset_id": dataset_id,
            "resource_id": dataset_id,
            "source": "socrata",
            "title": _clean_text(resource.get("name")) or dataset_id,
            "description": _clean_html(resource.get("description")),
            "publisher": _clean_text(resource.get("attribution")) or _clean_text(domain_metadata.get("Dataset-Information_Agency")),
            "tags": _unique_strings(
                list(classification.get("domain_tags", []) or [])
                + list(classification.get("tags", []) or [])
                + list(classification.get("categories", []) or [])
                + [_clean_text(classification.get("domain_category"))]
            ),
            "created_at": _clean_text(resource.get("createdAt")) or _clean_text(resource.get("publication_date")),
            "modified_at": _clean_text(resource.get("data_updated_at")) or _clean_text(resource.get("updatedAt")) or _clean_text(resource.get("metadata_updated_at")),
            "dataset_url": dataset_url,
            "download_url": download_url,
            "format": _normalize_format("csv" if download_url else None),
            "columns": _zip_socrata_columns(resource),
        }
    )


def _normalize_ods_field(field: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _clean_text(field.get("name")),
        "label": _clean_text(field.get("label")) or _clean_text(field.get("name")),
        "description": _clean_html(field.get("description")),
        "type": _clean_text(field.get("type")),
    }


def _zip_socrata_columns(resource: dict[str, Any]) -> list[dict[str, Any]]:
    names = resource.get("columns_field_name", []) or []
    labels = resource.get("columns_name", []) or []
    descriptions = resource.get("columns_description", []) or []
    types = resource.get("columns_datatype", []) or []

    width = max(len(names), len(labels), len(descriptions), len(types), 0)
    columns: list[dict[str, Any]] = []

    for idx in range(width):
        name = _clean_text(names[idx]) if idx < len(names) else None
        label = _clean_text(labels[idx]) if idx < len(labels) else None
        description = _clean_html(descriptions[idx]) if idx < len(descriptions) else None
        col_type = _clean_text(types[idx]) if idx < len(types) else None

        if all(value is None for value in (name, label, description, col_type)):
            continue

        columns.append(
            {
                "name": name,
                "label": label or name,
                "description": description,
                "type": col_type,
            }
        )

    return columns


def _post_process_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        "dataset_id": _clean_text(record.get("dataset_id")),
        "resource_id": _clean_text(record.get("resource_id")) or _clean_text(record.get("dataset_id")),
        "source": _clean_text(record.get("source")),
        "title": _clean_text(record.get("title")),
        "description": _clean_html(record.get("description")),
        "publisher": _clean_text(record.get("publisher")),
        "tags": _unique_strings(record.get("tags") or []),
        "created_at": _clean_text(record.get("created_at")),
        "modified_at": _clean_text(record.get("modified_at")),
        "dataset_url": _clean_text(record.get("dataset_url")),
        "download_url": _clean_text(record.get("download_url")),
        "format": _normalize_format(record.get("format")),
        "columns": _normalize_columns(record.get("columns") or []),
    }

    for key in REQUIRED_SCHEMA_KEYS:
        cleaned.setdefault(key, [] if key == "columns" or key == "tags" else None)

    return cleaned


def _normalize_columns(columns: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for column in columns:
        if not isinstance(column, dict):
            continue
        cleaned = {
            "name": _clean_text(column.get("name")),
            "label": _clean_text(column.get("label")) or _clean_text(column.get("name")),
            "description": _clean_html(column.get("description")),
            "type": _clean_text(column.get("type")),
        }
        if all(value is None for value in cleaned.values()):
            continue
        normalized.append(cleaned)
    return normalized


def _extras_to_dict(extras: Iterable[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for extra in extras:
        if not isinstance(extra, dict):
            continue
        key = _clean_text(extra.get("key"))
        if key is None:
            continue
        output[key] = extra.get("value")
    return output


def _domain_metadata_to_dict(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = _clean_text(item.get("key"))
        if key is None:
            continue
        output[key] = item.get("value")
    return output


def _safe_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _build_ods_dataset_url(dataset_id: str) -> str:
    return f"https://public.opendatasoft.com/explore/dataset/{dataset_id}/"


def _build_socrata_dataset_url(dataset_id: str) -> str:
    return f"https://data.cityofnewyork.us/d/{dataset_id}"


def _build_socrata_download_url(resource: dict[str, Any]) -> str | None:
    dataset_id = _clean_text(resource.get("id"))
    if dataset_id is None:
        return None
    return f"https://data.cityofnewyork.us/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"


def _normalize_format(value: Any) -> str | None:
    value = _clean_text(value)
    return value.upper() if value is not None else None


def _clean_html(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _unique_strings(values: Iterable[Any] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    if values is None:
        return result
    for value in values:
        cleaned = _clean_text(value)
        if cleaned is None:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
