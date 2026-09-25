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
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Literal

from conf import OrQAConfig

MetadataSource = Literal["ckan", "ods", "socrata"]

REQUIRED_SCHEMA_KEYS = [
    "dataset_id",
    "resource_id",
    "resource_name",
    "resource_description",
    "source",
    "title",
    "description",
    "publisher",
    "responsible_entity",
    "tags",
    "temporal_coverage",
    "created_at",
    "modified_at",
    "dataset_url",
    "download_url",
    "format",
    "columns",
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Tell-tale byte sequences left behind when UTF-8 bytes get decoded as
# latin-1/cp1252 upstream (e.g. Bologna's ODS field labels hand out "EtÃ
# singolo" instead of "Età singolo") — never legitimate in the languages
# these portals publish in, so safe to detect and repair in _clean_text.
_MOJIBAKE_SIGNATURE_RE = re.compile(
    r"Ã[\x82-\x9f\xa0-\xbf]|â€[\x98\x99\x9c\x9d\x93\x94\xa6\xb0]|Â[\xa0-\xbf]"
)


def normalize_metadata_records(records: list[dict], source: str) -> list[dict]:
    """Normalize a list of raw metadata records into a flat unified schema."""
    normalized_records: list[dict] = []
    source_normalized = _normalize_source(source)
    print(f"{source_normalized=}")
    for record in records:
        if source_normalized == "ckan":
            normalized_records.extend(_normalize_ckan_record(record))
        elif source_normalized == "ods":
            item = _normalize_ods_record(record)
            if item is not None:
                normalized_records.append(item)
            else:
                print("Item is None!")
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
    responsible_entity = _distinct_entity(
        _clean_text(record.get("datasource")) or _clean_text(record.get("author")),
        publisher,
    )
    # data.gov.uk keeps the period as top-level "temporal_coverage-from"/"-to"
    # keys; other CKAN portals put the same keys in extras.
    temporal_coverage = _format_temporal_coverage(
        record.get("hasBegining")
        or record.get("time_period_coverage_start")
        or record.get("temporal_coverage-from")
        or extras.get("temporal_start")
        or extras.get("temporal_coverage-from"),
        record.get("hasEnd")
        or record.get("time_period_coverage_end")
        or record.get("temporal_coverage-to")
        or extras.get("temporal_end")
        or extras.get("temporal_coverage-to"),
    )
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
        resource_name = _clean_text(resource.get("name"))
        resource_description = _clean_html(resource.get("description"))
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
                    # The dataset title and description are shared by all its
                    # resources; a resource's own name and description tell
                    # them apart ("2021-12-31 Organogram (Junior)"), so they
                    # are kept unless they only repeat the dataset's — or,
                    # for an ArcGIS-harvested file, name nothing but its
                    # format (see _identifying_resource_name).
                    "resource_name": _identifying_resource_name(resource_name, title),
                    "resource_description": (
                        resource_description
                        if resource_description != description
                        else None
                    ),
                    "source": "ckan",
                    "title": title,
                    "description": description,
                    "publisher": publisher,
                    "responsible_entity": responsible_entity,
                    "tags": tags,
                    "temporal_coverage": temporal_coverage,
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
    creator = _clean_text(dcat_meta.get("creator"))
    publisher = _clean_text(default_meta.get("publisher")) or creator

    return _post_process_record(
        {
            "dataset_id": dataset_id,
            "resource_id": dataset_id,
            "source": "ods",
            "title": _clean_text(default_meta.get("title")) or dataset_id,
            "description": _clean_html(default_meta.get("description")),
            "publisher": publisher,
            "responsible_entity": _distinct_entity(creator, publisher),
            "tags": _unique_strings(
                list(default_meta.get("keyword", []) or [])
                + list(default_meta.get("theme", []) or [])
            ),
            "temporal_coverage": _format_temporal_coverage(
                dcat_meta.get("temporal_coverage_start"),
                dcat_meta.get("temporal_coverage_end"),
                fallback=dcat_meta.get("temporal"),
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

    agency = _clean_text(domain_metadata.get("Dataset-Information_Agency"))
    publisher = _clean_text(resource.get("attribution")) or agency

    return _post_process_record(
        {
            "dataset_id": dataset_id,
            "resource_id": dataset_id,
            "source": "socrata",
            "title": _clean_text(resource.get("name")) or dataset_id,
            "description": _clean_html(resource.get("description")),
            "publisher": publisher,
            "responsible_entity": _distinct_entity(agency, publisher),
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
        "resource_name": _clean_text(record.get("resource_name")),
        "resource_description": _clean_html(record.get("resource_description")),
        "source": _clean_text(record.get("source")),
        "title": _clean_text(record.get("title")),
        "description": _clean_html(record.get("description")),
        "publisher": _clean_text(record.get("publisher")),
        "responsible_entity": _clean_text(record.get("responsible_entity")),
        "tags": _unique_strings(record.get("tags") or []),
        "temporal_coverage": _clean_text(record.get("temporal_coverage")),
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


_UNINFORMATIVE_VALUES = {"other", "n/a", "na", "none", "unknown", "unlimited", "-"}
_PARENTHETICAL_RE = re.compile(r"\(([^)]*)\)")
# Entity names at least this similar are spellings of one body: typos,
# "and" / "&", or a translation ("Ajuntament" / "Ayuntamiento de València").
_SAME_ENTITY_SIMILARITY = 0.8


def _distinct_entity(entity: str | None, publisher: str | None) -> str | None:
    """Return ``entity`` only when it names a body other than the publisher.

    Portals often carry a second responsible body next to the publisher (the
    producing department, an external data owner, the city agency behind an
    attribution). Variants of the publisher's own name — acronym forms,
    typos, "and" / "&", translations — and placeholders add nothing. A more
    specific unit of the publisher ("Comune di Bologna - Ufficio Statistica"
    under "Comune di Bologna") is kept.
    """
    if entity is None or entity.casefold() in _UNINFORMATIVE_VALUES:
        return None
    if publisher is None:
        return entity
    entity_key, publisher_key = _entity_key(entity), _entity_key(publisher)
    if not entity_key:
        return None
    if not publisher_key:
        return entity
    entity_acronyms, publisher_acronyms = _entity_acronyms(entity), _entity_acronyms(publisher)
    if (
        f" {entity_key} " in f" {publisher_key} "
        or entity_key in publisher_acronyms
        or publisher_key in entity_acronyms
        or entity_acronyms & publisher_acronyms
        or SequenceMatcher(None, entity_key, publisher_key).ratio() >= _SAME_ENTITY_SIMILARITY
    ):
        return None
    return entity


def _entity_key(name: str) -> str:
    """Words of an entity name, ignoring case, punctuation and parentheticals."""
    return " ".join(re.findall(r"\w+", _PARENTHETICAL_RE.sub(" ", name.casefold())))


def _entity_acronyms(name: str) -> set[str]:
    """Parenthesised short forms of an entity name, e.g. ``dof``."""
    return {" ".join(re.findall(r"\w+", m)) for m in _PARENTHETICAL_RE.findall(name.casefold())}


# An end date in or after this year marks an ongoing series, not a period.
_OPEN_ENDED_YEAR = 2099


def _format_temporal_coverage(start: Any, end: Any, fallback: Any = None) -> str | None:
    """Render the period a dataset covers as ``<start> to <end>``.

    Either bound may be missing; with neither, the portal's free-text period
    (``fallback``) is kept verbatim. A far-future end is how a portal marks an
    ongoing series (data.gov.uk uses ``2099-12-31``), so it counts as missing.
    """
    start_date, end_date = _coverage_date(start), _coverage_date(end)
    if end_date and end_date[:4].isdigit() and int(end_date[:4]) >= _OPEN_ENDED_YEAR:
        end_date = None
    if start_date and end_date:
        return f"{start_date} to {end_date}"
    if start_date:
        return f"from {start_date}"
    if end_date:
        return f"until {end_date}"
    fallback = _clean_text(fallback)
    if fallback is None or fallback.casefold() in _UNINFORMATIVE_VALUES:
        return None
    return fallback


def _coverage_date(value: Any) -> str | None:
    """Date of a coverage bound, as ``YYYY-MM-DD`` when it parses.

    Timestamps are rounded to the nearest day first: ODS stores local midnight
    in UTC, so Paris' ``2005-12-30T23:00:00+00:00`` means 2005-12-31. A list
    bound (data.gov.uk stores ``["2016-05-01"]``) is read as its first value.
    """
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if _clean_text(item)), None)
    text = _clean_text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    if "T" in text or " " in text:
        parsed += timedelta(hours=12)
    return parsed.date().isoformat()


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
    value = _repair_mojibake(value)
    value = value.strip()
    return value or None


def _repair_mojibake(text: str) -> str:
    """Undo a latin-1/cp1252 mis-decode of UTF-8 bytes, if one is detected.

    Only touches text carrying the tell-tale byte pattern, and only keeps
    the repair if the round-trip cleanly produces text free of both the
    original signature and the unicode replacement character — so clean
    text is never touched.
    """
    if not _MOJIBAKE_SIGNATURE_RE.search(text):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text
    if "�" in repaired or _MOJIBAKE_SIGNATURE_RE.search(repaired):
        return text
    return repaired


# A resource "name" made only of these says nothing about the table. On
# data.gov.uk an ArcGIS-harvested resource is named after its format, so 5,508
# of 27,390 UK resources are called "CSV" or "CSV Download" — a field the index
# weights as highly as the title (see orqa.benchmark.index.FIELD_WEIGHTS).
_FORMAT_LABELS = frozenset({
    "csv", "tsv", "xls", "xlsx", "ods", "json", "geojson", "xml", "zip", "pdf",
    "txt", "kml", "kmz", "api", "html", "htm", "rdf", "wfs", "wms", "parquet",
    "shp", "shapefile", "doc", "docx", "rss", "atom",
})
_FILLER_WORDS = frozenset({
    "download", "downloads", "file", "files", "data", "link", "links",
    "resource", "open", "format", "export", "view", "here", "click",
    "dataset", "table", "attachment", "preview",
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "as", "this",
})
_FORMAT_EXTENSION = re.compile(
    r"\.(?:" + "|".join(sorted(_FORMAT_LABELS)) + r")\s*$", re.IGNORECASE)
_FORMAT_TOKEN = re.compile(
    r"(?<![0-9a-z])(?:" + "|".join(sorted(_FORMAT_LABELS)) + r")(?![0-9a-z])",
    re.IGNORECASE)
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
# Brackets are excluded: stripping them would turn "Organogram (Junior)" into
# "Organogram (Junior". Pairs left empty by token removal are collapsed instead.
_EDGE_SEPARATORS = " -–—_,;:/|"


def _says_nothing_but_its_format(value: Any) -> bool:
    """Is this name only a file format and publishing filler?

    ``"CSV"`` and ``"CSV Download"`` are; ``"Organogram - Senior CSV data"``,
    which keeps a word of its own, is not.
    """
    words = [word for word in re.split(r"[^0-9a-zA-Z]+", str(value or "")) if word]
    return not [
        word for word in words
        if word.casefold() not in _FORMAT_LABELS
        and word.casefold() not in _FILLER_WORDS
    ]


def _strip_format_noise(value: Any) -> str | None:
    """Drop a trailing file extension and any standalone format token.

    Filler words survive: they decide whether a name is worthless, but removing
    them from a real name would change what it says ("Price Paid Data").
    """
    text = _FORMAT_EXTENSION.sub("", str(value or "").strip())
    text = _FORMAT_TOKEN.sub(" ", text)
    text = _EMPTY_BRACKETS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(_EDGE_SEPARATORS) or None


def _identifying_resource_name(resource_name: Any, title: Any) -> str | None:
    """The resource's own name, or None when it adds nothing to the title."""
    name = _clean_text(resource_name)
    if name is None or name == title or _says_nothing_but_its_format(name):
        return None
    cleaned = _strip_format_noise(name)
    return None if cleaned is None or cleaned == title else cleaned


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
