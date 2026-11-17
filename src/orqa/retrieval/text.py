"""Text helpers shared by the question stage, the keyword-anchor search and the
retrievability gate: how a string becomes words, and which text of a
normalized-metadata record a portal search can match.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase, strip accents/diacritics and split on any
    non-alphanumeric character.

    Accent folding keeps comparisons usable across the languages of the
    crawled portals (English, French, Italian, Spanish, Catalan), e.g.
    both "credito" and "crédito" map to the same token.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _TOKEN_RE.findall(text)


def record_field_texts(record: dict) -> dict[str, str]:
    """The searchable text of a normalized-metadata record, per field:
    ``title``, ``resource_name``, ``tags``, ``columns`` (names, labels and
    descriptions), ``publisher`` and ``description`` (the file's own
    description, then the dataset's)."""
    columns_parts = []
    for col in record.get("columns") or []:
        for key in ("name", "label", "description"):
            value = col.get(key)
            if value:
                columns_parts.append(str(value))

    return {
        "title": record.get("title") or "",
        "resource_name": record.get("resource_name") or "",
        "tags": " ".join(record.get("tags") or []),
        "columns": " ".join(columns_parts),
        "publisher": record.get("publisher") or "",
        "description": " ".join(
            part
            for part in (record.get("resource_description"), record.get("description"))
            if part
        ),
    }
