"""CKAN dataset families.

A *family* is a CKAN ``dataset_id``: the group of files (``resource_id``s)
published together under one dataset entry. 83% of UK gold tables belong to
a family with other files (median 38, max 133 — "Organogram of Staff Roles &
Salaries" alone has 1,426), which is why sampling is stratified by family
(``orqa.statement_generation``, ``max_groups_per_family``). :class:`FamilyIndex`
maps resource ids to families and back to the on-disk file stem.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from ..utils import dataset_id_to_resource_id


class FamilyIndex:
    """``dataset_id -> [resource_id]`` (and back), built from normalized
    metadata records plus a scan of the datasets folder for the on-disk
    file stem each resource id maps to.

    A record with no ``dataset_id`` is its own singleton family (keyed by
    its own resource id) — CKAN records always carry one, but this keeps the
    index total over any record shape a caller hands it (e.g. Socrata/ODS
    metadata, which has no dataset grouping at all).
    """

    def __init__(self, records: Iterable[dict], datasets_path: Path):
        self.datasets_path = Path(datasets_path)

        self._family_of_resource: dict[str, str] = {}
        self._members: dict[str, list[str]] = defaultdict(list)
        for record in records:
            resource_id = record.get("resource_id") or record.get("dataset_id")
            if not resource_id:
                continue
            family_id = record.get("dataset_id") or resource_id
            self._family_of_resource[resource_id] = family_id
            self._members[family_id].append(resource_id)

        self._stem_of_resource: dict[str, str] = {}
        self._resource_of_stem: dict[str, str] = {}
        if self.datasets_path.exists():
            for filepath in sorted(self.datasets_path.iterdir()):
                if not filepath.is_file():
                    continue
                stem = filepath.stem
                resource_id = dataset_id_to_resource_id(stem)
                self._stem_of_resource[resource_id] = stem
                self._resource_of_stem[stem] = resource_id

    def family_id(self, resource_id: str) -> str:
        return self._family_of_resource.get(resource_id, resource_id)

    def family_id_of_stem(self, stem: str) -> str:
        resource_id = self._resource_of_stem.get(stem) or dataset_id_to_resource_id(stem)
        return self.family_id(resource_id)

    def members(self, family_id: str) -> list[str]:
        """Every resource id of this family, including the query itself."""
        return list(self._members.get(family_id, [family_id]))

    def siblings(self, resource_id: str) -> list[str]:
        """The OTHER resource ids sharing ``resource_id``'s family."""
        family = self.family_id(resource_id)
        return [r for r in self._members.get(family, []) if r != resource_id]

    def family_size(self, resource_id: str) -> int:
        family = self.family_id(resource_id)
        return len(self._members.get(family, [resource_id]))

    def stem(self, resource_id: str) -> Optional[str]:
        """The on-disk file stem for this resource id, when a local file exists."""
        return self._stem_of_resource.get(resource_id)
