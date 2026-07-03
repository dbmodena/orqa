"""Centralized alias substitution logic.

Extracted from the duplicated `clean_table_names` static methods in
StatementClient and StatementValidator. Provides word-boundary-aware regex
substitution of original dataset IDs (e.g. "qvir-knu3") with canonical
aliases (e.g. "Table_0").
"""

from __future__ import annotations

import re


class AliasSubstitution:
    """Centralized alias substitution logic.

    Replaces original dataset identifiers with canonical aliases using
    pre-compiled word-boundary-aware regex patterns that handle both
    dash-separated and underscore-separated variants.
    """

    def __init__(self, aliases: dict[str, str]) -> None:
        """
        Args:
            aliases: Mapping of canonical alias → original dataset ID
                     e.g. {"Table_0": "qvir-knu3", "Table_1": "abc-def"}
        """
        self._aliases = aliases
        # Invert: original_name → alias
        self._inverted: dict[str, str] = {v: k for k, v in aliases.items()}

        # Pre-compile word-boundary regex patterns for each original name
        # and its underscore variant.
        # Pattern: not preceded by a dot or word char, not followed by a word char
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for original_name, alias in self._inverted.items():
            underscore_variant = original_name.replace("-", "_")
            for name in {original_name, underscore_variant}:
                pattern = re.compile(
                    r"(?<![.\w])" + re.escape(name) + r"(?!\w)"
                )
                self._patterns.append((pattern, alias))

        # Pre-compile verification patterns (same logic, just for detection)
        self._verify_patterns: list[tuple[re.Pattern[str], str]] = []
        for original_name in self._inverted:
            underscore_variant = original_name.replace("-", "_")
            for name in {original_name, underscore_variant}:
                pattern = re.compile(
                    r"(?<![.\w])" + re.escape(name) + r"(?!\w)"
                )
                self._verify_patterns.append((pattern, name))

    def substitute(self, text: str) -> str:
        """Replace all original table names (dash and underscore variants)
        with canonical aliases using word-boundary-aware regex.

        Args:
            text: Input text potentially containing original table names.

        Returns:
            Text with all original names replaced by their canonical aliases.
        """
        result = text
        for pattern, alias in self._patterns:
            result = pattern.sub(alias, result)
        return result

    def substitute_query(self, query: dict) -> dict:
        """Apply substitution to code, question, motivation, and tables[].name fields.

        Args:
            query: A query dict with fields like code, question, motivation, tables.

        Returns:
            A new query dict with substitutions applied to relevant fields.
        """
        result = dict(query)

        # Substitute text fields
        for field in ("code", "question", "motivation"):
            if field in result and isinstance(result[field], str):
                result[field] = self.substitute(result[field])

        # Substitute table names in the tables list
        if "tables" in result and isinstance(result["tables"], list):
            new_tables = []
            for table in result["tables"]:
                if isinstance(table, dict):
                    new_table = dict(table)
                    if "name" in new_table and isinstance(new_table["name"], str):
                        new_table["name"] = self.substitute(new_table["name"])
                    new_tables.append(new_table)
                else:
                    new_tables.append(table)
            result["tables"] = new_tables

        return result

    def verify(self, text: str) -> list[str]:
        """Return list of original names still present at word boundaries.

        Args:
            text: Text to check for remaining original table names.

        Returns:
            List of original names (or their underscore variants) found in text.
            Should be empty after successful substitution.
        """
        found: list[str] = []
        for pattern, name in self._verify_patterns:
            if pattern.search(text):
                found.append(name)
        return found
