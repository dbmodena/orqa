"""
ErrorFormatter
==============
Formats validation errors for the correction LLM prompt.

Each query is rendered as a JSON object matching the Query Pydantic schema,
followed by its errors/feedback — giving the LLM the exact structure it must
return alongside the context it needs to fix it.
"""

import json
from collections import Counter


class ErrorFormatter:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_judge_feedback(
        self, query_id: int, error: str, suggestion: str = ""
    ) -> str:
        if suggestion is None:
            suggestion = ""
        parts = [f"[JUDGE FEEDBACK] Query #{query_id}:"]
        parts.append(f"  Error: {error}")
        parts.append(f"  Suggestion: {suggestion}")
        return "\n".join(parts)

    def format_static_error(self, query_id: int, error: str) -> str:
        sanitized_error = error.replace("[JUDGE FEEDBACK]", "[JUDGE_FEEDBACK]")
        return f"[STATIC VALIDATION ERROR] Query #{query_id}:\n  {sanitized_error}"

    def build_judge_feedback_block(
        self, queries: list[dict], judge_feedback: list[dict]
    ) -> str:
        """Build a correction prompt block for judge-feedback entries."""
        queries_by_id: dict = {str(q.get("id")): q for q in queries}

        entries: list[dict] = []
        for fb in judge_feedback:
            qid = str(fb.get("id"))
            q = queries_by_id.get(qid, {})

            error = fb.get("error", "")
            suggestion = fb.get("suggestion", "")
            error_text = error
            if suggestion:
                error_text = f"{error}\n  Suggestion: {suggestion}"

            entries.append({
                "query_id": qid,
                "question": q.get("question", ""),
                "translated_question": q.get("translated_question", ""),
                "detected_language": q.get("detected_language", ""),
                "code": q.get("code", ""),
                "tables": q.get("tables"),
                "errors": [error_text],
                "source_labels": ["Judge Feedback"],
            })

        return self.build_correction_prompt(entries, [])

    def format_per_query(
        self,
        query_id: int,
        errors: list[str],
        source_labels: list[str],
        question: str = "",
        translated_question: str = "",
        detected_language: str = "",
        code: str = "",
        tables: list = None,
    ) -> str:
        """Format a single query as a JSON block (matching the Query Pydantic schema)
        followed by its errors/feedback.

        The JSON block gives the LLM the exact structure it must reproduce in its
        response, with every field populated so it has full context to correct only
        what is broken.
        """
        # --- Build the query JSON object (mirrors Query Pydantic schema) ---
        # Normalise tables into the Table schema shape regardless of how they
        # arrived (dict with name/reason/columns_involved, or plain string).
        normalised_tables: list[dict] = []
        for t in (tables or []):
            if isinstance(t, dict):
                normalised_tables.append({
                    "name": t.get("name", ""),
                    "reason": t.get("reason", ""),
                    "columns_involved": t.get("columns_involved", []),
                })
            else:
                normalised_tables.append({
                    "name": str(t),
                    "reason": "",
                    "columns_involved": [],
                })

        query_obj = {
            "question": question,
            "translated_question": translated_question,
            "detected_language": detected_language,
            "code": code,
            "tables": normalised_tables,
        }
        query_json = json.dumps(query_obj, ensure_ascii=False, indent=2)

        # --- Build the error/feedback block --------------------------------
        error_lines: list[str] = []
        for error, label in zip(errors, source_labels):
            error_lines.append(f"[{label}] {error}")
        if not error_lines:
            error_lines.append("[No errors]")

        return (
            f"--- Query #{query_id} ---\n"
            f"{query_json}\n\n"
            + "\n".join(error_lines)
        )

    def detect_recurring(self, all_errors: list[str]) -> list[str]:
        """Identify error patterns appearing 2+ times across queries."""
        if not all_errors:
            return []
        normalized = [e.strip() for e in all_errors if e.strip()]
        counts = Counter(normalized)
        return [
            f"Recurring issue ({count} occurrences): {pattern}"
            for pattern, count in counts.items()
            if count >= 2
        ]

    def build_correction_prompt(
        self, queries_with_errors: list, recurring: list[str]
    ) -> str:
        """Build the full correction prompt with an optional recurring-mistakes header."""
        sections: list[str] = []

        if recurring:
            header_lines = ["=== RECURRING MISTAKES ==="]
            header_lines.append(
                "The following issues appear across multiple queries. "
                "Apply these fixes universally:"
            )
            for rule in recurring:
                header_lines.append(f"  • {rule}")
            header_lines.append("")
            sections.append("\n".join(header_lines))

        sections.append("=== PER-QUERY ERRORS ===")
        for entry in queries_with_errors:
            formatted = self.format_per_query(
                query_id=entry["query_id"],
                errors=entry["errors"],
                source_labels=entry["source_labels"],
                question=entry.get("question", ""),
                translated_question=entry.get("translated_question", ""),
                detected_language=entry.get("detected_language", ""),
                code=entry.get("code", ""),
                tables=entry.get("tables"),
            )
            sections.append(formatted)

        return "\n\n".join(sections)

    def escalate_error(self, error: str, context: dict) -> str:
        """Add diagnostic context for errors repeating across attempts."""
        parts = [error, "", "--- ESCALATED: Additional diagnostic context ---"]

        if "columns" in context:
            parts.append(f"Available columns: {context['columns']}")
        if "shape" in context:
            shape = context["shape"]
            parts.append(f"DataFrame shape: {shape[0]} rows × {shape[1]} columns")
        if "dtypes" in context:
            dtype_str = ", ".join(
                f"{col}: {dt}" for col, dt in context["dtypes"].items()
            )
            parts.append(f"Column dtypes: {dtype_str}")

        parts.append(
            "This error has persisted across attempts. "
            "Please try a fundamentally different approach."
        )
        return "\n".join(parts)