"""
ErrorFormatter
==============
Formats validation errors for the correction LLM prompt.

Each query is rendered as a JSON object matching the Query Pydantic schema,
followed by its errors/feedback — giving the LLM the exact structure it must
return alongside the context it needs to fix it. The JSON block includes the
question-level metadata bundle (question/question_keywords/translated_question/
translated_question_keywords/topic/story) that planning produces as one unit,
so a correction that rewrites the question can regenerate all of it
consistently rather than leaving the rest stale.

Every correction call is for exactly ONE query (see
``StatementValidator._correct_queries_concurrently``) — there is no
cross-query batching, so there's nothing to compare across queries within a
single call.
"""

import json


class ErrorFormatter:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_per_query(
        self,
        query_id: int,
        errors: list[str],
        source_labels: list[str],
        question: str = "",
        translated_question: str = "",
        detected_language: str = "",
        question_keywords: list[str] | None = None,
        translated_question_keywords: list[str] | None = None,
        topic: str = "",
        story: str = "",
        code: str = "",
        tables: list = None,
    ) -> str:
        """Format a single query as a JSON block (matching the Query Pydantic schema)
        followed by its errors/feedback.

        The JSON block gives the LLM the exact structure it must reproduce in its
        response, with every field populated so it has full context to correct only
        what is broken.

        ``question``, ``question_keywords``, ``translated_question``,
        ``translated_question_keywords``, ``topic``, and ``story`` are shown as
        ONE linked bundle (they were all produced together during planning —
        see ``prompting.models.SQLQueryPlan``/``PandasQueryPlan``): the
        correction prompt instructs the model to regenerate all of them
        together if it rewrites ``question``, and to return them unchanged
        otherwise.
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
            "question_keywords": question_keywords or [],
            "translated_question": translated_question,
            "translated_question_keywords": translated_question_keywords or [],
            "detected_language": detected_language,
            "topic": topic,
            "story": story,
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

    def build_correction_prompt(self, queries_with_errors: list) -> str:
        """Build the correction prompt for a (single-entry) list of queries-with-errors."""
        sections: list[str] = ["=== PER-QUERY ERRORS ==="]
        for entry in queries_with_errors:
            formatted = self.format_per_query(
                query_id=entry["query_id"],
                errors=entry["errors"],
                source_labels=entry["source_labels"],
                question=entry.get("question", ""),
                translated_question=entry.get("translated_question", ""),
                detected_language=entry.get("detected_language", ""),
                question_keywords=entry.get("question_keywords"),
                translated_question_keywords=entry.get("translated_question_keywords"),
                topic=entry.get("topic", ""),
                story=entry.get("story", ""),
                code=entry.get("code", ""),
                tables=entry.get("tables"),
            )
            sections.append(formatted)

        return "\n\n".join(sections)