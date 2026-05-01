"""
ErrorFormatter
==============
Formats validation errors for the correction LLM prompt.

Responsibilities:
- Per-query error formatting with source labels (static vs judge)
- Token-limited output (300 tokens max per error message)
- Recurring error pattern detection across queries
- Error escalation with diagnostic context for repeated failures
- Building the full correction prompt with recurring mistakes header

Token approximation: words * 1.3 ≈ tokens
"""

from collections import Counter


class ErrorFormatter:
    """Formats validation errors for the correction LLM prompt."""

    MAX_ERROR_TOKENS = 300

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def truncate_to_tokens(self, message: str, max_tokens: int = 300) -> str:
        """Truncate message to approximate token limit.

        Uses word-count approximation: words * 1.3 ≈ tokens.
        So max_tokens=300 allows roughly floor(300 / 1.3) ≈ 230 words.

        Args:
            message: The text to truncate.
            max_tokens: Maximum token budget (default 300).

        Returns:
            The original message if within budget, or a truncated version
            ending with '...' if it exceeds the limit.
        """
        if not message:
            return message

        words = message.split()
        max_words = int(max_tokens / 1.3)

        if len(words) <= max_words:
            return message

        truncated_words = words[:max_words]
        return " ".join(truncated_words) + "..."

    def format_judge_feedback(
        self, query_id: int, error: str, suggestion: str = ""
    ) -> str:
        """Format a single judge feedback entry with clear labeling.

        Produces a string prefixed with ``[JUDGE FEEDBACK]`` that includes
        both the error reason and the suggestion text.  The output is
        truncated to MAX_ERROR_TOKENS.

        Args:
            query_id: The positional integer ID of the query.
            error: The judge's rejection reason.
            suggestion: The judge's suggestion for improvement.
                        Defaults to empty string if not provided.

        Returns:
            A formatted, token-limited string clearly labeled as judge feedback.
        """
        if suggestion is None:
            suggestion = ""

        parts = [f"[JUDGE FEEDBACK] Query #{query_id}:"]
        parts.append(f"  Error: {error}")
        parts.append(f"  Suggestion: {suggestion}")

        combined = "\n".join(parts)
        return self.truncate_to_tokens(combined, self.MAX_ERROR_TOKENS)

    def format_static_error(self, query_id: int, error: str) -> str:
        """Format a single static validation error with clear labeling.

        Produces a string prefixed with ``[STATIC VALIDATION ERROR]`` that
        includes the error message.  The output is truncated to
        MAX_ERROR_TOKENS and is guaranteed to never contain the
        ``[JUDGE FEEDBACK]`` label.

        Args:
            query_id: The positional integer ID of the query.
            error: The static validation error message.

        Returns:
            A formatted, token-limited string clearly labeled as a static
            validation error.
        """
        # Sanitize: ensure the output never contains the judge feedback label
        sanitized_error = error.replace("[JUDGE FEEDBACK]", "[JUDGE_FEEDBACK]")
        combined = f"[STATIC VALIDATION ERROR] Query #{query_id}:\n  {sanitized_error}"
        return self.truncate_to_tokens(combined, self.MAX_ERROR_TOKENS)

    def build_judge_feedback_block(
        self, queries: list[dict], judge_feedback: list[dict]
    ) -> str:
        """Build the complete feedback block for the judge-feedback-first cycle.

        Iterates over judge feedback entries, formats each with
        ``format_judge_feedback()``, and returns a combined block with a
        clear section header.

        Args:
            queries: List of pending query dicts (used for context/ID mapping).
            judge_feedback: List of feedback entries, each with:
                - ``"id"`` (int): The query ID this feedback applies to.
                - ``"error"`` (str): The judge's rejection reason.
                - ``"suggestion"`` (str, optional): Improvement suggestion.
                  Defaults to ``""`` if missing.

        Returns:
            A formatted string block with a section header and all feedback
            entries clearly labeled as ``[JUDGE FEEDBACK]``.
        """
        sections: list[str] = ["--- Judge Feedback ---"]

        for entry in judge_feedback:
            query_id = entry["id"]
            error = entry["error"]
            suggestion = entry.get("suggestion", "")
            formatted = self.format_judge_feedback(query_id, error, suggestion)
            sections.append(formatted)

        return "\n\n".join(sections)

    def format_per_query(
        self, query_id: int, errors: list[str], source_labels: list[str]
    ) -> str:
        """Format errors for a single query with source labels,
        truncating to MAX_ERROR_TOKENS.

        Args:
            query_id: The positional integer ID of the query.
            errors: List of error messages for this query.
            source_labels: Corresponding source label for each error
                           (e.g., "Static validation error" or "Judge feedback").

        Returns:
            A formatted string with labeled errors, truncated to token limit.
        """
        if not errors:
            return f"Query #{query_id}: No errors."

        parts: list[str] = []
        for error, label in zip(errors, source_labels):
            parts.append(f"[{label}] {error}")

        combined = f"Query #{query_id}:\n" + "\n".join(parts)
        return self.truncate_to_tokens(combined, self.MAX_ERROR_TOKENS)

    def detect_recurring(self, all_errors: list[str]) -> list[str]:
        """Identify error patterns appearing 2+ times across queries.

        Normalizes errors by stripping leading/trailing whitespace and
        counts occurrences. Patterns appearing 2 or more times are returned
        as rule strings.

        Args:
            all_errors: Flat list of all error messages across all queries.

        Returns:
            List of rule strings for recurring patterns. Empty if no pattern
            appears more than once.
        """
        if not all_errors:
            return []

        # Normalize errors for comparison
        normalized = [e.strip() for e in all_errors if e.strip()]
        counts = Counter(normalized)

        recurring = []
        for pattern, count in counts.items():
            if count >= 2:
                recurring.append(
                    f"Recurring issue ({count} occurrences): {pattern}"
                )

        return recurring

    def build_correction_prompt(
        self, queries_with_errors: list, recurring: list[str]
    ) -> str:
        """Build the full correction prompt with recurring mistakes header.

        Args:
            queries_with_errors: List of dicts, each with:
                - "query_id" (int): The query's positional ID
                - "errors" (list[str]): Error messages for this query
                - "source_labels" (list[str]): Source label per error
            recurring: List of recurring pattern rule strings (from detect_recurring).

        Returns:
            The complete correction prompt text with optional recurring
            mistakes header followed by per-query error sections.
        """
        sections: list[str] = []

        # Recurring mistakes header
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

        # Per-query error sections
        sections.append("=== PER-QUERY ERRORS ===")
        for entry in queries_with_errors:
            query_id = entry["query_id"]
            errors = entry["errors"]
            source_labels = entry["source_labels"]
            formatted = self.format_per_query(query_id, errors, source_labels)
            sections.append(formatted)

        return "\n\n".join(sections)

    def escalate_error(self, error: str, context: dict) -> str:
        """Add diagnostic context for errors repeating across attempts.

        When the same error appears in consecutive correction attempts,
        this method augments the error message with additional diagnostic
        information to help the LLM produce a different fix.

        Args:
            error: The original error message.
            context: Diagnostic context dict, may contain:
                - "columns" (list[str]): Available column names
                - "shape" (tuple): DataFrame shape (rows, cols)
                - "dtypes" (dict): Column name → dtype mapping

        Returns:
            The error message augmented with diagnostic context.
        """
        parts = [error, "", "--- ESCALATED: Additional diagnostic context ---"]

        if "columns" in context:
            cols = context["columns"]
            parts.append(f"Available columns: {cols}")

        if "shape" in context:
            shape = context["shape"]
            parts.append(f"DataFrame shape: {shape[0]} rows × {shape[1]} columns")

        if "dtypes" in context:
            dtypes = context["dtypes"]
            dtype_str = ", ".join(f"{col}: {dt}" for col, dt in dtypes.items())
            parts.append(f"Column dtypes: {dtype_str}")

        parts.append(
            "This error has persisted across attempts. "
            "Please try a fundamentally different approach."
        )

        return "\n".join(parts)
