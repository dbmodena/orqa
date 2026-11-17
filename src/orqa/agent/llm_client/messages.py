"""Fixed-structure message builders for the statement pipeline agents.

# Feature: statement-pipeline-overhaul, Task 2: Fixed-Structure Message Builders

Each agent has a defined message template that is rebuilt (not appended to)
on each cycle, preventing context window overflow and ensuring clean, focused
prompts.

- ClientMessageBuilder: 2 blocks (system + generation prompt)
- JudgeMessageBuilder: 2 blocks (system + single query evaluation)
- sanitize_messages: last-line guard applied at every router call site.
"""

# Substituted for any empty/whitespace-only message content. Some providers
# hard-reject zero-token messages (e.g. OCI 400: "message must be at least 1
# token long or tool results must be specified").
_EMPTY_CONTENT_PLACEHOLDER = "(empty message)"


def sanitize_messages(messages: list[dict], placeholder: str = _EMPTY_CONTENT_PLACEHOLDER) -> list[dict]:
    """Return a copy of ``messages`` in which no message has empty content.

    A few code paths can legitimately produce an empty message — an empty
    first-attempt payload, or a model's own empty response echoed back as an
    ``assistant`` message on a JSON-repair retry — and some providers reject
    the whole request over it. Non-string content is coerced to ``str`` and
    empty/whitespace-only content is replaced with a short placeholder, so the
    request is always transport-valid regardless of which path built it.
    """
    sanitized = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if not content.strip():
            content = placeholder
        sanitized.append({**m, "content": content})
    return sanitized


class ClientMessageBuilder:
    """Builds the fixed 2-block message array for Statement Client.

    The client always sends exactly 2 messages:
      1. System prompt (role: system)
      2. Generation prompt (role: user) — the full generation instruction
    """

    def build(self, system_prompt: str, generation_prompt: str) -> list[dict]:
        """Returns exactly 2 message blocks: [system, generation]

        Args:
            system_prompt: The system-level instruction for the generation LLM.
            generation_prompt: The user-facing generation prompt with all context.

        Returns:
            A list of exactly 2 message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generation_prompt},
        ]


class JudgeMessageBuilder:
    """Builds the fixed 2-block message array for Statement Judge.

    The judge always sends exactly 2 messages:
      1. System prompt (role: system)
      2. Single query evaluation (role: user) — one query payload to judge
    """

    def build(self, system_prompt: str, query_payload: str) -> list[dict]:
        """Returns exactly 2 message blocks: [system, query_evaluation]

        Args:
            system_prompt: The system-level instruction for the judge LLM.
            query_payload: The formatted single query to evaluate.

        Returns:
            A list of exactly 2 message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_payload},
        ]
