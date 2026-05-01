"""Fixed-structure message builders for the statement pipeline agents.

# Feature: statement-pipeline-overhaul, Task 2: Fixed-Structure Message Builders

Each agent has a defined message template that is rebuilt (not appended to)
on each cycle, preventing context window overflow and ensuring clean, focused
prompts.

- ValidatorMessageBuilder: 4 blocks (system + schema context + queries + errors/feedback)
- ClientMessageBuilder: 2 blocks (system + generation prompt)
- JudgeMessageBuilder: 2 blocks (system + single query evaluation)
"""


class ValidatorMessageBuilder:
    """Builds the fixed 4-block message array for correction LLM calls.

    The validator always sends exactly 4 messages:
      1. System prompt (role: system)
      2. Schema context (role: user) — table schemas and alias info
      3. Generated queries (role: user) — the queries from the previous cycle
      4. Validation errors/feedback (role: user) — errors and/or judge feedback
    """

    def build(
        self,
        system_prompt: str,
        table_schemas: str,
        queries_text: str,
        errors_text: str,
    ) -> list[dict]:
        """Returns exactly 4 message blocks:
        [system, schema_context, queries, errors_feedback]

        Args:
            system_prompt: The system-level instruction for the correction LLM.
            table_schemas: Pre-formatted schema string with table definitions.
            queries_text: Formatted text of the generated queries to correct.
            errors_text: Formatted validation errors and/or judge feedback.

        Returns:
            A list of exactly 4 message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": table_schemas},
            {"role": "user", "content": queries_text},
            {"role": "user", "content": errors_text},
        ]


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
