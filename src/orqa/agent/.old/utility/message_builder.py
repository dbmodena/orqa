"""Retired message builder(s) from the statement pipeline.

Moved out of ``orqa/agent/utility/message_builder.py``: nothing in the live
pipeline instantiates ``ValidatorMessageBuilder`` (``StatementValidator.py``
imported it but never called it). Kept here for reference rather than
deleted outright.
"""


class ValidatorMessageBuilder:
    """Builds the fixed 2-message array for correction LLM calls.

    Each correction call sends exactly:
      1. System message (role: system) — minimal generic instruction
      2. User message (role: user) — full correction prompt with all context

    Each call is stateless: no previous responses are threaded between cycles.
    """

    def build(
        self,
        system_content: str,
        user_content: str,
    ) -> list[dict]:
        """Returns exactly 2 message blocks: [system, user].

        Args:
            system_content: The minimal system instruction.
            user_content: The full rendered correction prompt containing
                fix rules, schemas, queries+errors, and pydantic constraint.

        Returns:
            A list of exactly 2 message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
