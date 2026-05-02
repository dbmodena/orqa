"""Fixed-structure message builders for the statement pipeline agents.

# Feature: statement-pipeline-overhaul, Task 2: Fixed-Structure Message Builders

Each agent has a defined message template that is rebuilt (not appended to)
on each cycle, preventing context window overflow and ensuring clean, focused
prompts.

- ValidatorMessageBuilder: 2 or 3 blocks (system + [assistant] + user)
- ClientMessageBuilder: 2 blocks (system + generation prompt)
- JudgeMessageBuilder: 2 blocks (system + single query evaluation)
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
