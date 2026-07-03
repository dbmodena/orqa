"""Utility submodule for the statement-generation agent.

Groups the supporting components of the agent package:

- ``alias_substitution`` — canonical alias substitution.
- ``budget_guard`` — overall wall-clock/token budget ceiling.
- ``code_generator`` — skill-aware code generator + client_id contract.
- ``error_formatter`` — validation-error prompt formatting.
- ``message_builder`` — fixed-structure LLM message builders.
- ``query_planner`` — structured query planner.
- ``table_analyzer`` — batched table analysis.
- ``unified_agent`` — unified mode-aware statement-generation agent.
"""
