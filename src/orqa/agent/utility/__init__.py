"""Utility submodule for the statement-generation agent.

Groups the supporting components of the agent package:

- ``alias_substitution`` — canonical alias substitution.
- ``generation_coordinator`` — generation orchestration + client_id contract.
- ``error_formatter`` — validation-error prompt formatting.
- ``message_builder`` — fixed-structure LLM message builders.
- ``structured_outputs`` — shared Pydantic response-model schemas.

``PipelineLogger`` (pretty console logger for the pipeline, plus discovery-
stage logging) now lives at ``orqa.utils.pipeline_logger`` — it's shared
outside the statement-generation agent (embedding/clustering/discovery
stages use it too), so it no longer belongs under this agent-scoped package.

The pipeline orchestrator itself (``StatementOrchestrator``) lives one level
up, at ``orqa.agent.agent`` — it is the agent, not a supporting utility, so
it is not grouped here. The phase-specific collaborators (``TaskProposer``,
``table_analyzer``, ``QueryPlanner``, ``StatementClient``,
``StatementValidator``, ``StatementJudge``, ``StatementAgent``,
``SingleStatementAgent``) live in the sibling ``orqa.agent.agents`` package.
"""
