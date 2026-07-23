"""Phase-specific collaborators for the statement-generation pipeline.

- ``TaskProposer`` — proposes candidate discovery tasks for a dataset.
- ``table_analyzer`` — batched per-table analysis (one LLM call for all tables).
- ``QueryPlanner`` — structured, ordered query plans grounded in table analysis.
- ``StatementClient`` — generates the initial query set from a prompt.
- ``StatementValidator`` — static validation + LLM correction of generated queries.
- ``StatementJudge`` — evaluates executed queries and produces the business-facing response.
- ``StatementAgent`` / ``SingleStatementAgent`` — the multi-table / single-table
  entry points into the pipeline; each subclasses the orchestrator and pins
  ``generate_statements`` to the positional signature its caller expects.

The top-level pipeline orchestrator (``StatementOrchestrator``) lives one
level up, at ``orqa.agent.agent`` — it owns and coordinates these
collaborators, rather than being one itself.
"""
