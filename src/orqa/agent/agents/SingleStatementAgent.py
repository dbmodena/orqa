"""Single-table statement-generation agent."""
from ..agent import StatementOrchestrator


class SingleStatementAgent(StatementOrchestrator):
    """Single-table entry point for the statement-generation orchestrator.

    IMPORTANT: the single-table call site invokes
    ``generate_statements(csv_path, aliases, kind, metadata, max_cols, sample_size)``,
    so this class's ``generate_statements`` MUST accept the SINGLE-table
    positional signature ``(dataset_path, alias, kind, metadata, ...)`` — NOT the
    multi-table one — and forward to the orchestrator's single-table adapter
    (``generate_statements_single`` -> ``_run(mode="single", ...)``).
    """

    def generate_statements(
        self,
        dataset_path,
        alias,
        kind,
        metadata,
        max_cols: int = 10,
        sample_size: int = 5,
    ):
        return self.generate_statements_single(
            dataset_path,
            alias,
            kind,
            metadata,
            max_cols=max_cols,
            sample_size=sample_size,
        )
