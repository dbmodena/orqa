"""Multi-table statement-generation agent."""
from ..agent import StatementOrchestrator, MULTI


class StatementAgent(StatementOrchestrator):
    """Multi-table entry point for the statement-generation orchestrator.

    Its ``generate_statements`` keeps the multi-table positional signature and
    delegates to the orchestrator's multi-table adapter (``_run(mode="multi", ...)``).
    """

    def generate_statements(
        self,
        dataset_paths,
        aliases,
        kind,
        match,
        involved_cols,
        metadatas,
        max_cols: int = 10,
        sample_size: int = 5,
    ):
        return self._run(
            mode=MULTI,
            dataset_paths=dataset_paths,
            aliases=aliases,
            kind=kind,
            match=match,
            involved_cols=involved_cols,
            metadata=metadatas,
            max_cols=max_cols,
            sample_size=sample_size,
        )
