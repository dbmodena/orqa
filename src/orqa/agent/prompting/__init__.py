"""Prompting package for the OrQA agent.

This package replaces the former single-module ``prompting.py``. All prompt
classes and helpers now live in :mod:`orqa.agent.prompting.prompts` and are
re-exported here so existing imports such as ``from .prompting import ...`` and
``from orqa.agent.prompting import ...`` continue to resolve unchanged.
"""

from .prompts import (
    Prompt,
    DatasetDescription,
    LightDatasetDescription,
    CandidatesDiscoveryPrompt,
    PairTaskSelectionPrompt,
    PandasStatementGenerationPrompt,
    SQLStatementGenerationPrompt,
    SingleTablePandasPrompt,
    SingleTableSQLPrompt,
    JudgementResponseGenerationPrompt,
    SingleTableJudgementResponseGenerationPrompt,
    PlanJudgementPrompt,
    ResponseGenerationPrompt,
    PandasValidatorCorrectionPrompt,
    SQLValidatorCorrectionPrompt,
    GenerationEnrichmentPrompt,
    QueryPlannerPrompt,
    TableAnalyzerBatchPrompt,
    SkillUsageInstructionPrompt,
    load_skill_section,
    build_skill_sections,
    build_generation_prompt,
)
from .models import (
    SQLPlanStep,
    PandasPlanStep,
    SQLQueryPlan,
    PandasQueryPlan,
    SQLQueryPlanSet,
    PandasQueryPlanSet,
    ColumnStat,
    TableStats,
)
from .statistics import ColumnStatistics

__all__ = [
    "Prompt",
    "DatasetDescription",
    "LightDatasetDescription",
    "CandidatesDiscoveryPrompt",
    "PairTaskSelectionPrompt",
    "PandasStatementGenerationPrompt",
    "SQLStatementGenerationPrompt",
    "SingleTablePandasPrompt",
    "SingleTableSQLPrompt",
    "JudgementResponseGenerationPrompt",
    "SingleTableJudgementResponseGenerationPrompt",
    "PlanJudgementPrompt",
    "ResponseGenerationPrompt",
    "PandasValidatorCorrectionPrompt",
    "SQLValidatorCorrectionPrompt",
    "GenerationEnrichmentPrompt",
    "QueryPlannerPrompt",
    "TableAnalyzerBatchPrompt",
    "SkillUsageInstructionPrompt",
    "load_skill_section",
    "build_skill_sections",
    "build_generation_prompt",
    "SQLPlanStep",
    "PandasPlanStep",
    "SQLQueryPlan",
    "PandasQueryPlan",
    "SQLQueryPlanSet",
    "PandasQueryPlanSet",
    "ColumnStat",
    "TableStats",
    "ColumnStatistics",
]
