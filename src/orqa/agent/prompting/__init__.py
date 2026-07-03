"""Prompting package for the OrQA agent.

This package replaces the former single-module ``prompting.py``. All prompt
classes and helpers now live in :mod:`orqa.agent.prompting.prompts` and are
re-exported here so existing imports such as ``from .prompting import ...`` and
``from orqa.agent.prompting import ...`` continue to resolve unchanged.
"""

from .prompts import (
    PROMPT_PATH,
    Prompt,
    DatasetDescription,
    LightDatasetDescription,
    CandidatesDiscoveryPrompt,
    PandasStatementGenerationPrompt,
    SQLStatementGenerationPrompt,
    SingleTablePandasPrompt,
    SingleTableSQLPrompt,
    JudgementResponseGenerationPrompt,
    SingleTableJudgementResponseGenerationPrompt,
    ResponseGenerationPrompt,
    PandasValidatorCorrectionPrompt,
    SQLValidatorCorrectionPrompt,
    _load_prompt,
    _extract_section,
)
from .models import (
    PlanStep,
    StructuredQueryPlan,
    ColumnStat,
    TableStats,
)
from .statistics import ColumnStatistics
from .skill_registry import (
    SKILLS_DIR,
    SkillCard,
    SkillGateContext,
    SkillSelection,
    SkillRegistry,
)

__all__ = [
    "PROMPT_PATH",
    "Prompt",
    "DatasetDescription",
    "LightDatasetDescription",
    "CandidatesDiscoveryPrompt",
    "PandasStatementGenerationPrompt",
    "SQLStatementGenerationPrompt",
    "SingleTablePandasPrompt",
    "SingleTableSQLPrompt",
    "JudgementResponseGenerationPrompt",
    "SingleTableJudgementResponseGenerationPrompt",
    "ResponseGenerationPrompt",
    "PandasValidatorCorrectionPrompt",
    "SQLValidatorCorrectionPrompt",
    "_load_prompt",
    "_extract_section",
    "PlanStep",
    "StructuredQueryPlan",
    "ColumnStat",
    "TableStats",
    "ColumnStatistics",
    "SKILLS_DIR",
    "SkillCard",
    "SkillGateContext",
    "SkillSelection",
    "SkillRegistry",
]
