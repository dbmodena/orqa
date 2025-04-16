from dataclasses import dataclass

from autogen_core import TopicId
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import FunctionExecutionResult


def get_model_client(model, family: str = "unknown") -> OpenAIChatCompletionClient:  # type: ignore
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model=model,
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://localhost:4000/",
        model_capabilities={
            "json_output": True,
            "vision": False,
            "function_calling": True,
            "family": family
        },
    )    




GENERATION_TOPIC_TYPE   = "generation-result"
RESULT_TOPIC_TYPE       = "final-result"
generation_topic_id     = TopicId(type=GENERATION_TOPIC_TYPE    , source="default")
result_topic_id         = TopicId(type=RESULT_TOPIC_TYPE        , source="default")


@dataclass
class GenerationTask:
    task: str

@dataclass
class GenerationResult:
    sql: str
    nl: str
    n_rev: int

@dataclass
class ReviewTask:
    task: str
    sql: str
    nl: str
    execution_result: str

@dataclass
class ReviewResult:
    review: str
    json_review: dict
    approved: bool


# For SQL Query Generation

SQL_GENERATION_TOPIC_TYPE   = "sql-generation-result"
SQL_INTERMEDIATE_TOPIC_TYPE = "sql-intermediate-result"
SQL_RESULT_TOPIC_TYPE       = "sql-final-result"
sql_generation_topic_id     = TopicId(type=SQL_GENERATION_TOPIC_TYPE    , source="default")
sql_intermediate_topic_id   = TopicId(type=SQL_INTERMEDIATE_TOPIC_TYPE  , source="default")
sql_result_topic_id         = TopicId(type=SQL_RESULT_TOPIC_TYPE        , source="default")


@dataclass
class SQLGenerationTask:
    sql_task                : str

@dataclass
class SQLGenerationResult:
    sql_task                : str
    sql_query               : str
    review                  : str
    sql_success             : bool
    n_rev                   : int
    input_tokens            : int
    output_tokens           : int

@dataclass
class SQLReviewTask:
    sql_task                : str
    sql_query               : str
    execution_result        : str

@dataclass
class SQLReviewResult:
    review                  : str
    json_review             : dict
    approved                : bool
    execution_result        : str
    input_tokens            : int
    output_tokens           : int



# For Natural Language Generation

NL_GENERATION_TOPIC_TYPE    = "nl-generation-result"
NL_INTERMEDIATE_TOPIC_TYPE  = "nl-intermediate-result"
NL_RESULT_TOPIC_TYPE        = "nl-final-result"
nl_generation_topic_id      = TopicId(type=NL_GENERATION_TOPIC_TYPE     , source="default")
nl_intermediate_topic_id    = TopicId(type=NL_INTERMEDIATE_TOPIC_TYPE   , source="default")
nl_result_topic_id          = TopicId(type=NL_RESULT_TOPIC_TYPE         , source="default")

@dataclass
class NLGenerationTask:
    nl_task: str

@dataclass
class NLGenerationResult:
    nl_task: str
    nl_question: str
    review: str
    n_rev: int
    input_tokens: int
    output_tokens: int

@dataclass
class NLReviewTask:
    nl_task: str
    nl_question: str

@dataclass
class NLReviewResult:
    review: str
    json_review: dict
    approved: bool
    input_tokens: int
    output_tokens: int



# Debating 

JOIN_EVALUATION_TOPIC_TYPE = "evaluation-result"
final_join_evaluation_topic_id = TopicId(type=JOIN_EVALUATION_TOPIC_TYPE, source="default")

@dataclass
class Question:
    content: str


@dataclass
class Answer:
    score: int

@dataclass
class IntermediateEvaluatorResponse:
    content: str
    question: str
    answer: str
    nround: int

@dataclass
class EvaluatorRequest:
    content: str
    question: str

@dataclass
class FinalEvaluatorResponse:
    answer: str

@dataclass
class ResetOrder:
    pass