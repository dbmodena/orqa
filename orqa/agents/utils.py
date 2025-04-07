from dataclasses import dataclass
from autogen_core import TopicId


# For SQL Query Generation

SQL_GENERATION_TOPIC_TYPE   = "sql-generation-result"
SQL_INTERMEDIATE_TOPIC_TYPE = "sql-intermediate-result"
SQL_RESULT_TOPIC_TYPE       = "sql-final-result"
sql_generation_topic_id     = TopicId(type=SQL_GENERATION_TOPIC_TYPE    , source="default")
sql_intermediate_topic_id   = TopicId(type=SQL_INTERMEDIATE_TOPIC_TYPE  , source="default")
sql_result_topic_id         = TopicId(type=SQL_RESULT_TOPIC_TYPE        , source="default")


@dataclass
class SQLGenerationTask:
    sql_task: str

@dataclass
class SQLGenerationResult:
    sql_task: str
    sql_query: str
    review: str
    n_rev: int

@dataclass
class SQLReviewTask:
    sql_task: str
    sql_query: str
    execution_result: str

@dataclass
class SQLReviewResult:
    review: str
    json_review: dict
    approved: bool



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

@dataclass
class NLReviewTask:
    nl_task: str
    nl_question: str

@dataclass
class NLReviewResult:
    review: str
    json_review: dict
    approved: bool


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
    received: list
    