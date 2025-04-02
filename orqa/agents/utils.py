from dataclasses import dataclass

from autogen_core import TopicId

# For SQL Query Generation


SQL_GENERATION_TOPIC_TYPE = "sql-generation-result"
sql_generation_topic_id = TopicId(type=SQL_GENERATION_TOPIC_TYPE, source="default")


@dataclass
class SQLGenerationTask:
    sql_task: str

@dataclass
class SQLGenerationResult:
    sql_task: str
    sql_query: str
    review: str

@dataclass
class SQLReviewTask:
    session_id: str
    sql_task: str
    sql_query: str
    execution_result: str

@dataclass
class SQLReviewResult:
    session_id: str
    review: str
    json_review: dict
    approved: bool



# For Natural Language Generation

NL_GENERATION_TOPIC_TYPE = "nl-generation-result"
nl_generation_topic_id = TopicId(type=NL_GENERATION_TOPIC_TYPE, source="default")

@dataclass
class NLGenerationTask:
    nl_task: str

@dataclass
class NLGenerationResult:
    nl_task: str
    nl_question: str
    review: str

@dataclass
class NLReviewTask:
    session_id: str
    nl_task: str
    nl_question: str

@dataclass
class NLReviewResult:
    session_id: str
    review: str
    json_review: dict
    approved: bool



# Debating 

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
    score: int


