from dataclasses import dataclass

# For SQL Query Generation

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
    approved: bool



# For Natural Language Generation 

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
    approved: bool

