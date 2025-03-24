import logging.handlers
import os
import re
import json
import time
import asyncio
import logging
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from typing import List, Dict, Tuple
import duckdb
import sqlite3
from typing_extensions import Annotated

import jsonlines
import pandas as pd
import polars as pl

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.tools import FunctionTool, Tool
from autogen_core import (
    AgentId,
    TopicId,
    DefaultTopicId,
    default_subscription,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
    CancellationToken
)
from autogen_core.models import (
    ChatCompletionClient,
    ModelFamily,
    LLMMessage,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage
)


from orqa.utils import sanitize_string


def get_package_id(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    return metadata[rsc_id]['id']
    

def get_resource_metadata(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    md = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    return md['name'], metadata[rsc_id]['title'], metadata[rsc_id]['notes']




@dataclass
class QueryGenerationTask:
    sql_task: str

@dataclass
class QueryGenerationResult:
    sql_task: str
    sql_query: str
    review: str

@dataclass
class QueryReviewTask:
    session_id: str
    sql_task: str
    sql_query: str
    execution_result: str

@dataclass
class QueryReviewResult:
    session_id: str
    review: str
    approved: bool


import uuid


@default_subscription
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, system_message: str):
        super().__init__("A query reviewer agent.")
        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._session_memory : Dict[str, List[QueryReviewTask | QueryReviewResult]] = dict()
        self._model_client = model_client

    @message_handler
    async def handle_review_task(self, message: QueryReviewTask, ctx: MessageContext) -> None:
        # format the prompt for the code review
        # gather the previous feedback if available
        previous_feedback =""
        if message.session_id in self._session_memory:
            previous_review = next(
                (m for m in reversed(self._session_memory[message.session_id]) 
                 if isinstance(m, QueryReviewResult)),
                None
            )
            if previous_review is not None:
                previous_feedback = previous_review.review

        # store the messages in a temporary memory for this request only
        self._session_memory.setdefault(message.session_id, []).append(message)
        prompt = f"""
            The problem statement is: 
            {message.sql_task}

            The proposed SQL query is: 
            {message.sql_query}

            The execution of this query is:
            {message.execution_result}

            Previous feedback: 
            {previous_feedback}

            If the tool execution was successful, everything is already good and not revise.
            Please review the query. If previous feedback was provided, see if it was addressed.
            Respond with the following format:
        """ + """
            ```json
            {
                "correctness": "<Your comments>",
                "approval": "<APPROVE or REVISE>",
                "suggested_changes": "<Your comments>"
            }
            ```
        """

        response = await self._model_client.create(
            self._system_messages + [UserMessage(content=prompt, source=self.metadata["type"])],
            cancellation_token=ctx.cancellation_token,
            # json_output=True
        )

        assert isinstance(response.content, str)

        # parse the response JSON
        # logger = logging.getLogger("agentJobLogger")
        # logger.info('>>>' + c + '<<<')
        m = re.search(r"```(\w+)\s*(.*?)\s*```", response.content, re.DOTALL)        
        # logger.info(m)
        m = m.groups()
        # logger.info(m)

        review = json.loads(m[1])

        # construct the review text
        review_text = f"""
            Query review:
            {'\n'.join([f'{k}: {v}' for k, v in review.items()])}
        """

        approved = review['approval'].lower().strip() == 'approve'
        result = QueryReviewResult(
            review=review_text,
            session_id=message.session_id,
            approved=approved
        )

        self._session_memory[message.session_id].append(result)

        # publish the review result
        await self.publish_message(result, topic_id=TopicId("default", self.id.key))




@default_subscription
class QueryGeneratorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, system_message: str, tool_schema: List[Tool], output_results: List) -> None:
        super().__init__("Natural Language and SQL query generator assistant")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(content=system_message)
        ]
        self._model_client = model_client
        self._tools = tool_schema
        self._session_memory: Dict[str, List[QueryGenerationTask| QueryReviewTask | QueryReviewResult]] = dict()
        self._number_of_revision: Dict[str, int] = dict()
        self._final_queries: Dict[str, str | None] = dict()
        self._output_results = output_results

    @message_handler
    async def handle_generate_task(self, message: QueryGenerationTask, ctx: MessageContext) -> None:
        # Create a session of messages.
        session_id = str(uuid.uuid4())
        self._session_memory.setdefault(session_id, []).append(message)
        self._number_of_revision.setdefault(session_id, -1)
        self._final_queries.setdefault(session_id, None)
        
        # Run the chat completion with the tools.
        response = await self._model_client.create(
            messages=self._system_messages + [UserMessage(content=message.sql_task, source=self.metadata["type"])], 
            tools=self._tools, 
            cancellation_token=ctx.cancellation_token
        )

        # the agent must generate a tool call to verify its SQL query        
        assert isinstance(response.content, list) and all(
            isinstance(call, FunctionCall) for call in response.content
        )

        # Execute the tool calls.
        func_exe_result = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in response.content]
        )
        func_exe_result = eval([r.content for r in func_exe_result][0])
            
        logger = logging.getLogger("agentJobLogger")
        logger.debug(f"In handle_generate_task: {func_exe_result=}")
        sql_query = func_exe_result["sql_query"]
        del func_exe_result["sql_query"]
        logger.debug(f"In handle_generate_task: {sql_query=}")

        sql_review_task = QueryReviewTask(
            session_id=session_id,
            sql_task=message.sql_task,
            sql_query=sql_query,
            execution_result=str(func_exe_result)
        )

        self._session_memory[session_id].append(sql_review_task)        
        await self.publish_message(sql_review_task, topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_review_result(self, message: QueryReviewResult, ctx: MessageContext) -> None:
        self._session_memory[message.session_id].append(message)
        self._number_of_revision[message.session_id] += 1

        review_request = next(
            m for m in reversed(self._session_memory[message.session_id]) 
            if isinstance(m, QueryReviewTask)
        )
        
        assert review_request is not None

        self._final_queries[message.session_id] = review_request.sql_query

        if message.approved:
            logger = logging.getLogger('agentJobLogger')
            logger.debug("Query Writing Result:")
            logger.debug("-" * 80)
            logger.debug("Task:")
            logger.debug(review_request.sql_task)
            logger.debug("-" * 80)
            logger.debug("Query:")
            logger.debug(review_request.sql_query)
            logger.debug("-" * 80)
            logger.debug("Review:")
            logger.debug(message.review)
            logger.debug("-" * 80)

            self._output_results.append([self._number_of_revision[message.session_id], self._final_queries[message.session_id]])

            # publish the code writing result
            await self.publish_message(
                QueryGenerationResult(
                    sql_task=review_request.sql_task,
                    sql_query=review_request.sql_query,
                    review=message.review
                ),
                topic_id=TopicId("default", self.id.key)
            )
        else:
            # create a list of LLM messages to send to the model
            messages: List[LLMMessage] = [*self._system_messages]
            for m in self._session_memory[message.session_id]:
                if isinstance(m, QueryReviewResult):
                    messages.append(UserMessage(content=m.review, source="Reviewer"))
                elif isinstance(m, QueryReviewTask):
                    messages.append(AssistantMessage(content=m.sql_query, source="QueryGenerator"))
                elif isinstance(m, QueryGenerationTask):
                    messages.append(UserMessage(content=m.sql_task, source="User"))
                else:
                    raise ValueError(f"Unexpected message type: {m}")
            
            # generate a revision using the chat completion API
            response = await self._model_client.create(messages, tools=self._tools, cancellation_token=ctx.cancellation_token)

            # the agent must generate a tool call to verify its SQL query
            assert isinstance(response.content, list) and all(
                isinstance(call, FunctionCall) for call in response.content
            )

            # Add the first model create result to the session.
            messages.append(AssistantMessage(content=response.content, source="QueryGenerator"))

            # Execute the tool calls.
            func_exe_result = await asyncio.gather(
                *[self._execute_tool_call(call, ctx.cancellation_token) for call in response.content]
            )
            func_exe_result = eval([r.content for r in func_exe_result][0])
            
            # logger = logging.getLogger("agentJobLogger")
            # logger.debug(f"Tool execution result: {result}")
            sql_query = func_exe_result["sql_query"]
            del func_exe_result["sql_query"]

            query_review_task = QueryReviewTask(
                session_id=message.session_id,
                sql_task=review_request.sql_task,
                sql_query=sql_query,
                execution_result=str(func_exe_result)
            )

            # store the question review task in the session memory
            self._session_memory[message.session_id].append(query_review_task)

            # publish a new review task
            await self.publish_message(query_review_task, topic_id=TopicId("default", self.id.key))
        
    async def _execute_tool_call(
        self, call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        # Find the tool by name.
        tool = next((tool for tool in self._tools if tool.name == call.name), None)
        assert tool is not None
        # Run the tool and capture the result.
        try:
            arguments = json.loads(call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            return FunctionExecutionResult(
                call_id=call.id, content=tool.return_value_as_string(result), is_error=False, name=tool.name
            )
        except Exception as e:
            return FunctionExecutionResult(call_id=call.id, content=str(e), is_error=True, name=tool.name)




async def verify_sql(
        sql_query: Annotated[str, "A SQL query which represents the natural language question."],
        # natural_language_question: Annotated[str, "A natural language question which involves the two tables and the columns specified by the user."],
        # human_like_question: Annotated[str, "The question without any explicit reference to table and columns names."]
        ):
    con, cur, result, error = None, None, None, None
    try:
        con = sqlite3.connect("tables.db")
        cur = con.cursor()
        cur.execute(sql_query)
        
        result = cur.fetchmany()
    except sqlite3.Error as e:
        error = str(e)
    finally:
        if con: con.close()

    if error:
        return {
            "status": "error",
            "error_description": error,
            "sql_query": sql_query,
            "result": result if result else "N/A"            
        }
    elif len(result) == 0:
        return {
            "status": "error",
            "error_description": "Empty result set",
            "sql_query": sql_query,
            "result": "empty result"
        }
    else:
        return {
            "status": "success", 
            "error_description": "No error",
            "sql_query": sql_query,
            "result": str(result)
            #"natural_language_question": natural_language_question, 
            #"human_like_question": human_like_question
        }



async def amain():
    data_path           = f"{os.path.dirname(__file__)}/../data"
    # tables_path         = f"{data_path}/datasets/CAN/tables/tables_from10000_to15000"
    # metadata_path       = f"{data_path}/datasets/CAN/metadata/metadata_from10000_to15000.jsonl"
    tables_path         = f"{data_path}/datasets/CAN/tables/tables_from0_to10000"
    metadata_path       = f"{data_path}/datasets/CAN/metadata/metadata_from0_to10000.jsonl"
    log_path            = f"{data_path}/log/CAN_QuestGen.log"
    
    evaluated_path      = f"{data_path}/outputs/evaluated_joins.csv"
    queries_path        = f"{data_path}/outputs/generated_queries.csv"
    add_header          = False

    UP_TO_ROW           = 10
    WRITE_BATCH_SIZE    = 10

    MAX_SELF_RETRIES    = 3

    # number of queries to generate from each join
    N_GEN_QUERIES       = 3

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 200
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 3

    # number of values in common between the joinable columns
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    # the model name (here we will use LiteLLM and Ollama models)
    # model               = "ollama/deepseek-r1:14b"
    # model               = "ollama/llama3.3:latest"
    # model               = "ollama/qwen2.5:14b"
    model               = "ollama/deepseek-r1:70b"
    
    base_url            = "http://localhost:4000"
    api_key             = "NotRequiredSinceWeAreLocal"
    temperature         = 0.4
    
    model_info          = {
        "json_output"       : False,
        "vision"            : False,
        "function_calling"  : True,
        "family"            : ModelFamily.R1,
        "keepalive"        : "6h", # to keep the model in memory more time
        "num_ctx"           : 8192 # to increase the context size (not sure)         
    }
    
    final_results           = []
    tmp_results             = []

    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("agentJobLogger")
    logger.setLevel(logging.DEBUG)
    handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=3)
    handler.suffix = "%y-%m-%d_%H:%M:%S.log"
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    stdout_hanlder = logging.StreamHandler()
    logger.addHandler(stdout_hanlder)

    
    logger.info("Reading Table IDs")
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    logger.info("Loading evaluated JOIN pairs")
    joins = pl.read_csv(evaluated_path).filter(pl.col('size_r_col') < 20)

    # create the Natural Language-SQL Query Generator Agent
    # We use LiteLLM, so ollama and other models should not 
    # give problems with this client type anyway 
    logger.info(f"Definine Chat Completion Client ({model=})")
    qgen_model_client = OpenAIChatCompletionClient(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info
    )

    reviewer_model_client = OpenAIChatCompletionClient(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info
    )


    qgen_system_message = """
        You are a smart AI assistant. 
        Your task is to generate SQL queries relative to joining tables.
        
        Respond with the following format:

        SQL: <Your SQL query>
    """

    reviewer_system_message = """
        You are a query reviewer. You focus on the correctness of a proposed query.
        Do not focus on formatting.
        Pay attention that the new query must be different from old ones.
        Respond using the following JSON format:
        {
            "correctness": "<Your comments>",
            "approval": "<APPROVE or REVISE>",
            "suggested_changes": "<Your comments>"
        } 
    """

    # create the agent runtime
    runtime = SingleThreadedAgentRuntime()

    # create the tools used by the agent
    tools: List[Tool] = [
        FunctionTool(
            verify_sql, 
            description="Checks if the proposed SQL query works."
        )
    ]

    # register the query generator agent
    logger.debug("Registering Query Generator Agent")
    await QueryGeneratorAgent.register(
        runtime, "query_generator_agent",
        lambda: QueryGeneratorAgent(qgen_model_client, qgen_system_message, tools, tmp_results)
    )

    logger.debug("Registering Reviewer Agent")
    await ReviewerAgent.register(
        runtime, "reviewer_agent",
        lambda: ReviewerAgent(reviewer_model_client, reviewer_system_message)
    )

    # start processing messages and create one agent
    logger.info("Started Queries Generation")
    
    for i, row in enumerate(joins.rows()[:UP_TO_ROW], start=1):                
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, _, _, r_col_name, s_col_name = row[:6]
            r_rsc_name, _, r_pkg_note = get_resource_metadata(r_tab_id, table_ids, metadata)
            s_rsc_name, _, s_pkg_note = get_resource_metadata(s_tab_id, table_ids, metadata)

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_note = re.sub(r"(\n|\r|\t)", " ", r_pkg_note)[:MAX_LENGTH_NOTES]
            s_pkg_note = re.sub(r"(\n|\r|\t)", " ", s_pkg_note)[:MAX_LENGTH_NOTES]

            # get a small sample from the dataframes
            r_df = pl.read_parquet(f'{tables_path}/{table_ids[r_tab_id]}').select(pl.all().map_elements(sanitize_string, return_dtype=pl.String)).to_pandas().convert_dtypes()
            s_df = pl.read_parquet(f'{tables_path}/{table_ids[s_tab_id]}').select(pl.all().map_elements(sanitize_string, return_dtype=pl.String)).to_pandas().convert_dtypes()

            # drop null columns
            r_df.dropna(axis=1, how='all', inplace=True)
            s_df.dropna(axis=1, how='all', inplace=True)

            # load tables into the database for tool check
            try: r_df.to_sql(name=r_rsc_name, con="sqlite:///tables.db", index=False, if_exists="fail")
            except ValueError: pass

            try: s_df.to_sql(name=s_rsc_name, con="sqlite:///tables.db", index=False, if_exists="fail")
            except ValueError: pass
            
            r_df = r_df.sample(N_ROWS_SAMPLE, replace=True, ignore_index=True)
            s_df = s_df.sample(N_ROWS_SAMPLE, replace=True, ignore_index=True)
            
            old_questions = []
            for nq in range(N_GEN_QUERIES):
                logger.debug("Starting/Resuming Runtime")
                runtime.start()
           
                await runtime.publish_message(                    
                    QueryGenerationTask(sql_task=f"""
                        Generate a SQL query with the following information: 
                        The columns that joins are {r_col_name=}, {s_col_name=}.

                        The r table is:
                        \"{r_rsc_name}\"

                        Example rows: 
                        {r_df},
                        ############################

                        The s table is:
                        \"{s_rsc_name}\",

                        Example rows:
                        {s_df},
                        ############################

                        Create a JOIN SQL query over the r column {r_col_name} and the s column {s_col_name}.
                        
                        Do not modify tables and columns names.
                        Focus on the joining columns. 
                        Use the tool \"verify_sql\" to check if the SQL query is correct.
                        Use only columns that are really present in the schema.
                        In the SQL put column and table names inside ``.
                        Create different abd more complex queries with respect to old questions: {old_questions}.
                        """
                    ),
                    topic_id=DefaultTopicId()                  
                )

                await runtime.stop_when_idle()
                logger.debug("Runtime idle state.")
                n_rev, sql = tmp_results[-1]
                logger.info(f"Output: {n_rev=}, {sql=}")
                old_questions.append(sql)
                final_results.append([r_tab_id, s_tab_id, r_col_name, s_col_name, nq, n_rev, sql])
                
        except Exception as e:
            logger.error(f'Exception: {e}')
        finally:
            pass
    
    try:
        await runtime.stop_when_idle()
        await runtime.close()
    except:
        pass
    logger.info("Final Output: " + str(final_results))
    import csv
    with open(queries_path, 'w') as file:
        csv.writer(file).writerow(["r_tab_id", "s_tab_id", "r_col_name", "s_col_name", "num_query", "num_reviews", "sql"])
        csv.writer(file).writerows(final_results)
    
    logger.info("Done")


if __name__ == '__main__':
    asyncio.run(amain())
