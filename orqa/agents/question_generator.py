import json
import asyncio
import warnings
from typing import List
from logging import Logger

from autogen_core import (
    RoutedAgent,
    FunctionCall,
    MessageContext,
    message_handler,
    CancellationToken,
    default_subscription
)
from autogen_core.models import (
    LLMMessage,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
)
from autogen_core.tools import Tool

from orqa.agents.utils import *

warnings.filterwarnings('ignore')


__all__ = ['SQLQueryGeneratorAgent', 'NLQuestionGeneratorAgent']



@default_subscription
class SQLQueryGeneratorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool], max_num_revisions: int = 3, logger: Logger|None = None) -> None:
        super().__init__("SQL query generator assistant")
        self._model_client      : ChatCompletionClient  = model_client
        self._tools             : List[Tool] = tool_schema
        self.max_num_revisions  : int = max_num_revisions
        self._logger            : Logger | None = logger

        self._input_tokens      : int = 0
        self._output_tokens     : int = 0
        self._session_memory    : List[SQLGenerationTask| SQLReviewTask | SQLReviewResult] = []
        self._num_sql_revisions : str = -1
        self._system_messages   : List[LLMMessage] = [
            SystemMessage(content=(
                    "You are a SQL coder assistant. "
                    "Your task is to generate SQL queries of different difficult levels. "
                    "A 'simple' query involves just basic operations, like simple WHERE clauses. "
                    "A 'moderate' query could use also casting, string replacement, grouping functions and other forms of aggregations."
                    "A 'challenging' query may require window functions, subqueries and other complex operations. "
                    "You are using DuckDB: if necessary, put column names inside double-quotes, like \"column_name\". "
                    "Do not cast FLOAT to REAL. "
                    "If a VARCHAR attribute is similar to a datetime, try to cast it to DATE or DATETIME. "
                    "When using regex operations, use proper options. "
                    "Use the given tool to validate your SQL query: your response must be only a valid function call."
                )
            )
        ]


    @message_handler
    async def handle_sql_task(self, message: SQLGenerationTask | SQLReviewResult, ctx: MessageContext) -> None:    
        # create a list of LLM messages to send to the model
        session: List[LLMMessage] = [*self._system_messages] 
        
        # add the message to the current session memory 
        self._session_memory.append(message)
        
        if isinstance(message, SQLGenerationTask):
            # Reset values from previous session
            self._session_memory.clear()
            self._num_sql_revisions = -1
            self._input_tokens = 0
            self._output_tokens = 0
            
            task = message.sql_task
            session.append(UserMessage(content=message.sql_task, source="User"))

        elif isinstance(message, SQLReviewResult):
            review_request = next(
                m for m in reversed(self._session_memory) 
                if isinstance(m, SQLReviewTask)
            )
            
            task = review_request.sql_task
            
            assert review_request is not None

            # keep track if the SQL actually works or not
            # TODO in some cases, the reviewer stucks on some fix need
            # even if it already addressed
            exe_res = eval(review_request.execution_result)
            sql_success = 'success' if exe_res['status'] == "success" else exe_res['error_description']

            # add tokens from the reviewer 
            self._input_tokens += message.input_tokens
            self._output_tokens += message.output_tokens

            too_many_revs = self._num_sql_revisions >= self.max_num_revisions

            # if the laast reveiw was successful or we have reached the max number of reviews,
            #  publish the code writing result
            if message.approved or too_many_revs:
                await self.publish_message(
                    SQLGenerationResult(
                        sql_task=review_request.sql_task,
                        sql_query=f"{'FAILURE: ' if too_many_revs and not message.approved else ''}{review_request.sql_query}",
                        review=message.review,
                        sql_success=sql_success,
                        n_rev=self._num_sql_revisions,
                        input_tokens=self._input_tokens,
                        output_tokens=self._output_tokens
                    ),
                    topic_id=sql_result_topic_id
                )

                return
            else: 
                # list of past reveiws (if any)
                reviews : list[str] = []

                for m in self._session_memory:
                    if isinstance(m, SQLReviewResult):
                        reviews.append(f"Previous execution result: {m.execution_result}. Relative review: {m.review}")
                    
                # TODO handle better this part
                m = (
                    "Re-consider the original task. Be different from old queries. "
                    f"Rewrite the query considering the previous SQL execution outputs and relative reviews: {'\n'.join(reviews)}."
                )
                
                # generate a revision using the chat completion API
                session.append(UserMessage(content=m, source="User"))        

        response = await self._model_client.create(
            messages=session,
            tools=self._tools, 
            cancellation_token=ctx.cancellation_token
        )
    
        try:
            # count the current input/output session input tokens
            self._input_tokens  += self._model_client.count_tokens(messages=session, tools=self._tools)
            self._output_tokens += self._model_client.count_tokens(messages=[AssistantMessage(content=response.content, source=self.metadata["type"])], tools=self._tools)
        except:
            self._input_tokens  += 0
            self._output_tokens += 0
            
        try:
            # should we consider only cases where the model outpus 
            # a perfectly valid function call?
            if isinstance(response.content, FunctionCall):
                func_call = [response.content]
            elif isinstance(response.content, list) and all(isinstance(call, FunctionCall) for call in response.content):
                func_call = response.content
            elif isinstance(response.content, str):
                str_function_call = eval(response.content.replace('json', '').replace('`', ''))
                func_call = [FunctionCall(id=str(hash(response.content)), arguments=str_function_call['arguments'], name=str_function_call['name'])]
            
            # Execute the tool calls.
            func_exe_result = await asyncio.gather(*[self._execute_tool_call(call, ctx.cancellation_token) for call in func_call])
            execution_result : dict = eval([r.content for r in func_exe_result][0])
        except Exception as e:
            if self._logger: self._logger.error(f"Error in tool handling: {e}")
            raise e
        
        sql_query = execution_result.pop("sql_query")
        
        # increase the number of reviews
        self._num_sql_revisions += 1

        query_review_task = SQLReviewTask(
            sql_task=task,
            sql_query=sql_query,
            execution_result=str(execution_result)
        )

        # add the review task to the session memory
        self._session_memory.append(query_review_task)

        if self._logger:
            self._logger.debug(
                (
                    f"\n{'-' * 100}\n"
                    "Question Generator:\n"
                    f"Number of Reviews: {self._num_sql_revisions}\n\n"      
                    f"Execution result:\n{execution_result}\n"
                    f"Query:\n{sql_query}"
                    f"\n{'-' * 100}\n"
                )
            )

        # publish the new review task
        await self.publish_message(query_review_task, topic_id=sql_intermediate_topic_id)
        
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
            return FunctionExecutionResult(call_id=call.id, content=str({"status": "error", "error_description": str(e), "sql_query": None}), is_error=True, name=tool.name)



@default_subscription
class NLQuestionGeneratorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, max_num_revisions: int = 3, logger: Logger|None = None) -> None:
        super().__init__("Natural Language generator assistant")
        self._model_client      : ChatCompletionClient  = model_client
        self.max_num_revisions  : int = max_num_revisions
        self._logger            : Logger | None  = logger         
        
        self._input_tokens      : int = 0
        self._output_tokens     : int = 0
        self._session_memory    : List[NLGenerationTask | NLReviewTask | NLReviewResult] = []
        self._num_nl_revisions  : int = 0
        self._system_messages   : List[LLMMessage] = [
            SystemMessage(content=(
                "Your task is to generate natural language questions, related to table from Open Data, from a given SQL query. "
                "Pretend to be a user that is using Open Data search portals and needs to get answers that's inside the results of that query. "
                "The questions you create must be fluent and human-like: do not use SQL-like words, such as null or select. "
                "Keep focus on join and union operations between tables, if any. "
                "Because a common Open Data user (as you, in this case) does not know anything in advance about the final result, "
                "you can't use terms like records, data, datasets, tables, csv, packages and resources. "
                "If values are used inside the SQL query, try to understand what they means based on the given context: "
                "for example, 'ref' may mean 'refused' in a column about orders status. "
                "You must not use explicit table or column names into the question. "
                "Your response must be only the question, nothing else."
                )
            )
        ]
    
    @message_handler
    async def handle_nl_task(self, message: NLGenerationTask | NLReviewResult, ctx: MessageContext) -> None:
        # create a list of LLM messages to send to the model
        session: List[LLMMessage] = [*self._system_messages] 
        
        if isinstance(message, NLGenerationTask):
            # Reset values from previous session
            self._session_memory.clear()
            self._num_nl_revisions = -1
            self._input_tokens = 0
            self._output_tokens = 0
            
            task = message.nl_task
            session.append(UserMessage(content=message.nl_task, source="User"))

        elif isinstance(message, NLReviewResult):
            review_request = next(
                m for m in reversed(self._session_memory) 
                if isinstance(m, NLReviewTask)
            )            
            assert review_request is not None
            task = review_request.nl_task
            
            # add the tokens from the reviewer
            self._input_tokens += message.input_tokens
            self._output_tokens += message.output_tokens

            # and check termination conditions
            too_many_revs = self._num_nl_revisions >= self.max_num_revisions

            if message.approved or too_many_revs:
                # publish the code writing result
                await self.publish_message(
                    NLGenerationResult(
                        nl_question=f"{'FAILURE: ' if too_many_revs and not message.approved else ''}{review_request.nl_question}",
                        nl_task=review_request.nl_task,
                        review=message.review,
                        n_rev=self._num_nl_revisions,
                        input_tokens=self._input_tokens,
                        output_tokens=self._output_tokens
                    ),
                    topic_id=nl_result_topic_id
                )

                return
            else:
                # create a list of LLM messages to send to the model
                messages: List[LLMMessage] = [*self._system_messages]

                # each message isn't a Query** dataclass, but a default message type
                # with some specific source
                for msg in self._session_memory:
                    if isinstance(msg, NLGenerationTask):
                        task = msg.nl_task
                        messages.append(UserMessage(content=msg.nl_task, source="User"))
                    
                msg = (
                    f"Re-consider the task: {task} and rewrite the question considering the review: {message.review}. "
                    "Pay attention also to previous feedbacks."
                )

                # add the message to the current generation session memory
                session.append(UserMessage(content=msg, source=self.metadata["type"]))

        # create the response with the client
        response = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token
        )

        try:
            # count input/output tokens
            self._input_tokens  += self._model_client.count_tokens(messages=session)
            self._output_tokens += self._model_client.count_tokens(messages=[AssistantMessage(content=response.content, source=self.metadata["type"])])
        except:
            # due to connection error
            self._input_tokens  += 0
            self._output_tokens += 0

        # the agent must generate a tool call to verify its SQL query
        assert isinstance(response.content, str)        
        
        question_review_task = NLReviewTask(
            nl_task=task,
            nl_question=response.content
        )

        # increment the number of revisions
        self._num_nl_revisions += 1

        # store the question review task in the session memory
        self._session_memory.append(question_review_task)

        # publish a new review task
        await self.publish_message(question_review_task, topic_id=nl_intermediate_topic_id)

        if self._logger:
            self._logger.debug(
                (
                    f"\n{'-' * 100}\n"
                    "NL Question Generator:\n"
                    f"Number of Reviews: {self._num_nl_revisions}\n"
                    f"Question (new):\n{response.content}"
                    f"\n{'-' * 100}\n"
                )
            )
