import json
import asyncio
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
    FunctionExecutionResult    
)
from autogen_core.tools import Tool

from orqa.agents.utils import *


__all__ = ['SQLQueryGeneratorAgent', 'NLQuestionGeneratorAgent']

# WITH age_group_dwelling_counts AS ( 
#   SELECT R.age_group, SUM(CAST(REPLACE(S.private_dwellings_occupied_by_usual_residents,_2016, '_', '') AS INTEGER)) AS total_dwellings 
#   FROM R INNER JOIN S ON R.geo = S.geographic_name,_english WHERE R.geo = 'canada' GROUP BY R.age_group 
# ) SELECT age_group, AVG(total_dwellings) AS average_dwellings FROM age_group_dwelling_counts GROUP BY age_group;

@default_subscription
class SQLQueryGeneratorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List[Tool], max_num_revisions: int = 3, logger: Logger|None = None) -> None:
        super().__init__("SQL query generator assistant")
        self._model_client      : ChatCompletionClient  = model_client
        self._tools             : List[Tool] = tool_schema
        self.max_num_revisions  : int = max_num_revisions
        self._logger            : Logger | None = logger

        self._session_memory    : List[SQLGenerationTask| SQLReviewTask | SQLReviewResult] = []
        self._num_sql_revisions : str = -1
        self._system_messages   : List[LLMMessage] = [
            SystemMessage(content=(
                "You are a smart AI assistant. "
                "Your task is to generate SQL queries and natural language questions. "                
                )
            )
        ]
        
    @message_handler
    async def handle_sql_generate_task(self, message: SQLGenerationTask, ctx: MessageContext) -> None:
        # Reset values from previous session
        self._session_memory.clear()
        self._num_sql_revisions = -1

        self._session_memory.append(message)        

        session = self._system_messages + [UserMessage(content=message.sql_task, source=self.metadata["type"])]
        
        # Run the chat completion with the tools.
        response = await self._model_client.create(
            messages=session, 
            tools=self._tools, 
            cancellation_token=ctx.cancellation_token
        )

        # the agent must generate a tool call to verify its SQL query        
        assert isinstance(response.content, FunctionCall) or isinstance(response.content, list) and all(
            isinstance(call, FunctionCall) for call in response.content
        ), f"Bad response content: {type(response.content)=}, {response.content=}"

        # Execute the tool calls.
        func_exe_result = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in response.content]
        )

        assert len(func_exe_result) > 0, f"Empty function result set! {func_exe_result}"
        try:
            assert isinstance(eval(func_exe_result[0].content), dict), f"Wrong function return type! {func_exe_result}"
        except SyntaxError:
            if self._logger: self._logger.error(f"Some error here: {func_exe_result[0]=}")
            if self._logger: self._logger.error(f"Or here: {func_exe_result[0].content=}")

        func_exe_result = eval([r.content for r in func_exe_result][0])
            
        sql_query = func_exe_result["sql_query"]
        del func_exe_result["sql_query"]

        sql_review_task = SQLReviewTask(
            sql_task=message.sql_task,
            sql_query=sql_query,
            execution_result=str(func_exe_result)
        )

        # self._logger.debug(f"SQL results in handle_generate_task: {func_exe_result=}")

        self._session_memory.append(sql_review_task)        
        await self.publish_message(sql_review_task, topic_id=sql_intermediate_topic_id)

    @message_handler
    async def handle_sql_review_result(self, message: SQLReviewResult, ctx: MessageContext) -> None:
        self._session_memory.append(message)

        review_request = next(
            m for m in reversed(self._session_memory) 
            if isinstance(m, SQLReviewTask)
        )
        
        assert review_request is not None

        self._num_sql_revisions += 1
        old_sql_query = review_request.sql_query
        too_many_revs = self._num_sql_revisions >= self.max_num_revisions

        if message.approved or too_many_revs:
            # publish the code writing result            
            await self.publish_message(
                SQLGenerationResult(
                    sql_task=review_request.sql_task,
                    sql_query=("FAILURE: " if too_many_revs else "") + review_request.sql_query,
                    review=message.review,
                    n_rev=self._num_sql_revisions
                ),
                topic_id=sql_result_topic_id
            )
            
        else:
            # create a list of LLM messages to send to the model
            messages: List[LLMMessage] = [*self._system_messages]

            # each message isn't a Query** dataclass, but a default message type
            # with some specific source
            reviews = [message.review]
            for m in self._session_memory:
                if isinstance(m, SQLReviewResult):
                    reviews.append(m.json_review['suggested_changes'])
                    messages.append(UserMessage(content=m.review, source="Reviewer"))
                elif isinstance(m, SQLReviewTask):
                    messages.append(AssistantMessage(content=m.sql_query, source="QueryGenerator"))
                elif isinstance(m, SQLGenerationTask):
                    task = m.sql_task
                    messages.append(UserMessage(content=m.sql_task, source="User"))
                else:
                    raise ValueError(f"Unexpected message type: {m}")
            
            # TODO handle this part
            m = f"Re-consider the task: {task}. Rewrite the query considering the reviews: {'\n'.join(reviews)}. "
            
            # generate a revision using the chat completion API
            response = await self._model_client.create(
                self._system_messages + [UserMessage(content=m, source="User")],
                tools=self._tools, 
                cancellation_token=ctx.cancellation_token
            )

            assert isinstance(response.content, FunctionCall) or isinstance(response.content, list) and all(
                isinstance(call, FunctionCall) for call in response.content
            ), f"Bad response content: {type(response.content)}"

            # Execute the tool calls.
            func_exe_result = await asyncio.gather(
                *[self._execute_tool_call(call, ctx.cancellation_token) for call in response.content])
            try:
                assert len(func_exe_result) > 0, f"Empty function result set! {func_exe_result}"
                assert isinstance(eval(func_exe_result[0].content), dict), f"Wrong function return type! {func_exe_result}"
            except SyntaxError:
                if self._logger: self._logger.error(f"Some error here: {func_exe_result[0]=}")
                if self._logger: self._logger.error(f"Or here: {func_exe_result[0].content=}")

            func_exe_result = eval([r.content for r in func_exe_result][0])
            
            sql_query = func_exe_result["sql_query"]
            del func_exe_result["sql_query"]

            # self._logger.debug(f"SQL results in handle_review_task: {func_exe_result=}")            

            query_review_task = SQLReviewTask(
                sql_task=review_request.sql_task,
                sql_query=sql_query,
                execution_result=str(func_exe_result)
            )

            # store the question review task in the session memory
            self._session_memory.append(query_review_task)

            # publish a new review task
            await self.publish_message(query_review_task, topic_id=sql_intermediate_topic_id)

        self._logger.debug((
            f"\n{'-' * 100}\n"
            "Question Generator:\n"
            f"Number of Reviews: {self._num_sql_revisions}\n\n"            
            f"Query (old):\n{old_sql_query}\n"            
            f"Approved:\n{message.approved}\n"            
            f"Query (new):\n{sql_query}"
            f"\n{'-' * 100}\n"
            )
        )
        
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



@default_subscription
class NLQuestionGeneratorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, max_num_revisions: int = 3, logger: Logger|None = None) -> None:
        super().__init__("Natural Language generator assistant")
        self._model_client      : ChatCompletionClient  = model_client
        self.max_num_revisions  : int = max_num_revisions
        self._logger            : Logger | None  = logger 
        
        self._session_memory    : List[NLGenerationTask | NLReviewTask | NLReviewResult] = []
        self._num_nl_revisions  : int = 0
        self._system_messages   : List[LLMMessage] = [
            SystemMessage(content=(
                "You are a smart AI assistant. "
                "Your task is to generate natural language questions based on SQL queries."
                )
            )
        ]
    
    @message_handler
    async def handle_nl_generate_task(self, message: NLGenerationTask, ctx: MessageContext) -> None:
        # Reset previous session values
        self._session_memory.clear()
        self._num_nl_revisions = -1

        # Add the message to the session memory
        self._session_memory.append(message)
        
        # Run the chat completion with the tools.
        response = await self._model_client.create(
            messages=self._system_messages + [UserMessage(content=message.nl_task, source=self.metadata["type"])],             
            cancellation_token=ctx.cancellation_token
        )

        nl_question = response.content
        
        nl_review_task = NLReviewTask(
            nl_task=message.nl_task,
            nl_question=nl_question
        )

        self._session_memory.append(nl_review_task)        
        await self.publish_message(nl_review_task, topic_id=nl_intermediate_topic_id)

    @message_handler
    async def handle_nl_review_result(self, message: NLReviewResult, ctx: MessageContext) -> None:
        self._session_memory.append(message)        

        review_request = next(
            m for m in reversed(self._session_memory) 
            if isinstance(m, NLReviewTask)
        )
        
        assert review_request is not None

        self._num_nl_revisions += 1
        old_nl_question = review_request.nl_question
        too_many_revs = self._num_nl_revisions >= self.max_num_revisions
        if message.approved or too_many_revs:
            # publish the code writing result
            await self.publish_message(
                NLGenerationResult(
                    nl_question=("FAILURE: " if too_many_revs else "") + review_request.nl_question,
                    nl_task=review_request.nl_task,
                    review=message.review,
                    n_rev=self._num_nl_revisions
                ),
                topic_id=nl_result_topic_id
            )
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

            response = await self._model_client.create(
                self._system_messages + [UserMessage(content=msg, source=self.metadata["type"])],
                cancellation_token=ctx.cancellation_token
            )

            # the agent must generate a tool call to verify its SQL query
            assert isinstance(response.content, str)        
            
            question_review_task = NLReviewTask(
                nl_task=review_request.nl_task,
                nl_question=response.content
            )

            # store the question review task in the session memory
            self._session_memory.append(question_review_task)

            # publish a new review task
            await self.publish_message(question_review_task, topic_id=nl_intermediate_topic_id)

        self._logger.debug((
            f"\n{'-' * 100}\n"
            "NL Question Generator:\n"
            f"Number of Reviews: {self._num_nl_revisions}\n"
            f"Question (old):\n{old_nl_question}\n"
            f"Approved: {message.approved}\n"            
            f"Question (new):\n{response.content}"
            f"\n{'-' * 100}\n"
            )
        )
