import json
import uuid
import asyncio
import logging
import logging.handlers

from typing import List, Dict
from autogen_core.tools import Tool
from autogen_core import (
    TopicId,
    default_subscription,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    message_handler,
    CancellationToken
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    FunctionExecutionResult    
)

from orqa.agents.utils import *


__all__ = ['SQLQueryGeneratorAgent']

@default_subscription
class SQLQueryGeneratorAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, system_message: str, tool_schema: List[Tool], output_results: List, max_num_revisions: int = 3) -> None:
        super().__init__("Natural Language and SQL query generator assistant")
        self._system_messages   : List[LLMMessage] = [SystemMessage(content=system_message)]
        self._session_memory    : Dict[str, List[SQLGenerationTask| SQLReviewTask | SQLReviewResult | NLGenerationTask | NLReviewTask | NLReviewResult]] = dict()
        self._num_sql_revisions : Dict[str, int] = dict()
        self._num_nl_revisions  : Dict[str, int] = dict()
        self._final_sql_queries : Dict[str, str | None] = dict()
        self._final_nl_questions: Dict[str, str | None] = dict()
        
        self._model_client = model_client
        self._tools = tool_schema
        self._output_results = output_results
        self.max_num_revisions = max_num_revisions

    @message_handler
    async def handle_sql_generate_task(self, message: SQLGenerationTask, ctx: MessageContext) -> None:
        # Create a session of messages.
        session_id = str(uuid.uuid4())
        self._session_memory.setdefault(session_id, []).append(message)
        self._num_sql_revisions.setdefault(session_id, -1)
        self._final_sql_queries.setdefault(session_id, None)
        
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
        assert len(func_exe_result) > 0, f"Empty function result set! {func_exe_result}"
        assert isinstance(eval(func_exe_result[0].content), dict), f"Wrong function return type! {func_exe_result}"

        func_exe_result = eval([r.content for r in func_exe_result][0])
            
        sql_query = func_exe_result["sql_query"]
        del func_exe_result["sql_query"]
        self._final_sql_queries[session_id] = sql_query

        sql_review_task = SQLReviewTask(
            session_id=session_id,
            sql_task=message.sql_task,
            sql_query=sql_query,
            execution_result=str(func_exe_result)
        )

        logger = logging.getLogger("agentJobLogger")
        logger.debug(f"In handle_generate_task: {func_exe_result=} ##### {sql_query=}")        

        self._session_memory[session_id].append(sql_review_task)        
        await self.publish_message(sql_review_task, topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_sql_review_result(self, message: SQLReviewResult, ctx: MessageContext) -> None:
        self._session_memory[message.session_id].append(message)
        logger = logging.getLogger('agentJobLogger')

        review_request = next(
            m for m in reversed(self._session_memory[message.session_id]) 
            if isinstance(m, SQLReviewTask)
        )
        
        assert review_request is not None

        self._num_sql_revisions[message.session_id] += 1
        old_sql_query = review_request.sql_query

        if message.approved:
            self._output_results.append([self._num_sql_revisions[message.session_id], self._final_sql_queries[message.session_id]])
            # publish the code writing result
            await self.publish_message(
                SQLGenerationResult(
                    sql_task=review_request.sql_task,
                    sql_query=review_request.sql_query,
                    review=message.review
                ),
                topic_id=TopicId("default", self.id.key)
            )
        elif self._num_sql_revisions[message.session_id] >= self.max_num_revisions:            
            self._output_results.append([self._num_sql_revisions[message.session_id], f"FAILURE: {self._final_sql_queries[message.session_id]}"])

            # publish the code writing result
            await self.publish_message(
                SQLGenerationResult(
                    sql_task=review_request.sql_task,
                    sql_query=f"FAILURE: {self._final_sql_queries[message.session_id]}",
                    review=message.review
                ),
                topic_id=TopicId("default", self.id.key)
            )
        else:
            # create a list of LLM messages to send to the model
            messages: List[LLMMessage] = [*self._system_messages]

            # each message isn't a Query** dataclass, but a default message type
            # with some specific source
            reviews = [message.review]
            for m in self._session_memory[message.session_id]:
                if isinstance(m, SQLReviewResult):
                    reviews.append(eval(m.review)['suggested_changes'])
                    messages.append(UserMessage(content=m.review, source="Reviewer"))
                elif isinstance(m, SQLReviewTask):
                    messages.append(AssistantMessage(content=m.sql_query, source="QueryGenerator"))
                elif isinstance(m, SQLGenerationTask):
                    task = m.sql_task
                    messages.append(UserMessage(content=m.sql_task, source="User"))
                else:
                    raise ValueError(f"Unexpected message type: {m}")
            
            # TODO handle this part
            m = f"""Re-consider the task: {task} and rewrite the query considering the reviews: {'\n'.join(reviews)}. Always use `` to encapsulate table and column names."""

            # generate a revision using the chat completion API
            response = await self._model_client.create(
                self._system_messages + [UserMessage(content=m, source=self.metadata["type"])],
                tools=self._tools, 
                cancellation_token=ctx.cancellation_token
            )

            # the agent must generate a tool call to verify its SQL query
            assert isinstance(response.content, list) and all(
                isinstance(call, FunctionCall) for call in response.content
            ) and len(response.content) == 1

            # Execute the tool calls.
            func_exe_result = await asyncio.gather(
                *[self._execute_tool_call(call, ctx.cancellation_token) for call in response.content])
            func_exe_result = eval([r.content for r in func_exe_result][0])
            
            sql_query = func_exe_result["sql_query"]
            del func_exe_result["sql_query"]

            self._final_sql_queries[message.session_id] = sql_query

            logger.debug(f"In handle_review_task: {func_exe_result=} ##### {sql_query=}")            

            query_review_task = SQLReviewTask(
                session_id=message.session_id,
                sql_task=review_request.sql_task,
                sql_query=sql_query,
                execution_result=str(func_exe_result)
            )

            # store the question review task in the session memory
            self._session_memory[message.session_id].append(query_review_task)

            # publish a new review task
            await self.publish_message(query_review_task, topic_id=TopicId("default", self.id.key))

        logger.debug(f"""
Question Generator:
{"-" * 100}

Session ID: {message.session_id}
Number of Reviews: {self._num_sql_revisions[message.session_id]}
{"-" * 100}
Query (old):
{old_sql_query}
{"-" * 100}
Approved:
{message.approved}
{"-" * 100}
Query (new):
{self._final_sql_queries[message.session_id]}
{"-" * 100}
""")
        
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

    @message_handler
    async def handle_nl_generate_task(self, message: NLGenerationTask, ctx: MessageContext) -> None:
        # Create a session of messages.
        session_id = str(uuid.uuid4())
        self._session_memory.setdefault(session_id, []).append(message)
        self._num_nl_revisions.setdefault(session_id, -1)
        self._final_nl_questions.setdefault(session_id, None)
        
        # Run the chat completion with the tools.
        response = await self._model_client.create(
            messages=self._system_messages + [UserMessage(content=message.nl_task, source=self.metadata["type"])],             
            cancellation_token=ctx.cancellation_token
        )

        # the agent must generate a tool call to verify its SQL query        
        assert isinstance(response.content, str), "The Agent has returned a function call instance!"

        nl_question = response.content
        self._final_nl_questions[session_id] = nl_question

        nl_review_task = NLReviewTask(
            session_id=session_id,
            nl_task=message.nl_task,
            nl_question=nl_question
        )

        self._session_memory[session_id].append(nl_review_task)        
        await self.publish_message(nl_review_task, topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_nl_review_result(self, message: NLReviewResult, ctx: MessageContext) -> None:
        self._session_memory[message.session_id].append(message)
        logger = logging.getLogger('agentJobLogger')

        review_request = next(
            m for m in reversed(self._session_memory[message.session_id]) 
            if isinstance(m, NLReviewTask)
        )
        
        assert review_request is not None

        self._num_nl_revisions[message.session_id] += 1
        old_nl_question = review_request.nl_question

        if message.approved:
            self._output_results.append([self._num_nl_revisions[message.session_id], self._final_nl_questions[message.session_id]])
            # publish the code writing result
            await self.publish_message(
                NLGenerationResult(
                    nl_question=review_request.nl_question,
                    nl_task=review_request.nl_task,
                    review=message.review
                ),
                topic_id=TopicId("default", self.id.key)
            )
        elif self._num_nl_revisions[message.session_id] >= self.max_num_revisions:            
            self._output_results.append([self._num_nl_revisions[message.session_id], f"FAILURE: {self._final_nl_questions[message.session_id]}"])

            # publish the code writing result
            await self.publish_message(
                NLGenerationResult(
                    nl_task=review_request.nl_task,
                    nl_question=f"FAILURE: {self._final_nl_questions[message.session_id]}",
                    review=message.review
                ),
                topic_id=TopicId("default", self.id.key)
            )
        else:
            # create a list of LLM messages to send to the model
            messages: List[LLMMessage] = [*self._system_messages]

            # each message isn't a Query** dataclass, but a default message type
            # with some specific source
            for m in self._session_memory[message.session_id]:
                if isinstance(m, NLGenerationTask):
                    task = m.nl_task
                    messages.append(UserMessage(content=m.nl_task, source="User"))
                
            m = f"""Re-consider the task: {task} and rewrite the question considering the review: {message.review}. Pay attention also to previous feedbacks."""

            response = await self._model_client.create(
                self._system_messages + [UserMessage(content=m, source=self.metadata["type"])],
                cancellation_token=ctx.cancellation_token
            )

            # the agent must generate a tool call to verify its SQL query
            assert isinstance(response.content, str)
            
            nl_question = response.content
            self._final_sql_queries[message.session_id] = nl_question

            question_review_task = NLReviewTask(
                session_id=message.session_id,
                nl_task=review_request.nl_task,
                nl_question=nl_question
            )

            # store the question review task in the session memory
            self._session_memory[message.session_id].append(question_review_task)

            # publish a new review task
            await self.publish_message(question_review_task, topic_id=TopicId("default", self.id.key))

        logger.debug(f"""
NL Question Generator:
{"-" * 100}

Session ID: {message.session_id}
Number of Reviews: {self._num_nl_revisions[message.session_id]}
{"-" * 100}
Question (old):
{old_nl_question}
{"-" * 100}
Approved:
{message.approved}
{"-" * 100}
Question (new):
{self._final_nl_questions[message.session_id]}
{"-" * 100}
""")
