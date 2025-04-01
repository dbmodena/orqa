import re
import json

import logging
from typing import List, Dict

from autogen_core import (
    TopicId,
    default_subscription,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
    SystemMessage
)


from orqa.agents.utils import SQLReviewTask, SQLReviewResult, NLReviewTask, NLReviewResult


@default_subscription
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, system_message: str):
        super().__init__("A reviewer agent.")
        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._session_memory : Dict[str, List[SQLReviewTask | SQLReviewResult | NLReviewTask | NLReviewResult]] = dict()
        self._model_client = model_client
        self._logger = logging.getLogger('agentJobLogger')

    @message_handler
    async def handle_sql_review_task(self, message: SQLReviewTask, ctx: MessageContext) -> None:
        # format the prompt for the code review
        # gather the previous feedback if available
        previous_feedback =""
        if message.session_id in self._session_memory:
            previous_review = next(
                (m for m in reversed(self._session_memory[message.session_id]) 
                 if isinstance(m, SQLReviewResult)),
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
            Don't approve the query if:
                - Previous feedback was not addressed.
                - the query is too simple (like "SELECT * FROM table1 JOIN table2", if there are WHERE, GROUP, etc is ok)
                - column names are not used correctly or not enclosed by ``.
                - the query is identical to old questions.
            
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
            cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        review = self._extract_that_damn_json_output(response.content, self._logger)
        
        # construct the review text
        review_text = f"""
            Query review:
            {'\n'.join([f'{k}: {v}' for k, v in review.items()])}
        """

        approved = review['approval'].lower().strip() == 'approve'
        review_result = SQLReviewResult(
            review=review_text,
            session_id=message.session_id,
            approved=approved,
            json_review=review
        )

        self._session_memory[message.session_id].append(review_result)

        self._logger.debug(f"""
Review Result:
{"-" * 100}

Query:
{message.sql_query}
{"-" * 100}

Review:
{review_result.review}

Approved:
{review_result.approved}

""")        
        # publish the review result
        await self.publish_message(review_result, topic_id=TopicId("default", self.id.key))

    @message_handler
    async def handle_nl_review_task(self, message: NLReviewTask, ctx: MessageContext) -> None:
        # format the prompt for the code review
        # gather the previous feedback if available
        previous_feedback =""
        if message.session_id in self._session_memory:
            previous_review = next(
                (m for m in reversed(self._session_memory[message.session_id]) 
                 if isinstance(m, NLReviewResult)),
                None
            )
            if previous_review is not None:
                previous_feedback = previous_review.review

        # store the messages in a temporary memory for this request only
        self._session_memory.setdefault(message.session_id, []).append(message)
        prompt = f"""
            The problem statement is: 
            {message.nl_task}

            The proposed Natural Language Question is: 
            {message.nl_question}

            Previous feedback: 
            {previous_feedback}
            
            Don't approve the question if:
                - columns and tables names are explicitly present into the question.
                - the question is very very simple (like "Where is Canada?").
                - The question seems to be uncorrelated to the task or not clear.
                - Previous feedback was not addressed.                        
            
            Respond with the following JSON format:
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
            cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)

        # parse the response JSON
        review = self._extract_that_damn_json_output(response.content, self._logger)

        # construct the review text
        review_text = f"""
            Question review:
            {'\n'.join([f'{k}: {v}' for k, v in review.items()])}
        """

        approved = review['approval'].lower().strip() == 'approve'
        review_result = NLReviewResult(
            session_id=message.session_id,
            review=review_text,
            approved=approved,
            json_review=review
        )

        self._session_memory[message.session_id].append(review_result)

        self._logger.debug(f"""
Review Result:
{"-" * 100}

Question:
{message.nl_question}
{"-" * 100}

Review:
{review_result.review}

Approved:
{review_result.approved}
{"-" * 100}
""")        
        # publish the review result
        await self.publish_message(review_result, topic_id=TopicId("default", self.id.key))


    def _extract_that_damn_json_output(self, content, logger):
        try:
            return json.loads(re.search(r"```(\w+)\s*(.*?)\s*```", content, re.DOTALL).groups()[-1])
        except:
            try:
                return json.loads(re.search(r"(\{.*?\})", content, re.DOTALL).groups()[-1])
            except:
                logger.error(f"Bad formatting in JSON review response:\n{content}")
                raise ValueError(f"Bad formatting in JSON review response: {content}")
        