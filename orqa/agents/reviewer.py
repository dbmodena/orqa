import re
import json

from logging import Logger
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


from orqa.agents.utils import SQLReviewTask, SQLReviewResult, NLReviewTask, NLReviewResult, sql_intermediate_topic_id, nl_intermediate_topic_id


@default_subscription
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, logger: Logger | None = None):
        super().__init__("A reviewer agent.")
        self._session_memory: List[SQLReviewTask | SQLReviewResult | NLReviewTask | NLReviewResult] = []
        self._model_client  : ChatCompletionClient = model_client
        self._logger        : Logger | None = logger

        self._system_messages: List[LLMMessage] = [
            SystemMessage(content=(
                "You are a query reviewer. "
                "You focus on the correctness of a proposed query."
                )
            )
        ]

    @message_handler
    async def handle_sql_review_task(self, message: SQLReviewTask, ctx: MessageContext) -> None:                
        # format the prompt for the code review
        # gather the previous feedback if available
        previous_feedback =""
        
        previous_review = next(
            (m for m in reversed(self._session_memory) 
                if isinstance(m, SQLReviewResult)),
            None
        )
        if previous_review is not None:
            previous_feedback = previous_review.review

        # store the messages in a temporary memory for this request only
        self._session_memory.append(message)

        prompt = (
            f"The problem statement is:\n{message.sql_task}\n"
            f"The proposed SQL query is:\n{message.sql_query}\n"
            f"The execution of this query is:\n{message.execution_result}\n"
            f"Previous feedback:\n{previous_feedback}\n"
            "If the tool execution was successful, everything is already good and not revise."
            "Don't approve the query if:\n"
            "- Previous feedback was not addressed.\n"
            "- the query is too simple (like 'SELECT * FROM table1 JOIN table2').\n"
            "- the query is identical to old questions.\n"            
            "Respond with the following format:\n"
            "```json\n{\n"
            "    \"correctness\": \"<Your comments>\",\n"
            "    \"approval\": \"<APPROVE or REVISE>\",\n"
            "    \"suggested_changes\": \"<Your comments>\"\n"
            "}\n```"
        )
        # "- column names are not used correctly or not enclosed by ``.\n"

        response = await self._model_client.create(
            self._system_messages + [UserMessage(content=prompt, source=self.metadata["type"])],
            cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        review = self._extract_that_damn_json_output(response.content, self._logger)
        
        # construct the review text
        review_text = f"Query review: {'\n'.join([f'{k}: {v}' for k, v in review.items()])}"

        approved = review['approval'].lower().strip() == 'approve'
        review_result = SQLReviewResult(
            review=review_text,
            approved=approved,
            json_review=review
        )

        self._session_memory.append(review_result)

        # publish the review result
        await self.publish_message(review_result, topic_id=sql_intermediate_topic_id)
        
        """
        if self._logger:
            self._logger.debug((
                f'\n{"-" * 100}\n'
                "Review Result:\n"
                f"Query:\n{message.sql_query}\n"
                f"Review:\n{review_result.review}\n"
                f'{"-" * 100}\n'
                )
            )
        """

    @message_handler
    async def handle_nl_review_task(self, message: NLReviewTask, ctx: MessageContext) -> None:
        # format the prompt for the code review
        # gather the previous feedback if available
        previous_feedback =""
        previous_review = next(
            (m for m in reversed(self._session_memory) 
                if isinstance(m, NLReviewResult)),
            None
        )
        if previous_review is not None:
            previous_feedback = previous_review.review

        self._session_memory.append(message)

        prompt = (
            f"The problem statement is:\n{message.nl_task}\n"
            f"The proposed Natural Language Question is:\n{message.nl_question}\n"            
            f"Previous feedback:\n{previous_feedback}\n"            
            "Don't approve the question if:\n"
            "- Previous feedback was not addressed.\n"
            "- the question is too simple (like 'Where is Canada?').\n"
            "- The question seems to be uncorrelated to the task or not clear..\n"
            "- columns and tables names are explicitly present into the question..\n"
            "Respond with the following format:\n"
            "```json\n{\n"
            "    \"correctness\": \"<Your comments>\",\n"
            "    \"approval\": \"<APPROVE or REVISE>\",\n"
            "    \"suggested_changes\": \"<Your comments>\"\n"
            "}\n```"
        )

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
            review=review_text,
            approved=approved,
            json_review=review
        )

        self._session_memory.append(review_result)

        """
        if self._logger:
            self._logger.debug((
                f'\n{"-" * 100}\n'
                "Review Result:\n"
                f"Question:\n{message.nl_question}\n"
                f"{review_result.review}"
                f'\n{"-" * 100}\n'
                )
            )
        """

        if approved:
            # if the review was successful, then current data are no longer needed
            self._session_memory.clear()

        # publish the review result
        await self.publish_message(review_result, topic_id=nl_intermediate_topic_id)


    def _extract_that_damn_json_output(self, content, logger):
        try:
            return json.loads(re.search(r"```(\w+)(\s*?)(.*?)(\s*?)```", content, re.DOTALL).groups()[2])
        except:
            try:
                return json.loads(re.search(r"(\{.*?\})", content, re.DOTALL).groups()[-1])
            except:
                if self._logger:
                    self._logger.error(f"Bad formatting in JSON review response:\n{content}")
                raise ValueError(f"Bad formatting in JSON review response: {content}")
            