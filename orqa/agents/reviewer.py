import json
import warnings
from logging import Logger
from typing import List

from autogen_core import (
    default_subscription,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
    SystemMessage,
    AssistantMessage
)

from orqa.agents.utils import SQLReviewTask, SQLReviewResult, NLReviewTask, NLReviewResult, sql_intermediate_topic_id, nl_intermediate_topic_id

warnings.filterwarnings('ignore')



@default_subscription
class ReviewerAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, system_prompt: str, task_prompt_template: str, max_reviews: int = 3, logger: Logger | None = None):
        super().__init__("A reviewer agent.")
        self._session_memory: List[SQLReviewTask | SQLReviewResult | NLReviewTask | NLReviewResult] = []
        self._model_client  : ChatCompletionClient = model_client
        self._logger        : Logger | None = logger
        self._max_reviews   : int = max_reviews
        self._n_reviews     : int = 0

        self._system_messages: List[LLMMessage] = [SystemMessage(content=system_prompt)]
        self._task_prompt_template: str = task_prompt_template

    @message_handler
    async def handle_review_task(self, message: SQLReviewTask | NLReviewTask, ctx: MessageContext) -> None:
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

        if isinstance(message, SQLReviewTask):
            prompt = self._task_prompt_template.format(message.sql_task, message.sql_query, 
                                                       message.execution_result, previous_feedback)
        elif isinstance(message, NLReviewTask):
            prompt = self._task_prompt_template.format(message.nl_task, message.nl_question, previous_feedback)

        session = self._system_messages + [UserMessage(content=prompt, source=self.metadata["type"])]
        response = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
            json_output=True
        )

        try:
            # count the current input/output session input tokens
            input_tokens  = self._model_client.count_tokens(messages=session)
            output_tokens = self._model_client.count_tokens(messages=[AssistantMessage(content=response.content, source=self.metadata["type"])])
        except:
            input_tokens = 0
            output_tokens = 0

        assert isinstance(response.content, str)
        review = json.loads(response.content)

        # construct the review text
        review_text = f"Query review: {'\n'.join([f'{k}: {v}' for k, v in review.items()])}"

        approved = review['approval'].lower().strip() == 'approve'
        if isinstance(message, SQLReviewTask):
            review_result = SQLReviewResult(
                review=review_text,
                approved=approved,
                json_review=review,
                execution_result=message.execution_result,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        elif isinstance(message, NLReviewTask):
            review_result = NLReviewResult(
                review=review_text,
                approved=approved,
                json_review=review,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        self._session_memory.append(review_result)

        self._n_reviews += 1
        if approved or self._n_reviews >= self._max_reviews:
            self._n_reviews = 0
            self._session_memory.clear()

        # publish the review result
        if isinstance(message, SQLReviewTask):
            await self.publish_message(review_result, topic_id=sql_intermediate_topic_id)
        elif isinstance(message, NLReviewTask):
            await self.publish_message(review_result, topic_id=nl_intermediate_topic_id)

