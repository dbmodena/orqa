import re
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
    def __init__(self, model_client: ChatCompletionClient, max_reviews: int = 3, logger: Logger | None = None):
        super().__init__("A reviewer agent.")
        self._session_memory: List[SQLReviewTask | SQLReviewResult | NLReviewTask | NLReviewResult] = []
        self._model_client  : ChatCompletionClient = model_client
        self._logger        : Logger | None = logger
        self._max_reviews   : int = max_reviews
        self._n_reviews     : int = 0

        self._system_messages: List[LLMMessage] = [
            SystemMessage(content=(
                "You are a query reviewer. "
                "You focus on the correctness of proposed SQL queries or Natural Language Questions."
                "For the SQL, focus on the query syntax. "
                "Consider that is used DuckDB syntax."
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
            "Revise the query if the execution was not successful."
            "In the query has given an error, check if:\n"
            "- Previous feedback was not addressed.\n"
            "- the query does not involve required columns (if any).\n"
            "- the query is identical to any previously generated query.\n"
            "Respond with the following format:\n"
            "```json\n{\n"
            "    \"correctness\": \"<Your comments>\",\n"
            "    \"approval\": \"<APPROVE or REVISE>\",\n"
            "    \"suggested_changes\": \"<Your comments>\"\n"
            "}\n```"
        )

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
        review = json.loads(response.content) # self._extract_json_output(response.content)
        #review = self._extract_json_output(response.content)
        
        # construct the review text
        review_text = f"Query review: {'\n'.join([f'{k}: {v}' for k, v in review.items()])}"

        approved = review['approval'].lower().strip() == 'approve'
        review_result = SQLReviewResult(
            review=review_text,
            approved=approved,
            json_review=review,
            execution_result=message.execution_result,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        self._session_memory.append(review_result)

        self._n_reviews += 1
        if approved or self._n_reviews >= self._max_reviews:
            self._n_reviews = 0
            self._session_memory.clear()

        # publish the review result
        await self.publish_message(review_result, topic_id=sql_intermediate_topic_id)
        
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
            "- The question is too generic (like 'What is the average value?') or simple (like 'Where is Canada?').\n"
            "- The question seems to be uncorrelated to the current task.\n"
            "- columns and tables names are explicitly present into the question.\n"
            "- columns required by the user are not correctly used (if any).\n"
            "- the question use too specific terms, like 'tables', 'datasets', 'packages', 'data', 'records'.\n"
            "Respond with the following format:\n"
            "```json\n{\n"
            "    \"correctness\": \"<Your comments>\",\n"
            "    \"approval\": \"<APPROVE or REVISE>\",\n"
            "    \"suggested_changes\": \"<Your comments>\"\n"
            "}\n```"
        )

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

        # parse the response JSON
        review = json.loads(response.content) # self._extract_json_output(response.content)

        # construct the review text
        review_text = f"""
            Question review:
            {'\n'.join([f'{k}: {v}' for k, v in review.items()])}
        """

        approved = review['approval'].lower().strip() == 'approve'
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
        await self.publish_message(review_result, topic_id=nl_intermediate_topic_id)


    def _extract_json_output(self, content: str):
        try:        
            return json.loads(re.search(r"```(\w+)(\s*?)(.*?)(\s*?)```", content, re.DOTALL).groups()[2])
        except:
            try:
                return json.loads(re.search(r"(\{.*?\})", content, re.DOTALL).groups()[-1])
            except:
                if self._logger: self._logger.error(f"Bad formatting in JSON review response:\n{content}")
                return {
                    "correctness": content,
                    "approval": "APPROVE" if "approval: APPROVE" in content else "REVISE",
                    "suggested_changes": "Formatting error in reviewer response: check the 'correctness' field for its full response."
                }
                
            