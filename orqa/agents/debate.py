import re
import logging
from typing import Dict, List

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)

from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from orqa.agents.utils import (
    IntermediateEvaluatorResponse, Question, Answer, EvaluatorRequest, FinalEvaluatorResponse, ResetOrder,
    final_evaluation_topic_id
)


@default_subscription
class Evaluator(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, topic_type: str, 
                 num_neighbors: int, max_rounds: int, 
                 min_score:int = 0, max_score: int = 5, 
                 logger: logging.Logger|None = None):
        super().__init__("Evaluates candidates with other agents")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._round = 0
        self._max_round = max_rounds
        self._min_score = min_score
        self._max_score = max_score
        self._logger = logger
        self._history: List[LLMMessage] = []
        self._buffer : Dict[int, List[IntermediateEvaluatorResponse]] = {}
        self._system_messages = [
            SystemMessage(
                content=(
                    "You are an helpful assistant in tabular data comprehension. "
                    "Your task is to evaluate pairs of candidate tables for a SQL operation "
                    "by providing a numerical score. If given, reason on other assistants observations. "
                    "Limit your output to 50 words, "
                    f"and your final answer should be a single integer number, between {self._min_score} and {self._max_score}. "
                    "Respond with the form:\n"
                    "Answer: <your numerical score here>\n"
                    "Explanation: <your concise explanation>"
                )
            )
        ]

    @message_handler
    async def handle_request(self, message: EvaluatorRequest, ctx: MessageContext) -> None:
        # add the question to the memory
        self._history.append(UserMessage(content=message.content, source="User"))

        # make an inference using the underlying model
        response = await self._model_client.create(self._system_messages + self._history)
        assert isinstance(response.content, str)

        # add the response to the memory
        self._history.append(AssistantMessage(content=response.content, source=self.metadata["type"]))        

        if self._logger:
            self._logger.debug(f"\n{'-' * 80}\nEvaluator {self.id} round {self._round}:\n{response.content}")
        
        try:
            # get the integer score
            m = re.search(r"answer: (\d+)", response.content.lower())
            answer = int(m.group(1))
            assert self._min_score <= answer <= self._max_score
        except:
            if self._logger: self._logger.error(f"No valid response! {response.content}")
            answer = -1

        # increment the round number
        self._round += 1

        if self._round >= self._max_round:
            # if we reach the max number of rounds, publish the final answer
            await self.publish_message(FinalEvaluatorResponse(answer=answer), topic_id=DefaultTopicId())
        else:
            # otherwise continue with an intermediate response
            await self.publish_message(
                IntermediateEvaluatorResponse(
                    content=response.content,
                    task=message.question,
                    answer=answer,
                    nround=self._round
                ),
                topic_id=DefaultTopicId()
            )

    @message_handler
    async def handle_response(self, message: IntermediateEvaluatorResponse, ctx: MessageContext) -> None:
        # Add neighbor's response to the buffer.
        self._buffer.setdefault(message.nround, []).append(message)

        # Check if all neighbors have responded.
        if len(self._buffer[message.nround]) == self._num_neighbors:
            if self._logger:
                self._logger.debug(
                    f"\n{'-'*80}\nSolver {self.id} round {message.nround}:\nReceived all responses from {self._num_neighbors} neighbors."
                )
            
            # Prepare the prompt for the next question.
            prompt = "These are the evaluations from other agents:\n"
            for resp in self._buffer[message.nround]:
                prompt += f"One agent evaluation: {resp.content}\n"
            
            prompt += (
                "Using the evaluations from other agents as additional information, "
                "provide your score to the current table pairs. "
                f"The original task is {message.task}. "
            )

            # Send the question to the agent itself to solve.
            await self.send_message(EvaluatorRequest(content=prompt, question=message.task), self.id)
            
            # Clear the buffer.
            self._buffer.pop(message.nround)
        
    @message_handler
    async def handle_reset(self, message: ResetOrder, ctx: MessageContext) -> None:
        # set the round number to 0
        if self._logger: 
            self._logger.debug(f"Solver {self.id} has done reset.")
        self._round = 0
        self._history = []
        self._buffer = {}


@default_subscription
class ScoreAggregator(RoutedAgent):
    def __init__(self, num_solvers: int, min_score:int = 0, max_score:int = 5, logger: logging.Logger|None = None) -> None:
        super().__init__("A score aggregator")
        self._num_solvers = num_solvers
        self._logger = logger
        self._min_score = min_score
        self._max_score = max_score
        self._buffer: List[FinalEvaluatorResponse] = []

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        if self._logger: 
            self._logger.debug(f"\n{'-' * 80}\nAggregator {self.id} received question: \n{message.content}")
        
        prompt = (
            f"Give a score for the following task:\n{message.content}\n"
            "Explain briefly your reasoning. Your final answer should be a single integer number, " \
            f"between {self._min_score} and {self._max_score}, respond with the form:\n"
            "Answer: <your numberical score here>\n"
            "Explanation: <your concise explanation>"
        )
        if self._logger: 
            self._logger.debug(f"\n{'-' * 80}\nAggregator {self.id} published initial request.")
        await self.publish_message(EvaluatorRequest(content=prompt, question=message.content), topic_id=DefaultTopicId())
        
    @message_handler
    async def handle_final_solver_response(self, message: FinalEvaluatorResponse, ctx: MessageContext) -> None:
        self._buffer.append(message)
        if len(self._buffer) == self._num_solvers:
            if self._logger:
                self._logger.debug(f"\n{'-'*80}\nAggregator {self.id} received all final answers from {self._num_solvers} solvers.")
            
            # Find the majority answer.
            answers = [resp.answer for resp in self._buffer]
            majority_answer = max(set(answers), key=answers.count)
            
            # Publish the aggregated response.
            await self.publish_message(
                Answer(score=majority_answer), 
                topic_id=final_evaluation_topic_id
            )

            # Send the reset message to the solvers
            await self.publish_message(ResetOrder(), topic_id=DefaultTopicId())
            
            # Clear the responses.
            self._buffer.clear()
