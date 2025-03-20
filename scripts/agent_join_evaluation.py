import re
import os
import re 
import csv
import time
import asyncio
import logging
from typing import List
from typing_extensions import Annotated
from logging.handlers import TimedRotatingFileHandler

import jsonlines
import polars as pl

import asyncio
import json
from dataclasses import dataclass
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import (
    AgentId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
    CancellationToken
)
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    AssistantMessage,
    SystemMessage,
    UserMessage,
    FunctionExecutionResult, 
    FunctionExecutionResultMessage,
    ModelFamily
)
from autogen_core.tools import FunctionTool, Tool
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

from orqa.utils import sanitize_string


# @dataclass
# class Message:
#     content: str

""" 
class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient|OllamaChatCompletionClient, system_message: str, tool_schema: List[Tool]) -> None:
        super().__init__("JOIN Assistant with Tool")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(content=system_message)
        ]
        self._model_client = model_client
        self._tools = tool_schema

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            return Message(content=create_result.content)
        assert isinstance(create_result.content, list) and all(
            isinstance(call, FunctionCall) for call in create_result.content
        )

        # Add the first model create result to the session.
        session.append(AssistantMessage(content=create_result.content, source="assistant"))

        # Execute the tool calls.
        results = await asyncio.gather(
            *[self._execute_tool_call(call, ctx.cancellation_token) for call in create_result.content]
        )

        # Add the function execution results to the session.
        session.append(FunctionExecutionResultMessage(content=results))

        # Run the chat completion again to reflect on the history and function execution results.
        create_result = await self._model_client.create(
            messages=session,
            cancellation_token=ctx.cancellation_token,
        )
        assert isinstance(create_result.content, str)

        # Return the result as a message.
        return Message(content=create_result.content)

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

 """

results = []

# in this case prob is not necessary to equip the assistant agent with this tool
# we can simply call it with agent.create()
async def save_join_score(agent_score: Annotated[int, 
                                                 # TODO integer score 
                          """A score of the JOIN between 0 and 3 where:
                              0 is CASUAL;
                              1 is SUPERFICIAL;
                              2 is ENGAGED;
                              3 is MEANINGFUL;"""
                          ],
                         agent_explanation: Annotated[str, "A clear, short and concise explanation of the given score"]):
    global score, explanation
    score = agent_score
    explanation = agent_explanation
    return agent_score, agent_explanation


async def create_agent(
        model: str = "ollama/llama3.3",
        base_url: str = "http://localhost:4000",
        api_key: str = "NotRequiredSinceWeAreLocal",
        temperature: int = 0,
        model_info: dict = {
            "json_output"       : False,
            "vision"            : False,
            "function_calling"  : True,
            "family"            : ModelFamily.UNKNOWN,
            "keep_alive"        : "6h", # to keep the model in memory more time
            "num_ctx"           : 8192 # to increase the context size (not sure) 
        # }, **kwargs) -> SingleThreadedAgentRuntime:
        }, **kwargs) -> AssistantAgent:
    
    # model_client = OllamaChatCompletionClient(
    model_client = OpenAIChatCompletionClient(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info,
        **kwargs,
    )

    # system_message = """
    #     You are a smart AI assistant.
    #     Do not write code. Do not respond. Only use the provided tool "save_join_score".
    # """
    system_message = """
        You are a smart AI assistant. 
        Your task is to evaluate if two candidate columns which have common values 
        represent or not a relevant join.
        A score of the JOIN is an INTEGER between 0 and 5 where:
            0 is CASUAL;
            5 is MEANINGFUL;
        Do not write code. Return your score between the tags <score>SCORE</score>.
        Use only values between 0 and 5. Do not use tools. Be clear, concise and short.
    """
    
    # create the assistant agent
    agent = AssistantAgent(
        name="JoinEvaluatorAssistant",
        model_client=model_client,
        system_message=system_message
    )
    # return runtime
    return agent


def get_package_id(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    return metadata[rsc_id]['id']
    

def get_resource_metadata(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    md = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    return md['name'], metadata[rsc_id]['title'], metadata[rsc_id]['notes']


async def amain():
    data_path       = f'{os.path.dirname(__file__)}/../data'
    # tables_path     = f'{data_path}/datasets/CAN/tables/tables_from10000_to15000'
    # metadata_path   = f'{data_path}/datasets/CAN/metadata/metadata_from10000_to15000.jsonl'
    tables_path     = f'{data_path}/datasets/CAN/tables/tables_from0_to10000'
    metadata_path   = f'{data_path}/datasets/CAN/metadata/metadata_from0_to10000.jsonl'
    log_path        = f'{data_path}/log/CAN_JoinEval.log'
    
    candidates_path = f'{data_path}/outputs/candidate_joins.csv'
    evaluated_path  = f'{data_path}/outputs/evaluated_joins.csv'

    add_header          = False
    save_explanation    = True

    UP_TO_ROW           = 100
    WRITE_BATCH_SIZE    = 10

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 200
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 3

    # number of values in common between the candidate joinable columnes
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    # the model name (here we will use LiteLLM and Ollama)
    # model               = "ollama/deepseek-r1:14b"
    # model               = "ollama/llama3.3:latest"
    model               = "ollama/qwen2.5:14b"


    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f'indexerLogger')
    logger.setLevel(logging.DEBUG)
    handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=3)
    handler.suffix = "%y-%m-%d_%H:%M:%S.log"
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    # set up the agent
    logger.info(f"Setup the Agent (model {model})")
    agent = await create_agent(model=model)
    logger.info(f"{type(agent)=}, {agent.name=}, {agent._system_messages=}")
    await agent.on_reset(cancellation_token=CancellationToken())
    logger.info(f"After reset: {type(agent)=}, {agent.name=}, {agent._system_messages=}")

    # runtime = await create_agent(model=model)

    # logger.debug("Starting Agent Runtime")
    # runtime.start()
    # tool_use_agent = AgentId(f"tool_use_agent_{model}", "default")

    logger.info("Reading Table IDs")
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    if not add_header:
        logger.info("Loading already evaluated Candidate JOINs from CSV")
        candidates = pl.read_csv(evaluated_path)
    if add_header:
        logger.info("Loading Candidate JOINs from CSV")
        candidates = pl.read_csv(candidates_path).filter(pl.col('r_col_name') != pl.col('s_col_name'))
        # add the header row to the output CSV file
        with open(evaluated_path, 'w') as file:
            wr = csv.writer(file)
            wr.writerow([
                'r_tab_id'    , 's_tab_id',
                'r_col_id'    , 's_col_id',
                'r_col_name'  , 's_col_name',
                'size_r_col'  , 'size_s_col',
                'r_pkg_id'    , 's_pkg_id',
                
                'size_intersection', 
                'size_union', 
                'jaccard', 
                'overlap',
                f'{model}_score',
                f'{model}_explanation',
                f'{model}_errors'
            ])

    evaluations = []
    default_score, default_explanation = -1, "NO_EXPLANATION"
    score, explanation, errors = default_score, default_explanation, ""
    start_batch_t = time.time()

    logging.info("Started Agent JOINs Evaluation")
    time.sleep(5)
    for i, row in enumerate(candidates.rows()[:UP_TO_ROW], start=1):
        if i % WRITE_BATCH_SIZE == 0:
            logger.info(f'Up to table {i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s')
            with open(evaluated_path, "a") as file:
                wr = csv.writer(file)
                wr.writerows(evaluations)
            evaluations = []
            start_batch_t = time.time()


        # ask the agent for the score
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, r_col_id, s_col_id, r_col_name, s_col_name = row[:6]
            r_rsc_name, _, r_pkg_note = get_resource_metadata(r_tab_id, table_ids, metadata)
            s_rsc_name, _, s_pkg_note = get_resource_metadata(s_tab_id, table_ids, metadata)

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_note = re.sub(r"(\n|\r|\t)", " ", r_pkg_note)[:MAX_LENGTH_NOTES]
            s_pkg_note = re.sub(r"(\n|\r|\t)", " ", s_pkg_note)[:MAX_LENGTH_NOTES]

            # get a small sample from the dataframes
            r_df = pl.read_parquet(f'{tables_path}/{table_ids[r_tab_id]}')
            r_df = r_df.sample(max(N_ROWS_SAMPLE, r_df.shape[0]))

            s_df = pl.read_parquet(f'{tables_path}/{table_ids[s_tab_id]}')
            r_df = s_df.sample(max(N_ROWS_SAMPLE, s_df.shape[0]))
            
            # get the cells that have made the join
            common_cells = list(set(map(sanitize_string, r_df.to_series(r_col_id))) & set(map(sanitize_string, s_df.to_series(s_col_id))))
            common_cells = common_cells[:MAX_COMM_CELLS]
            # response = await runtime.send_message(
            response = await agent.on_messages(
                messages=[
                    TextMessage(content=f"""
                        Define a score given the following information from the two tables: 
                        The columns that joins are {r_col_name=}, {s_col_name=}, 
                        r_table_name={r_rsc_name}, s_table_name={s_rsc_name}, 
                        
                        r_table_description={r_pkg_note}, s_table_description={s_pkg_note}
                        
                        common_cells: {common_cells}
                        
                        r_table_sample:
                        {r_df}, 
                        
                        s_table_sample:
                        {s_df}

                        Give only one overall score between 0 (casual) and 5 (meaningful) and a clear, short and concise explanation.
                        Write the score inside tags <score>SCORE</score>.
                        """,
                        source="user"
                    ), 
                ],                
                cancellation_token=CancellationToken()
            )

            explanation = str(response.chat_message.content).replace('\n', ' ').replace(',', ' ')
            score = int(re.match(r"\<score\>(\d)\<\/score\>", explanation).groups()[0])
            assert 0 <= score <= 5, "Score not in defined range"
        except Exception as e:
            # default values in case of error
            score = default_score
            explanation = default_explanation if explanation == default_explanation else explanation
            errors = str(e).replace('\n', ' ').replace(',', ' ')
        finally:
            # append the evaluation and reset the values (not needed?)
            evaluations.append((*row, score, explanation if save_explanation else "", errors))
            score, explanation, errors = default_score, default_explanation, ""
            
            # reset the agent to the initial state
            await agent.on_reset(cancellation_token=CancellationToken())

    # logger.debug("Stopping Agent Runtime")
    # await runtime.stop()

    logger.info(f"Up to table {i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s")
    with open(evaluated_path, "a") as file:
        wr = csv.writer(file)
        wr.writerows(evaluations)

    logger.info(f"Job Completed.")


if __name__ == '__main__':
    asyncio.run(amain())
    