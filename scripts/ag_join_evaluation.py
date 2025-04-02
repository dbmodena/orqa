import re
import os
import re 
import csv
import sys
import time
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler

import jsonlines
import polars as pl

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken, ClosureAgent, ClosureContext, DefaultTopicId, MessageContext, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.models import ModelFamily

from autogen_ext.models.openai import OpenAIChatCompletionClient

from orqa.agents.utils import Answer, Question
from orqa.utils import sanitize_string
from orqa.agents.debate import JOIN_EVALUATION_TOPIC_TYPE, JoinScoreAggregator, JoinEvaluator, final_join_evaluation_topic_id


def get_package_id(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    return metadata[rsc_id]['id']
    

def get_resource_metadata(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    md = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    return md['name'], metadata[rsc_id]['title'], metadata[rsc_id]['notes']


async def amain(tag="CAN", from_=0, to_=10_000):

    data_path       = f'{os.path.dirname(__file__)}/../data'
    tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
    metadata_path   = f'{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl'
    log_path        = f'{data_path}/log/{tag}_JoinEvaluation.log'

    candidates_path = f'{data_path}/outputs/{tag}_candidate_joins.csv'
    evaluated_path  = f'{data_path}/outputs/{tag}_evaluated_joins.csv'

    add_header          = True
    save_explanation    = True

    UP_TO_ROW           = 1000
    WRITE_BATCH_SIZE    = 10

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 500
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 5

    # number of values in common between the candidate joinable columnes
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    MIN_SCORE           = 0
    MAX_SCORE           = 5
    NUM_NEIGHS          = 2
    MAX_ROUNDS          = 3
    NUM_SOLVERS         = 3

    # the model name (here we will use LiteLLM and Ollama)
    model               = "ollama/qwen2.5:3b"


    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f'evaluationLogger')
    logger.setLevel(logging.DEBUG)
    handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=3)
    handler.suffix = "%y-%m-%d_%H:%M:%S.log"
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    
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
    }

    model_client = OpenAIChatCompletionClient(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info,
    )

    
    # define the runtime
    runtime = SingleThreadedAgentRuntime()

    # register three different agents for the debating mechanism
    await JoinEvaluator.register(
        runtime,
        "JoinEvaluatorA",
        lambda: JoinEvaluator(
            model_client,
            NUM_NEIGHS,
            MAX_ROUNDS,
            MIN_SCORE, MAX_SCORE, logger,
            topic_type="JoinEvaluatorA"
        )
    )

    await JoinEvaluator.register(
        runtime,
        "JoinEvaluatorB",
        lambda: JoinEvaluator(
            model_client,
            NUM_NEIGHS,
            MAX_ROUNDS,
            MIN_SCORE, MAX_SCORE, logger,
            topic_type="JoinEvaluatorB"
        )
    )

    await JoinEvaluator.register(
        runtime,
        "JoinEvaluatorC",
        lambda: JoinEvaluator(
            model_client,
            NUM_NEIGHS,
            MAX_ROUNDS,
            MIN_SCORE, MAX_SCORE, logger,
            topic_type="JoinEvaluatorC"
        )
    )

    # register also the final score aggregator
    await JoinScoreAggregator.register(
        runtime, 
        "JoinScoreAggregator", 
        lambda: JoinScoreAggregator(NUM_SOLVERS, MIN_SCORE, MAX_SCORE, logger))

    # now, every agent should subscribe to the right topic
    # for the agent names with "A"
    await runtime.add_subscription(TypeSubscription("JoinEvaluatorA", "JoinEvaluatorB"))
    await runtime.add_subscription(TypeSubscription("JoinEvaluatorA", "JoinEvaluatorC"))

    # for the agent names with "B"
    await runtime.add_subscription(TypeSubscription("JoinEvaluatorB", "JoinEvaluatorA"))
    await runtime.add_subscription(TypeSubscription("JoinEvaluatorB", "JoinEvaluatorC"))

    # and for the agent names with "C"
    await runtime.add_subscription(TypeSubscription("JoinEvaluatorC", "JoinEvaluatorA"))
    await runtime.add_subscription(TypeSubscription("JoinEvaluatorC", "JoinEvaluatorB"))


    # setup the mechanism to collect the final answers
    queue = asyncio.Queue[Answer]()

    async def collect_result(_agent: ClosureContext, message: Answer, ctx: MessageContext) -> None:
        await queue.put(message)

    # now start the runtime (don't know if it's actually necessary 
    # to start it before closure registration)
    runtime.start()

    CLOSURE_AGENT_TYPE = "collect_result_agent"
    await ClosureAgent.register_closure(
        runtime,
        CLOSURE_AGENT_TYPE,
        collect_result,
        subscriptions=lambda: [TypeSubscription(topic_type=JOIN_EVALUATION_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)]
    )



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
                'score'
            ])

    evaluations = []
    score = default_score = -1
    start_batch_t = time.time()

    logging.info("Started Agent JOINs Evaluation")
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

            r_df = pl.read_parquet(f'{tables_path}/{table_ids[r_tab_id]}')
            s_df = pl.read_parquet(f'{tables_path}/{table_ids[s_tab_id]}')
            
            # get the cells that have made the join
            common_cells = list(set(map(sanitize_string, r_df.to_series(r_col_id))) & set(map(sanitize_string, s_df.to_series(s_col_id))))
            assert len(common_cells) == row[10]
            common_cells = common_cells[:MAX_COMM_CELLS]                        

            # get a small sample from the dataframes
            r_df = r_df.sample(max(N_ROWS_SAMPLE, r_df.shape[0]))
            s_df = s_df.sample(max(N_ROWS_SAMPLE, s_df.shape[0]))
            
            await runtime.publish_message(
                Question(
                    content=f"""
                        Given the following information about two tables: 
                        
                        r_table_name: {r_rsc_name}, 
                        
                        r_table_description={r_pkg_note}, 
                        
                        r_table_sample:
                        {r_df}, 
                        
                        #########################################

                        s_table_name={s_rsc_name}, 
                        
                        s_table_description={s_pkg_note}
                        
                        s_table_sample:
                        {s_df}

                        ##########################################
                        
                        The columns that joins are {r_col_name=}, {s_col_name=},

                        Cells in common in the joining columns: {common_cells}
                        """,                        
                ),
                topic_id=DefaultTopicId()
            )

            await runtime.stop_when_idle()
            while not queue.empty():
                answer = await queue.get()
            score = answer.score
        except Exception as e:
            score = e
        evaluations.append(score)

    logger.info(f"Up to table {i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s")
    with open(evaluated_path, "a") as file:
        wr = csv.writer(file)
        wr.writerows(evaluations)

    logger.info(f"Done.")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
    