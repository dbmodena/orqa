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
os.environ["POLARS_MAX_THREADS"] = "2"
import polars as pl

from autogen_core import ClosureAgent, ClosureContext, DefaultTopicId, MessageContext, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient

from orqa.utils import sanitize_string
from orqa.agents.debate import JoinScoreAggregator, JoinEvaluator
from orqa.agents.utils import Answer, Question, JOIN_EVALUATION_TOPIC_TYPE


def get_resource_metadata(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    rsc = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    
    # get metadata and tags if present
    pkg_keywords = []
    if 'keywords' in metadata[rsc_id] and 'en' in metadata[rsc_id]['keywords']:
        pkg_keywords = metadata[rsc_id]['keywords']['en']

    pkg_tags = []
    if 'tags' in metadata[rsc_id]:
        pkg_tags = metadata[rsc_id]['tags']

    pkg_id = metadata[rsc_id]['id']
    pkg_title = metadata[rsc_id]['title']
    pkg_notes = metadata[rsc_id]['notes']
    rsc_name = rsc['name']

    return rsc_id, rsc_name, pkg_id, pkg_title, pkg_notes, pkg_keywords, pkg_tags



async def amain(tag="CAN", from_=0, to_=10_000):

    data_path       = f'{os.path.dirname(__file__)}/../data'
    tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
    metadata_path   = f'{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl'
    log_path        = f'{data_path}/log/{tag}/JoinEvaluation.log'

    candidates_path = f'{data_path}/outputs/{tag}/candidate_joins.csv'
    evaluated_path  = f'{data_path}/outputs/{tag}/evaluated_joins.csv'

    add_header          = True
    
    UP_TO_ROW           = 10_000
    WRITE_BATCH_SIZE    = 100

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 500
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 5

    # number of values in common between the candidate joinable columnes
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    MIN_SCORE           = 0
    MAX_SCORE           = 10
    NUM_NEIGHS          = 2
    MAX_ROUNDS          = 3
    NUM_SOLVERS         = 3

    # the model name (here we will use LiteLLM and Ollama)
    model               = "ollama/qwen2.5:7b"


    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f'evaluationLogger')
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=3)
    handler.suffix = "%y-%m-%d_%H:%M:%S.log"
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    # log also to stdout
    # stdout_hanlder = logging.StreamHandler()
    # logger.addHandler(stdout_hanlder)

    
    base_url: str = "http://localhost:4000"
    api_key: str = "NotRequiredSinceWeAreLocal"
    temperature: int = 0
    model_info: dict = {
        "json_output"       : False,
        "vision"            : False,
        "function_calling"  : True,
        "family"            : ModelFamily.UNKNOWN,
        "keep_alive"        : "6h", # to keep the model in memory more time
        "num_ctx"           : 8192, # to increase the context size (not sure)
        "keepalive"         : "1h"
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
    for solver_id in range(NUM_SOLVERS):
        await JoinEvaluator.register(
            runtime,
            f"JoinEvaluator{solver_id}",
            lambda: JoinEvaluator(
                model_client,
                f"JoinEvaluator{solver_id}",
                NUM_NEIGHS,
                MAX_ROUNDS,
                MIN_SCORE, MAX_SCORE, 
                logger
            )
        )
        
        # now, every agent should subscribe to the right topic(s)
        await runtime.add_subscription(TypeSubscription(f"JoinEvaluator{solver_id}", f"JoinEvaluator{solver_id - 1 % NUM_SOLVERS}"))
        await runtime.add_subscription(TypeSubscription(f"JoinEvaluator{solver_id}", f"JoinEvaluator{solver_id + 1 % NUM_SOLVERS}"))

    # register also the final score aggregator
    await JoinScoreAggregator.register(
        runtime, 
        "JoinScoreAggregator", 
        lambda: JoinScoreAggregator(NUM_SOLVERS, MIN_SCORE, MAX_SCORE, logger))

    # setup the mechanism to collect the final answers
    queue = asyncio.Queue[Answer]()

    async def collect_result(_agent: ClosureContext, message: Answer, ctx: MessageContext) -> None:
        await queue.put(message.score)

    runtime.start()

    CLOSURE_AGENT_TYPE = "collect_result_agent"
    await ClosureAgent.register_closure(
        runtime,
        CLOSURE_AGENT_TYPE,
        collect_result,
        subscriptions=lambda: [TypeSubscription(topic_type=JOIN_EVALUATION_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)]
    )

    await runtime.stop_when_idle()

    logger.info("Reading Table IDs")
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    logger.info("Loading Candidate JOINs from CSV")
    candidates = pl.read_csv(candidates_path)
    # add the header row to the output CSV file
    with open(evaluated_path, 'w') as file:
        wr = csv.writer(file)
        wr.writerow([
            'r_tab_id', 
            's_tab_id',
            'r_col_id', 
            's_col_id',
            'r_col_name', 
            's_col_name',
            'r_pkg_id', 
            's_pkg_id',                
            'score',
            'time(s)'
        ])

    evaluations = []
    score = -1
    start_batch_t = time.time()

    logger.info("Started Agent JOINs Evaluation")
    for i, row in enumerate(candidates.rows()[:UP_TO_ROW], start=1):
        if i % WRITE_BATCH_SIZE == 0:
            logger.info(f'Up to table {i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s')
            with open(evaluated_path, "a") as file:
                wr = csv.writer(file)
                wr.writerows(evaluations)
            evaluations = []
            start_batch_t = time.time()

        # ask the agent for the score
        start_debate_t = time.time()
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, r_col_id, s_col_id, r_col_name, s_col_name = row[:6]
            _, r_rsc_name, r_pkg_id, r_pkg_title, r_pkg_notes, r_pkg_keywords, r_pkg_tags = get_resource_metadata(r_tab_id, table_ids, metadata)
            _, s_rsc_name, s_pkg_id, s_pkg_title, s_pkg_notes, s_pkg_keywords, s_pkg_tags = get_resource_metadata(s_tab_id, table_ids, metadata)

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_notes = re.sub(r"(\n|\r|\t)", " ", r_pkg_notes)[:MAX_LENGTH_NOTES]
            s_pkg_notes = re.sub(r"(\n|\r|\t)", " ", s_pkg_notes)[:MAX_LENGTH_NOTES]

            r_df = pl.read_parquet(f'{tables_path}/{table_ids[r_tab_id]}')
            s_df = pl.read_parquet(f'{tables_path}/{table_ids[s_tab_id]}')
            
            # get the cells that have made the join
            common_cells = list(set(map(sanitize_string, r_df.to_series(r_col_id))) & set(map(sanitize_string, s_df.to_series(s_col_id))))
            assert len(common_cells) == row[10]
            common_cells = common_cells[:MAX_COMM_CELLS]                        

            # get a small sample from the dataframes
            r_df = r_df.sample(N_ROWS_SAMPLE)
            s_df = s_df.sample(N_ROWS_SAMPLE)
            
            # start/resume the runtime and publish the question message
            runtime.start()
            
            await runtime.publish_message(
                Question(
                    content=f"""
                        Given the following information about two tables: 

                        R table name: {r_rsc_name}, 
                        
                        some notes about the S table are:
                        {r_pkg_notes}, 

                        and the R table is related to the keywords and tags:
                        {r_pkg_keywords}, 
                        {r_pkg_tags}, 
                        
                        a sample of R table rows:
                        {r_df}, 
                        
                        #########################################

                        S table name: {s_rsc_name}, 
                        
                        some notes about the S table are:
                        {s_pkg_notes}
                        
                        and the s table is related to the keywords and tags:
                        {s_pkg_keywords}, 
                        {s_pkg_tags}, 
                        
                        a sample of S table rows:
                        {s_df}

                        ##########################################
                        
                        The columns that joins are {r_col_name=}, {s_col_name=}.

                        Example of common cells in the joining columns: {common_cells}.

                        Define a valid quality score. 
                        The columns may not be identical or may not perfectly align. 
                        Focus on the meaningfulness of the operation between the given tables.
                        """,                        
                ),
                topic_id=DefaultTopicId()
            )

            await runtime.stop_when_idle()
            end_debate_t = time.time()
            while not queue.empty():
                score = await queue.get()
            logger.debug(f"Evaluation get: score={score}")            
        except Exception as e:
            score = e
            logger.error(f"Exception: {e}")

        evaluations.append(
            [  
                r_tab_id, 
                s_tab_id,
                r_col_id, 
                s_col_id,
                r_col_name, 
                s_col_name,
                r_pkg_id,
                s_pkg_id,
                score,
                round(end_debate_t - start_debate_t, 3)
            ]
        )

    logger.info(f"Up to table {i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s")
    with open(evaluated_path, "a") as file:
        wr = csv.writer(file)
        wr.writerows(evaluations)

    logger.info(f"Done.")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
    