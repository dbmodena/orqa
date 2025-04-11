import re
import os
import re 
import csv
import sys
import time
import asyncio
import warnings
import logging
from logging.handlers import TimedRotatingFileHandler

import jsonlines
import polars as pl

warnings.filterwarnings('ignore')


from autogen_core import ClosureAgent, ClosureContext, DefaultTopicId, MessageContext, SingleThreadedAgentRuntime, TypeSubscription

from orqa.utils import sanitize_string, get_resource_metadata
from orqa.agents.debate import JoinScoreAggregator, JoinEvaluator
from orqa.agents.utils import Answer, Question, JOIN_EVALUATION_TOPIC_TYPE, get_model_client




async def amain(tag="CAN", from_=0, to_='END'):

    data_path       = f'{os.path.dirname(__file__)}/../data'
    tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
    metadata_path   = f'{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl'
    log_path        = f'{data_path}/log/{tag}/3_join_evaluation_{time.strftime('%y%m%d_%H_%M_%S')}.log'

    candidates_path = f'{data_path}/outputs/{tag}/candidate_joins.csv'
    evaluated_path  = f'{data_path}/outputs/{tag}/evaluated_joins.csv'

    UP_TO_ROW           = 'END'
    WRITE_BATCH_SIZE    = 100

    MAX_N_COLUMNS       = 10

    # to limit the context passed to the LLM-agent (the "notes" field may be very long)
    MAX_LENGTH_NOTES    = 300
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 3

    # number of values in common between the candidate joinable columnes
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    MIN_SCORE           = 0
    MAX_SCORE           = 10
    NUM_NEIGHS          = 2
    MAX_ROUNDS          = 2
    NUM_SOLVERS         = 3

    # the model name (here we will use LiteLLM and Ollama)
    model               = "qwen2.5-7b"


    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f'evaluationLogger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    # log also to stdout
    stdout_hanlder = logging.StreamHandler()
    logger.addHandler(stdout_hanlder)
    
    # define the runtime
    runtime = SingleThreadedAgentRuntime()

    # register three different agents for the debating mechanism
    for solver_id in range(NUM_SOLVERS):
        await JoinEvaluator.register(
            runtime,
            f"JoinEvaluator{solver_id}",
            lambda: JoinEvaluator(
                get_model_client(model),
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
    candidates = pl.read_csv(candidates_path, ignore_errors=True)

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
    
    for i, row in enumerate(candidates.rows()[:UP_TO_ROW if isinstance(UP_TO_ROW, int) else candidates.shape[0]]):
        if i % WRITE_BATCH_SIZE == 0:
            logger.info(f'Up to {i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s')
            with open(evaluated_path, "a") as file:
                wr = csv.writer(file)
                wr.writerows(evaluations)
            evaluations = []
            start_batch_t = time.time()

        # ask the agent for the score
        start_debate_t = end_debate_t = -1
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, r_col_id, s_col_id, original_r_col_name, original_s_col_name, _, _, r_pkg_id, s_pkg_id, _, _, _, _ = row
            _, r_rsc_name, r_pkg_id, _, r_pkg_notes, r_pkg_keywords, r_pkg_tags = get_resource_metadata(r_tab_id, table_ids, metadata)
            _, s_rsc_name, s_pkg_id, _, s_pkg_notes, s_pkg_keywords, s_pkg_tags = get_resource_metadata(s_tab_id, table_ids, metadata)

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_notes = re.sub(r"(\n|\r|\t)", " ", r_pkg_notes)[:MAX_LENGTH_NOTES]
            s_pkg_notes = re.sub(r"(\n|\r|\t)", " ", s_pkg_notes)[:MAX_LENGTH_NOTES]

            r_col_name, s_col_name = sanitize_string(original_r_col_name), sanitize_string(original_s_col_name)
            
            # read the tables 
            r_df = (
                pl
                .scan_parquet(f'{tables_path}/{table_ids[r_tab_id]}')
                .select(
                    pl.all().map_elements(sanitize_string, pl.String)
                )
                .rename(sanitize_string)
                .collect()                
            )

            s_df = (
                pl
                .scan_parquet(f'{tables_path}/{table_ids[s_tab_id]}')
                .select(
                    pl.all().map_elements(sanitize_string, pl.String)
                )
                .rename(sanitize_string)
                .collect()                
            )

            # dtype conversion to int/float
            for column in r_df.columns:
                try:
                    dtype = pl.Float32 if any(',' in str(x) or '.' in str(x) for x in set(r_df.select(column).sample(100, with_replacement=True).to_series())) else pl.Int32
                    r_df = r_df.with_columns(pl.col(column).cast(dtype))
                except:
                    continue

            for column in s_df.columns:
                try:
                    dtype = pl.Float32 if any(',' in str(x) or '.' in str(x) for x in set(s_df.select(column).sample(100, with_replacement=True).to_series())) else pl.Int32
                    s_df = s_df.with_columns(pl.col(column).cast(dtype))
                except:
                    continue
            
            # for the evaluation keep only the first MAX_N_COLUMNS columns 
            r_df = r_df.drop(r_col_name).insert_column(0, r_df.get_column(r_col_name))
            s_df = s_df.drop(s_col_name).insert_column(0, s_df.get_column(s_col_name))
                        
            s_df.get_column(s_col_name)
            common_cells = list(set(r_df.get_column(r_col_name)) & set(s_df.get_column(s_col_name)))
            common_cells = common_cells[:MAX_COMM_CELLS]                        
            
            # pass a formatted version of the tables to the completion client,
            # only a small portion of the tables
            with pl.Config(
                tbl_hide_dataframe_shape=True,
                tbl_width_chars=1000,
                tbl_formatting='MARKDOWN',
                tbl_cols=MAX_N_COLUMNS):
                r_df_str = str(r_df.select(r_df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))
                s_df_str = str(s_df.select(s_df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))

            
            # start/resume the runtime and publish the question message
            runtime.start()
            start_debate_t = time.time()

            await runtime.publish_message(
                Question(
                    content=(
                         "Consider the following information:\n"
                        f"The table {r_rsc_name} is about: {r_pkg_notes}.\n"
                        f"Some keywords and tags about it are: {r_pkg_keywords}, {r_pkg_tags}.\n"
                        f"Example rows:\n{r_df_str}"
                        f"\n{'-' * 50}\n"
                        f"The table {s_rsc_name} is about: {s_pkg_notes}.\n"
                        f"Some keywords and tags about it are: {s_pkg_keywords}, {s_pkg_tags}.\n"
                        f"Example rows: {s_df_str}"
                        f"\n{'-' * 50}\n"
                        f"The columns that joins are {r_col_name=}, {s_col_name=}, "
                        f"and some of the common values in these columns are: {common_cells} "
                        "Define a relationship quality score for the two tables. "
                        "The join columns may not be identical or may not perfectly align. "
                        "Focus on the meaningfulness of the operation between the given tables, "
                        "base your choice on their description and keywords."
                    )                   
                ),
                topic_id=DefaultTopicId()
            )

            await runtime.stop_when_idle()
            end_debate_t = time.time()
            while not queue.empty():
                score = await queue.get()

        except Exception as e:
            logger.error(f"Exception: {e}, {table_ids[r_tab_id]}, {table_ids[s_tab_id]}")
            score = -1

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

    logger.info(f"{i}({round(i * 100 / len(candidates), 3)}%);time:{round(time.time() - start_batch_t, 3)}s")
    with open(evaluated_path, "a") as file:
        wr = csv.writer(file)
        wr.writerows(evaluations)

    logger.info(f"Done.")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
    