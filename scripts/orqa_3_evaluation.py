import os
import csv
import sys
import time
import asyncio
import warnings
import logging

import jsonlines
import polars as pl

from autogen_core import ClosureAgent, ClosureContext, DefaultTopicId, MessageContext, SingleThreadedAgentRuntime, TypeSubscription

from orqa.utils import get_all_data
from orqa.agents.debate import ScoreAggregator, Evaluator
from orqa.agents.utils import Answer, Question, EVALUATION_TOPIC_TYPE, get_model_client

warnings.filterwarnings('ignore')


async def amain(tag: str = "CAN", 
                from_: int = 0, 
                to_: int = 'END'):

    data_path       = f'{os.path.dirname(__file__)}/../data'
    tables_path     = f'{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}'
    metadata_path   = f'{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl'
    log_path        = f'{data_path}/log/{tag}/3_evaluation_{time.strftime('%y%m%d_%H_%M_%S')}.log'

    candidates_path = f'{data_path}/outputs/{tag}/candidates_test.csv'
    evaluated_path  = f'{data_path}/outputs/{tag}/evaluated_test.csv'

    # how many rows are evaluated (if 'END', all)
    UP_TO_ROW           = 10

    WRITE_BATCH_SIZE    = 1

    # how many columns are kept for the evaluation
    MAX_N_COLUMNS       = 20

    # to limit the context passed to the LLM-agent (the "notes" field may be very long)
    MAX_LENGTH_NOTES    = 500
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 5

    # string cleaning strategies
    CLEAN_HEADERS       = "complex"
    CLEAN_ELEMENTS      = None

    # boundaries for the score value
    MIN_SCORE           = 0
    MAX_SCORE           = 10

    # how many solvers take part
    # to the debate
    NUM_SOLVERS         = 3
    
    # how many neighboors each 
    # solver has
    NUM_NEIGHS          = 2

    MAX_ROUNDS          = 2

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

    # keep only last three log files relative to this part
    old_dirs =  sorted([d for d in os.listdir(os.path.dirname(log_path)) if d.startswith('3_evaluation')], reverse=True)
    logs_to_delete = old_dirs[3:] if len(old_dirs) > 3 else []
    for log_to_delete in logs_to_delete:
        os.remove(os.path.join(os.path.dirname(log_path), log_to_delete))
    
    # define the runtime
    runtime = SingleThreadedAgentRuntime()

    # register three different agents for the debating mechanism
    for solver_id in range(NUM_SOLVERS):
        await Evaluator.register(
            runtime,
            f"Evaluator{solver_id}",
            lambda: Evaluator(
                get_model_client(model),
                f"Evaluator{solver_id}",
                NUM_NEIGHS,
                MAX_ROUNDS,
                MIN_SCORE, MAX_SCORE, 
                logger
            )
        )
        
        # now, every agent should subscribe to the right topic(s)
        await runtime.add_subscription(TypeSubscription(f"Evaluator{solver_id}", f"Evaluator{solver_id - 1 % NUM_SOLVERS}"))
        await runtime.add_subscription(TypeSubscription(f"Evaluator{solver_id}", f"Evaluator{solver_id + 1 % NUM_SOLVERS}"))

    # register also the final score aggregator
    await ScoreAggregator.register(
        runtime, 
        "ScoreAggregator", 
        lambda: ScoreAggregator(NUM_SOLVERS, MIN_SCORE, MAX_SCORE, logger))

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
        subscriptions=lambda: [TypeSubscription(topic_type=EVALUATION_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)]
    )

    await runtime.stop_when_idle()

    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    logger.info("Loading Candidates from CSV")
    candidates = pl.read_csv(candidates_path, ignore_errors=True, truncate_ragged_lines=True).drop_nulls()

    # add the header row to the output CSV file
    with open(evaluated_path, 'w') as file:
        wr = csv.writer(file)
        wr.writerow([
            'r_rsc_id', 
            's_rsc_id',
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

    logger.info("Started Agent Candidates Evaluation")
    
    for i, row in enumerate(candidates.rows()[:UP_TO_ROW if isinstance(UP_TO_ROW, int) else candidates.shape[0]]):
        if i % WRITE_BATCH_SIZE == 0 and i > 0:
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
            r_rsc_id, s_rsc_id, r_col_id, s_col_id, original_r_col_name, original_s_col_name, _, _, r_pkg_id, s_pkg_id, _, _, _, _ = row

            (
                r_rsc_id, r_rsc_name, r_pkg_name, r_pkg_notes, r_pkg_keywords, r_pkg_tags, r_org_name, r_org_title, r_org_desc, r_jur,
                r_col_name, _, r_df_str, _, _
            ) = get_all_data(r_rsc_id, tables_path, metadata, original_r_col_name, MAX_LENGTH_NOTES, MAX_N_COLUMNS, N_ROWS_SAMPLE, 'R', CLEAN_HEADERS, CLEAN_ELEMENTS)


            (
                s_rsc_id, s_rsc_name, s_pkg_name, s_pkg_notes, s_pkg_keywords, s_pkg_tags, s_org_name, s_org_title, s_org_desc, s_jur,
                s_col_name, _, s_df_str, _, _
            ) = get_all_data(s_rsc_id, tables_path, metadata, original_s_col_name, MAX_LENGTH_NOTES, MAX_N_COLUMNS, N_ROWS_SAMPLE, 'S', CLEAN_HEADERS, CLEAN_ELEMENTS)
            
            
            # start/resume the runtime and publish the question message
            runtime.start()
            start_debate_t = time.time()

            await runtime.publish_message(
                Question(
                    content=(
                        "Consider the following information:\n"
                        f"The table '{r_rsc_name}' belongs to the package '{r_pkg_name}'. "
                        f"This package is published by the organisaztion '{r_org_name}', titled as '{r_org_title}' that is about '{r_org_desc}', under the jurisdiction '{r_jur}'. "
                        f"The table description is: {r_pkg_notes}.\n"
                        f"Keywords and tags about it are: {r_pkg_keywords}, {r_pkg_tags}.\n"
                        f"Example rows with schema:\n{r_df_str}"
                        f"\n{'-' * 50}\n"
                        f"The table '{s_rsc_name}' belongs to the package '{s_pkg_name}'. "
                        f"This package is published by the organisaztion '{s_org_name}', titled as '{s_org_title}' that is about {s_org_desc}, under the jurisdiction '{s_jur}'. "
                        f"The table description is: {s_pkg_notes}.\n"
                        f"Keywords and tags about it are: {s_pkg_keywords}, {s_pkg_tags}.\n"
                        f"Example rows: {s_df_str}"
                        f"\n{'-' * 50}\n"
                        "Define a relationship quality score for the two tables. "
                        "Focus on the meaningfulness of a potential operation between the given tables. "
                    )                   
                ),
                topic_id=DefaultTopicId()
            )

            await runtime.stop_when_idle()
            end_debate_t = time.time()
            while not queue.empty():
                score = await queue.get()

        except Exception as e:
            logger.error(f"Exception: {e}, {r_rsc_id}, {s_rsc_id}")
            score = -1

        evaluations.append(
            [  
                r_rsc_id, 
                s_rsc_id,
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
    