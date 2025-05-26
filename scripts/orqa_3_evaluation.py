import os
import csv
import sys
import yaml
import time
import asyncio
import logging
import argparse
import warnings

from os.path import join as pjoin

import jsonlines
import polars as pl

from autogen_core import ClosureAgent, ClosureContext, DefaultTopicId, MessageContext, SingleThreadedAgentRuntime, TypeSubscription

from orqa.utils import get_all_data, setup_logger
from orqa.agents.debate import ScoreAggregator, Evaluator
from orqa.agents.utils import Answer, Question, EVALUATION_TOPIC_TYPE, get_model_client

warnings.filterwarnings('ignore')


async def amain(tag: str = "CAN", 
                from_: int = 0, 
                to_: int = 'END'):

    conf_path       = pjoin(os.path.dirname(__file__), '..', 'conf', 'configuration.yml')
    data_path       = pjoin(os.path.dirname(__file__), '..', 'data')
    tables_path     = pjoin(data_path, 'datasets', tag, 'tables', f'from{from_}_to{to_}')
    metadata_path   = pjoin(data_path, 'datasets', tag, 'metadata', f'from{from_}_to{to_}.jsonl')
    log_path        = pjoin(data_path, 'log', tag, f'3_evaluation_{time.strftime("%y%m%d_%H_%M_%S")}.log')
    candidates_path = pjoin(data_path, 'outputs', tag, f'from{from_}_to{to_}', 'candidates.csv')
    evaluated_path  = pjoin(data_path, 'outputs', tag, f'from{from_}_to{to_}', 'evaluated.csv')

    with open(conf_path, 'r') as file:
        raw = file.read()
        cfg = argparse.Namespace(**{**yaml.safe_load(raw)['general'], **yaml.safe_load(raw)['generation']})

    MAX_N_COLUMNS       = cfg.max_columns_number
    MAX_LENGTH_NOTES    = cfg.max_notes_length
    N_ROWS_SAMPLE       = cfg.rows_for_sampling

    CLEAN_HEADERS       = cfg.string_cleaning['headers']
    CLEAN_ELEMENTS      = cfg.string_cleaning['elements']

    BUDGET              = cfg.budget
    WRITE_BATCH_SIZE    = cfg.write_batch_size

    MIN_SCORE           = cfg.min_score
    MAX_SCORE           = cfg.max_score

    NUM_SOLVERS         = cfg.num_solvers
    NUM_NEIGHS          = cfg.neighbors_per_solver
    MAX_ROUNDS          = cfg.max_rounds

    model               = cfg.solver_model

    # set up the logging
    logger = setup_logger(log_path, "evaluation_logger", on_file=True, on_stdout=True)
    
    start_t = time.time()

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
    
    for i, row in enumerate(candidates.rows()[:BUDGET if isinstance(BUDGET, int) else candidates.shape[0]]):
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
                        "Focus on the meaningfulness of a potential operation between them."
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
        wr = csv.writer(file, lineterminator='\n')
        wr.writerows(evaluations)

    logger.info(f"Done.")
    
    end_t = time.time()
    total_t = round(end_t - start_t, 3)
    logger.info(f"Total time: {total_t}s")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
    