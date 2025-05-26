import os
import re
import shutil
import sys
import yaml
import time
import asyncio
import logging
import warnings
import argparse

from typing import List
from typing_extensions import Annotated
from os.path import join as pjoin

import duckdb
import jsonlines
import polars as pl

from autogen_core.tools import FunctionTool, Tool
from autogen_core import ClosureAgent, ClosureContext, MessageContext, SingleThreadedAgentRuntime, TypeSubscription

from orqa.agents.utils import *
from orqa.agents.reviewer import ReviewerAgent
from orqa.agents.question_generator import NLQuestionGeneratorAgent, SQLQueryGeneratorAgent
from orqa.utils import get_all_data, setup_logger


warnings.filterwarnings("ignore")


async def verify_sql(sql_query: Annotated[str, "A SQL query which represents the natural language question."]):
    error = results  = None
    
    try:
        R: pl.DataFrame = globals()['R']
        if 'S' in globals():
            S: pl.DataFrame = globals()['S']
        
        # duckdb can use directly the dataframe
        # without disk operations
        results = duckdb.sql(sql_query).fetchmany(size=1)
        is_empty = len(results) == 0
        results = str(results)
    except Exception as e:
        error = str(e)
        logging.getLogger("agent_logger").debug(f'{sql_query=}\n{error=}')
    
    if error:
        return {
            "status": "error",
            "error_description": error,
            "sql_query": sql_query,
            "result": results if results else "N/A"            
        }
    elif is_empty:
        return {
            "status": "error",
            "error_description": "Empty result set",
            "sql_query": sql_query,
            "result": "empty result"
        }
    else:
        return {
            "status": "success", 
            "error_description": "No error",
            "sql_query": sql_query,
            "result": results
        }


async def amain(tag: str = "CAN",
                from_: int = 0,
                to_: int = "END"):

    conf_path       = pjoin(os.path.dirname(__file__), '..', 'conf', 'configuration.yml')
    duckdb_tmp_path = pjoin(os.path.dirname(__file__), '..', '.tmp')
    data_path       = pjoin(os.path.dirname(__file__), '..', 'data')
    tables_path     = pjoin(data_path, 'datasets', tag, 'tables', f'from{from_}_to{to_}')
    metadata_path   = pjoin(data_path, 'datasets', tag, 'metadata', f'from{from_}_to{to_}.jsonl')
    log_path        = pjoin(data_path, 'log', tag, f'4_generation_{time.strftime("%y%m%d_%H_%M_%S")}.log')    
    evaluated_path  = pjoin(data_path, 'outputs', tag, f'from{from_}_to{to_}', 'evaluated.csv')
    queries_path    = pjoin(data_path, 'outputs', tag, f'from{from_}_to{to_}', 'queries.csv')

    with open(conf_path, 'r') as file:
        raw = file.read()
        cfg = argparse.Namespace(**{**yaml.safe_load(raw)['general'], **yaml.safe_load(raw)['generation']})

    MAX_N_COLUMNS       = cfg.max_columns_number
    MAX_LENGTH_NOTES    = cfg.max_notes_length
    N_ROWS_SAMPLE       = cfg.rows_for_sampling

    CLEAN_HEADERS       = cfg.string_cleaning['headers']
    CLEAN_ELEMENTS      = cfg.string_cleaning['elements']

    BUDGET              = cfg.budget
    
    MAX_REVIEWS         = cfg.max_reviews
    MIN_SCORE           = cfg.min_score

    DO_SINGLE_TABLE     = 'single' in cfg.types
    DO_JOIN             = 'join' in cfg.types
    DO_UNION            = 'union' in cfg.types

    sql_gen_model       = cfg.models["sql_gen"]
    sql_rev_model       = cfg.models["sql_rev"]
    nl_gen_model        = cfg.models["nl_gen"]
    nl_rev_model        = cfg.models["nl_rev"]

    levels              = cfg.levels

    # set up the logging
    logger = setup_logger(log_path, "generation_logger", on_file=True, on_stdout=True)
    
    # load the table IDs 
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))
    
    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    logger.info("Loading evaluated JOIN pairs")
    joins = (
        pl
        .read_csv(evaluated_path)
        .unique(subset=['r_tab_id', 's_tab_id'], maintain_order=True)
        .drop_nans(subset=["score"])
        .filter(pl.col('score') >= MIN_SCORE)
        .select(pl.all().shuffle(seed=130399))
    )
    
    logger.info(f"Evaluated pairs with score >= {MIN_SCORE}: {joins.shape[0]}")

    # create the tools used by the agent
    tools: List[Tool] = [
        FunctionTool(
            verify_sql, 
            name='verify_sql',
            description="A tool that checks if the proposed SQL query works."
        )
    ]

    start_t = time.time()

    # create the agent runtime
    runtime = SingleThreadedAgentRuntime()

    # create the Natural Language/SQL Query Generator Agents
    # We use LiteLLM, so ollama and other models should not 
    # give problems with this client type anyway 
    
    await SQLQueryGeneratorAgent.register(
        runtime, "sql_generator_agent",
        lambda: SQLQueryGeneratorAgent(get_model_client(sql_gen_model), tools, MAX_REVIEWS, logger)
    )

    await NLQuestionGeneratorAgent.register(
        runtime, "nl_generator_agent",
        lambda: NLQuestionGeneratorAgent(get_model_client(nl_gen_model), MAX_REVIEWS, logger)
    )

    await ReviewerAgent.register(
        runtime, "sql_reviewer_agent",
        lambda: ReviewerAgent(get_model_client(sql_rev_model), MAX_REVIEWS, logger)
    )
    
    await ReviewerAgent.register(
        runtime, "nl_reviewer_agent",
        lambda: ReviewerAgent(get_model_client(nl_rev_model), MAX_REVIEWS, logger)
    )

    # add subscriptions for pub/sub communications
    await runtime.add_subscription(TypeSubscription(SQL_GENERATION_TOPIC_TYPE   , "sql_generator_agent"))
    await runtime.add_subscription(TypeSubscription(SQL_INTERMEDIATE_TOPIC_TYPE , "sql_generator_agent"))
    
    await runtime.add_subscription(TypeSubscription(NL_GENERATION_TOPIC_TYPE    , "nl_generator_agent"))
    await runtime.add_subscription(TypeSubscription(NL_INTERMEDIATE_TOPIC_TYPE  , "nl_generator_agent"))
    
    await runtime.add_subscription(TypeSubscription(SQL_INTERMEDIATE_TOPIC_TYPE , "sql_reviewer_agent"))
    await runtime.add_subscription(TypeSubscription(NL_INTERMEDIATE_TOPIC_TYPE  , "nl_reviewer_agent"))

    
    # setup the mechanism to collect the final answers
    # the closure agent must be registered to the output topic 
    # in order to get the results

    runtime.start()

    queue = asyncio.Queue[SQLGenerationResult | NLGenerationResult]()    

    async def collect_result(_agent: ClosureContext, 
                             message: SQLGenerationResult | NLGenerationResult, 
                             ctx: MessageContext) -> None:
        if isinstance(message, SQLGenerationResult):
            await queue.put((message.sql_query.replace('\n', ' '), message.n_rev, message.sql_success.replace('\n', ' ').strip(), message.input_tokens, message.output_tokens, message.review.replace('\n', ' ').strip()))
        elif isinstance(message, NLGenerationResult):
            await queue.put((message.nl_question.replace('\n', ' '), message.n_rev, message.input_tokens, message.output_tokens, message.review.replace('\n', ' ').strip()))

    CLOSURE_AGENT_TYPE = "collect_result_agent"
    await ClosureAgent.register_closure(
        runtime,
        CLOSURE_AGENT_TYPE,
        collect_result,
        subscriptions=lambda: [
            TypeSubscription(topic_type=NL_RESULT_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE),
            TypeSubscription(topic_type=SQL_RESULT_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)
        ]
    )

    await runtime.stop_when_idle()

    # start processing messages and create one agent
    logger.info("Started Queries Generation")
    
    already_seen_tables = set()
    results = []
    start_t = time.time()

    prev_sql, prev_nl = [], []

    n_id = 0
    start = 0
    rows = joins.rows()[start:BUDGET if isinstance(BUDGET, int) else joins.shape[0]]
    for i, row in enumerate(rows, start=start):
        start_step_t = time.time()
        try:
            r_rsc_id, s_rsc_id, _, _, original_r_col_name, original_s_col_name, r_pkg_id, s_pkg_id, _, _ = row
            
            r_rsc_id, s_rsc_id = table_ids[r_rsc_id], table_ids[s_rsc_id]

            # for simplicity, because these could lead to some naming complications
            # TODO handle naming with multi-table resources
            if '_' in r_rsc_id or '_' in s_rsc_id:
                continue

            (
                r_rsc_id, r_rsc_name, r_pkg_name, r_pkg_notes, r_pkg_keywords, r_pkg_tags, r_org_name, r_org_title, r_org_desc, r_jur,
                r_col_name, r_df, r_df_str, r_sql_schema, r_columns_dtypes
            ) = get_all_data(r_rsc_id, tables_path, metadata, original_r_col_name, MAX_LENGTH_NOTES, MAX_N_COLUMNS, N_ROWS_SAMPLE, 'R', CLEAN_HEADERS, CLEAN_ELEMENTS)

            (
                s_rsc_id, s_rsc_name, s_pkg_name, s_pkg_notes, s_pkg_keywords, s_pkg_tags, s_org_name, s_org_title, s_org_desc, s_jur,
                s_col_name, s_df, s_df_str, s_sql_schema, s_columns_dtypes
            ) = get_all_data(s_rsc_id, tables_path, metadata, original_s_col_name, MAX_LENGTH_NOTES, MAX_N_COLUMNS, N_ROWS_SAMPLE, 'S', CLEAN_HEADERS, CLEAN_ELEMENTS)            

        except Exception as e:
            logger.error(f"Error in query preparation: >>>{e}<<< ")
            raise e

        #############################################
        ###### Generate Single-Table queries ########
        #############################################
        if DO_SINGLE_TABLE:
            for rsc_id, pkg_id, pkg_name, rsc_name, df, df_str, sql_schema, notes, keywords, tags, org_name, org_title, org_desc, jur in [
                (r_rsc_id, r_pkg_id, r_pkg_name, r_rsc_name, r_df, r_df_str, r_sql_schema, r_pkg_notes, r_pkg_keywords, r_pkg_tags, r_org_name, r_org_title, r_org_desc, r_jur),
                (s_rsc_id, s_pkg_id, s_pkg_name, s_rsc_name, s_df, s_df_str, s_sql_schema, s_pkg_notes, s_pkg_keywords, s_pkg_tags, s_org_name, s_org_title, s_org_desc, s_jur)]:
                
                if rsc_id in already_seen_tables:
                    continue

                already_seen_tables.add(rsc_id)

                # bring the dataframe in the global scope 
                # to be used in the tool
                globals()['R'] = df                

                # reset previous results
                prev_sql.clear() 
                prev_nl.clear()

                for nq, difficulty in enumerate(levels):            
                    logger.info(f"{i=}, step {nq} - SINGLE TABLE")
                    
                    # reset to default values
                    sql = nl = "ERROR"
                    sql_time = nl_time = -1
                    sql_n_rev = nl_n_rev = -1
                    sql_intok = sql_outok = -1
                    nl_intok = nl_outok = -1
                    sql_success = False
                    sql_review = nl_review = "NOREVIEW"

                    try:
                        logger.debug("Generating SQL query")
                        runtime.start()
                        sql_start_t = time.time()
                        await runtime.publish_message(                    
                            SQLGenerationTask(sql_task=(
                                "Given the following information:\n"                        
                                "Use 'R' to indicate the given table. "
                                f'Its schema is: {sql_schema}\n'
                                f"Example rows of the table:\n{df_str}"
                                f"\n{'-' * 50}\n"
                                f"Generate a {difficulty} SQL query based on the given table. "
                                f"The new query must be different from previous queries: {prev_sql}. "                                
                                )
                            ),
                            topic_id=sql_generation_topic_id
                        )
                        
                        # wait the agent response
                        await runtime.stop_when_idle()
                        sql_time = round(time.time() - sql_start_t, 3)

                        logger.debug("Idle state after SQL generation.")
                        
                        while not queue.empty():
                            sql, sql_n_rev, sql_success, sql_intok, sql_outok, sql_review = await queue.get()
                            prev_sql.append(sql.removeprefix("FAILURE: "))
                        if sql == 'ERROR' and sql_n_rev == -1:
                            raise ValueError('Agent output not correctly received')

                        # if the SQL generation failed, do not continue with NL generation 
                        if sql_success != 'success':
                            raise ValueError(f"SQL generation failed.")
                        
                        logger.debug("Generating NL question")                    
                        runtime.start()
                        nl_start_t = time.time()
                        await runtime.publish_message(
                            NLGenerationTask(
                                nl_task=(                                                        
                                    "Consider the following information:\n"
                                    f"The table '{rsc_name}' belongs to the package '{pkg_name}'. "
                                    f"This package is published by the organisaztion '{org_name}', titled as '{org_title}' that is about {org_desc}, under the jurisdiction '{jur}'. "
                                    f"The table description is: '{notes}'.\n"
                                    f"Keywords and tags about it are: {keywords}, {tags}.\n"
                                    f"Example rows with schema:\n{df_str}"
                                    f"\n{'-' * 50}\n"
                                    f"Generate a natural language question which represents the SQL query {sql} on the given table."                                    
                                    f"The new question must be different from previous: {prev_nl}. "
                                    "Pay attention to all the clauses used into the query. "
                                    "You must introduce into the question remainders to keywords, organization and other metadata."
                                )
                            ),
                            topic_id=nl_generation_topic_id
                        )
                        
                        # wait the agent response
                        await runtime.stop_when_idle()
                        nl_time = round(time.time() - nl_start_t, 3)
                        while not queue.empty():
                            nl, nl_n_rev, nl_intok, nl_outok, nl_review = await queue.get()                        
                            prev_nl.append(nl.removeprefix("FAILURE: "))
                        if nl == 'ERROR' and nl_n_rev == -1:
                            raise ValueError('Agent output not correctly received')
                        
                        # we do not want that table or columns named are explicitly referenced
                        # inside the question: sometimes the model does that, and that's not good
                        # TODO put this inside a tool for dynamic fixing
                        if any(re.search(r"(\"|\')" + column + r"(\"|\')", nl) or ('_' in column and column in nl) for column in df.columns):
                            nl = f'FAILURE: {nl}'
                            nl_review = f'Original column name inside question - {nl_review}'

                        logger.debug("Idle state after NL generation")
                    except Exception as e:
                        logger.error(f'Error in generation: {e} - resources {rsc_id}')                        
                        if sql == 'ERROR': sql = f'ERROR: {str(e)}, SQL: {nl}'
                        if nl == 'ERROR' : nl  = f'ERROR: {str(e)}, NL: {nl}'

                    success = not nl.startswith('ERROR') and not sql.startswith('ERROR') and not nl.startswith('FAILURE') and not sql.startswith('FAILURE')
                    
                    n_id += 1
                    results.append({
                        "id"        : n_id,
                        "type"      : "single-table",
                        "difficulty": difficulty,
                        "success"   : success,
                        
                        "r_rsc_id"  : rsc_id,
                        "s_rsc_id"  : None,
                        "r_pkg_id"  : pkg_id,
                        "s_pkg_id"  : None,
                        "r_rsc_name": rsc_name,
                        "s_rsc_name": None,
                        "r_col_name": None,
                        "s_col_name": None,     
                                                
                        "sql"       : sql,
                        "nl"        : nl,

                        "sql_success": sql_success,
                        "sql_time"  : sql_time,
                        "sql_n_rev" : sql_n_rev,
                        "sql_review": None if success else sql_review,
                        "sql_intok" : sql_intok,
                        "sql_outok" : sql_outok,

                        "nl_time"   : nl_time,
                        "nl_n_rev"  : nl_n_rev,
                        "nl_review" : None if success else nl_review,
                        "nl_intok"  : nl_intok,
                        "nl_outok"  : nl_outok,
                        
                        "tot_time"  : nl_time + sql_time,
                    })

        ############################################
        ####### Generate Multi-Table queries #######
        ############################################

        multi_table_sql_base_prompt = (
            "Given the following information:\n"
            "Use 'R' to indicate the first table. "
            f'Its schema is: {r_sql_schema}\n'
            f"Example rows of R table:\n{r_df_str}"
            f"\n{'-' * 50}\n"
            "Use 'S' to indicate the second table. "
            f"Its schema is: {s_sql_schema}\n"
            f"Example rows of S table: {s_df_str}"
            f"\n{'-' * 50}\n" 
        )

        multi_table_nl_base_prompt = (
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
        )

        for task in ["JOIN", "UNION"]:
            # do join when tables do not share the identical schema, and union
            # in the opposite case (for simplicity now)
            if task == "JOIN" and (not DO_JOIN or DO_JOIN and set(r_columns_dtypes) == set(s_columns_dtypes)):
                continue
            if task == "UNION" and (not DO_UNION or DO_UNION and set(r_columns_dtypes) != set(s_columns_dtypes)):
                continue

            # remove previous results
            prev_sql.clear() 
            prev_nl.clear()

            # bring the dataframe in the global scope 
            # to be used in the tool
            globals()['R'] = r_df
            globals()['S'] = s_df
            
            # get pairs of (colname, dtype) in common
            matches = list(zip(*(set(r_columns_dtypes) & set(s_columns_dtypes))))

            for nq, difficulty in enumerate(levels):            
                logger.info(f"{i=}, step {nq} - MULTI TABLE - {task.upper()}")

                # reset to default values
                sql = nl = "ERROR"
                sql_time = nl_time = -1
                sql_n_rev = nl_n_rev = -1
                sql_intok = sql_outok = -1
                nl_intok = nl_outok = -1
                sql_success = False
                sql_review = nl_review = "NOREVIEW"

                sql_join_prompt = (
                    f"Generate a {difficulty} SQL query based on the given tables. Use only 'R' and 'S' to reference the tables. "
                    f"The query must include a JOIN on the R column {r_col_name} and on the S column {s_col_name}. "
                    f"The new query must be different from previous queries: {prev_sql}. "
                )

                sql_union_prompt = (
                    f"Generate a {difficulty} UNION SQL query based on the given tables. Use only 'R' and 'S' to reference the tables. "
                    f"The two tables have in common these attribtues: {matches[0] if matches else []}. "
                    f"The new query must be different from previous queries: {prev_sql}. "
                )

                try:
                    logger.debug("Generating SQL query")
                    runtime.start()
                    sql_start_t = time.time()
                    await runtime.publish_message(                    
                        SQLGenerationTask(sql_task=(
                            multi_table_sql_base_prompt + \
                            sql_join_prompt if task == "JOIN" else sql_union_prompt
                            )
                        ),
                        topic_id=sql_generation_topic_id
                    )
                    
                    # wait the agent response
                    await runtime.stop_when_idle()
                    sql_time = round(time.time() - sql_start_t, 3)

                    logger.debug("Idle state after SQL generation.")
                    
                    while not queue.empty():
                        sql, sql_n_rev, sql_success, sql_intok, sql_outok, sql_review = await queue.get()
                        prev_sql.append(sql.removeprefix("FAILURE: "))
                    if sql == 'ERROR' and sql_n_rev == -1:
                        raise ValueError('Agent output not correctly received')
                            
                    if sql_success != 'success':
                        raise ValueError(f"SQL generation failed.")
                    
                    logger.debug("Generating NL question")
                    
                    runtime.start()
                    nl_start_t = time.time()
                    await runtime.publish_message(
                        NLGenerationTask(
                            nl_task=(                            
                                    multi_table_nl_base_prompt + 
                                    f"Generate a natural language question which accurately represents the SQL query {sql} on the given tables and its aim."                    
                                    "Pay attention to all the clauses used into the query. "
                                    "You must introduce into the question remainders to keywords, organization and other metadata."
                            )
                        ),
                        topic_id=nl_generation_topic_id
                    )
                    
                    # wait the agent response
                    await runtime.stop_when_idle()
                    nl_time = round(time.time() - nl_start_t, 3)
                    while not queue.empty():
                        nl, nl_n_rev, nl_intok, nl_outok, nl_review = await queue.get()
                        prev_nl.append(nl.removeprefix("FAILURE: "))  
                    if nl == 'ERROR' and nl_n_rev == -1:
                        raise ValueError('Agent output not correctly received')
                
                    # we do not want that table or columns named are explicitly referenced
                    # inside the question: sometimes the model does that, and that's not good
                    # TODO put this inside a tool for dynamic fixing
                    if any(re.search(r"(\"|\')" + column + r"(\"|\')", nl) or ('_' in column and column in nl) for column in r_df.columns) \
                            or any(re.search(r"(\"|\')" + column + r"(\"|\')", nl) or ('_' in column and column in nl) for column in s_df.columns) \
                            or re.search(r"(\"|\')" + r_rsc_name + r"(\"|\')", nl) \
                            or re.search(r"(\"|\')" + s_rsc_name + r"(\"|\')", nl):
                        nl = f'FAILURE: {nl}'
                        nl_review = f'Original column name inside question - {nl_review}'

                    logger.debug("Idle state after NL generation")                

                except Exception as e:
                    logger.error(f'Error in {task} generation: {e} - resources {r_rsc_id}, {s_rsc_id}')
                    if sql == 'ERROR': sql = f'ERROR: {str(e)}, SQL: {nl}'
                    if nl == 'ERROR' : nl  = f'ERROR: {str(e)}, NL: {nl}'

                success = not nl.startswith('ERROR') and not sql.startswith('ERROR') and not nl.startswith('FAILURE') and not sql.startswith('FAILURE')
                n_id += 1
                results.append({
                    "id"        : n_id,
                    "type"      : f"multi-table-{task.lower()}",
                    "difficulty": difficulty,
                    "success"   : success,

                    "r_rsc_id"  : r_rsc_id,
                    "s_rsc_id"  : s_rsc_id,
                    "r_pkg_id"  : r_pkg_id,
                    "s_pkg_id"  : s_pkg_id,
                    "r_rsc_name": r_rsc_name,
                    "s_rsc_name": s_rsc_name,
                    "r_col_name": r_col_name,
                    "s_col_name": s_col_name,
                
                    "sql"       : sql,
                    "nl"        : nl,

                    "sql_success": sql_success,
                    "sql_time"  : sql_time,
                    "sql_n_rev" : sql_n_rev,
                    "sql_review": None if success else sql_review,
                    "sql_intok" : sql_intok,
                    "sql_outok" : sql_outok,

                    "nl_time"   : nl_time,
                    "nl_n_rev"  : nl_n_rev,
                    "nl_review" : None if success else nl_review,
                    "nl_intok"  : nl_intok,
                    "nl_outok"  : nl_outok,
                    
                    "tot_time"  : nl_time + sql_time,
                })

        file_exists = os.path.exists(queries_path)        
        pl.DataFrame(results) \
            .write_csv(open(queries_path, 'a' if file_exists else 'w'), include_header=not file_exists)
        results.clear()

        logger.info(f"Run {i + 1}/{len(rows)} ({round((i + 1) * 100 / len(rows), 3)}%), (step_time: {round(time.time() - start_step_t, 3)}s, total_time:{round(time.time() - start_t, 3)}s)")

    try:
        await runtime.close()
    except Exception as e:
        logger.error(f"Error on closing: {e}")

    file_exists = os.path.exists(queries_path)
    pl.DataFrame(results) \
        .write_csv(open(queries_path, 'a' if file_exists else 'w'), include_header=not file_exists)
    results.clear()


    logger.info("Done")

    end_t = time.time()
    total_t = round(end_t - start_t, 3)
    logger.info(f"Total time: {total_t}s")

    if os.path.isdir(duckdb_tmp_path):
        logger.info("Deleting DuckDB tmp directory")
        shutil.rmtree(duckdb_tmp_path)

if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))