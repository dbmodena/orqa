import os
import re
import sys
import json
import time
import random
import asyncio
import logging
import warnings
from typing import List
from typing_extensions import Annotated

import duckdb
import jsonlines
import polars as pl

from autogen_core.tools import FunctionTool, Tool
from autogen_core import ClosureAgent, ClosureContext, MessageContext, SingleThreadedAgentRuntime, TypeSubscription

from orqa.utils import sanitize_string, get_resource_metadata
from orqa.agents.utils import *
from orqa.agents.reviewer import ReviewerAgent
from orqa.agents.question_generator import NLQuestionGeneratorAgent, SQLQueryGeneratorAgent


warnings.filterwarnings("ignore")


def map_dtype_to_sql(dtype):    
    if isinstance(dtype, pl.Int64):
        return 'BIGINT'
    elif isinstance(dtype, pl.Float64):
        return 'FLOAT'
    elif isinstance(dtype, pl.String):
        return 'VARCHAR(255)'
    elif isinstance(dtype, pl.Date):
        return 'DATE'
    elif isinstance(dtype, pl.Datetime):
        return 'DATETIME'
    else:
        return 'VARCHAR(255)'  # Default case for unknown types

def create_table_sql(df: pl.DataFrame, table_name):    
    columns_dtypes = []
    for column, dtype in df.schema.items():
        sql_type = map_dtype_to_sql(dtype)
        columns_dtypes.append((column, sql_type))
    
    columns_sql = ",\n    ".join(map(lambda cd: f"{cd[0], {cd[1]}}", columns_dtypes))
    create_table_stmt = f"CREATE TABLE {table_name} (\n    {columns_sql}\n);"
    return create_table_stmt, columns_dtypes


async def verify_sql(sql_query: Annotated[str, "A SQL query which represents the natural language question."]):
    error = results  = None
    
    try:
        R: pl.DataFrame = globals()['R']
        if 'S' in globals():
            S: pl.DataFrame = globals()['S']
        
        # duckdb can use directly the dataframe
        # without disk operations
        results = duckdb.query(sql_query).fetchmany()
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


async def amain(tag, from_, to_):     
    data_path           = f"{os.path.dirname(__file__)}/../data"
    tables_path         = f"{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}"
    metadata_path       = f"{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl"
    log_path            = f"{data_path}/log/{tag}/4_generation_{time.strftime('%y%m%d_%H_%M_%S')}.log"
    
    bird_dev_path       = f"{data_path}/bird-mini-dev/dev.json"

    evaluated_path      = f"{data_path}/outputs/{tag}/evaluated_joins.csv"
    queries_path        = f"{data_path}/outputs/{tag}/generated_queries_union.jsonl"

    UP_TO_ROW           = 1000000

    # maximum numbers of reviews
    MAX_REVIEWS         = 3

    # in some cases tables have many columns
    # to generate questions, we may use only
    # a subset, plus the join column 
    MAX_N_COLUMNS       = 20

    # to limit the context passed to the LLM-agent from the metadata notes
    # (as number of characters)
    MAX_LENGTH_NOTES    = 300
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 3

    # minimum score from the evaluation stage
    MIN_SCORE           = 8

    DO_SINGLE_TABLE     = False
    DO_JOIN             = False
    DO_UNION            = True

    # the model name (here we will use LiteLLM and Ollama models)
    sql_gen_model       = "qwen2.5-coder-32b" # "llama3.3" #
    sql_rev_model       = "qwen2.5-coder-32b" # "llama3.3" #
    nl_gen_model        = "qwen2.5-32b" # "llama3.3" #
    nl_rev_model        = "qwen2.5-32b" # "llama3.3" #
    
    # polars df-to-str configuration
    pl_str_config = {
        'tbl_hide_dataframe_shape': True,
        'tbl_width_chars': 1000,
        'tbl_formatting': 'MARKDOWN',
        'tbl_cols': MAX_N_COLUMNS
    }

    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("agent_logger_union")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter("%(asctime)s,[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    stdout_hanlder = logging.StreamHandler()
    logger.addHandler(stdout_hanlder)

    logger.info("Loading BIRD mini-dev")
    with open(bird_dev_path) as file:
        bird_dev = json.load(file)
    
    # take the three subsets of questions from BIRD, grouped by their difficulty
    bird_questions = {
        d: list(filter(lambda q: q['difficulty'] == d, bird_dev)) 
        for d in ['simple', 'moderate', 'challenging']
    }
    
    logger.info("Reading Table IDs")
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    logger.info("Loading evaluated JOIN pairs")
    joins = pl.read_csv(evaluated_path).drop_nans(subset=["score"]).filter(pl.col('score') >= MIN_SCORE)    
    
    logger.info(f"Evaluated pairs with score >= {MIN_SCORE}: {joins.shape[0]}")

    # create the tools used by the agent
    tools: List[Tool] = [
        FunctionTool(
            verify_sql, 
            name='verify_sql',
            description="A tool that checks if the proposed SQL query works."
        )
    ]

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
            await queue.put((message.sql_query, message.n_rev, message.sql_success, message.input_tokens, message.output_tokens, message.review))
        elif isinstance(message, NLGenerationResult):
            await queue.put((message.nl_question, message.n_rev, message.input_tokens, message.output_tokens, message.review))

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

    n_id = 0

    rows = joins.rows()[:UP_TO_ROW if isinstance(UP_TO_ROW, int) else joins.shape[0]]
    for i, row in enumerate(rows):
        start_step_t = time.time()
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, _, _, original_r_col_name, original_s_col_name, r_pkg_id, s_pkg_id, _, _ = row

            r_rsc_id, r_rsc_name, _, _, r_pkg_notes, r_pkg_keywords, r_pkg_tags = get_resource_metadata(r_tab_id, table_ids, metadata)
            s_rsc_id, s_rsc_name, _, _, s_pkg_notes, s_pkg_keywords, s_pkg_tags = get_resource_metadata(s_tab_id, table_ids, metadata)
                        
            # limit the length of the notes and remove some chars (needed?)
            r_pkg_notes = re.sub(r"(\n|\r|\t)", " ", r_pkg_notes)[:MAX_LENGTH_NOTES]
            s_pkg_notes = re.sub(r"(\n|\r|\t)", " ", s_pkg_notes)[:MAX_LENGTH_NOTES]

            r_col_name= sanitize_string(original_r_col_name)
            s_col_name= sanitize_string(original_s_col_name)

            # read the tables (use pandas to *try to* convert the dtypes after cleaning)
            r_df = (
                pl
                .scan_parquet(f'{tables_path}/{table_ids[r_tab_id]}')
                .select(pl.all().map_elements(sanitize_string, pl.String))
                .rename(sanitize_string)
                .collect()                
            )

            s_df = (
                pl
                .scan_parquet(f'{tables_path}/{table_ids[s_tab_id]}')
                .select(pl.all().map_elements(sanitize_string, pl.String))
                .rename(sanitize_string)
                .collect()                
            )

            # dtype conversion to int/float
            for column in r_df.columns:
                try:
                    dtype = pl.Float32 if any(',' in str(x) or '.' in str(x) for x in set(r_df.select(column).sample(1000, with_replacement=True).to_series())) else pl.Int32
                    r_df = r_df.with_columns(pl.col(column).cast(dtype))
                except:
                    continue
            
            for column in s_df.columns:
                try:
                    dtype = pl.Float32 if any(',' in str(x) or '.' in str(x) for x in set(s_df.select(column).sample(1000, with_replacement=True).to_series())) else pl.Int32
                    s_df = s_df.with_columns(pl.col(column).cast(dtype))
                except: continue
            
            # keep only the first MAX_N_COLUMNS columns 
            r_df = r_df.drop(r_col_name).insert_column(0, r_df.get_column(r_col_name)).select(r_df.columns[:MAX_N_COLUMNS])
            s_df = s_df.drop(s_col_name).insert_column(0, s_df.get_column(s_col_name)).select(s_df.columns[:MAX_N_COLUMNS])

            # create the SQL schema
            r_sql_schema, r_columns_dtypes = create_table_sql(r_df, 'R')
            s_sql_schema, s_columns_dtypes = create_table_sql(s_df, 'S')
            
        except Exception as e:
            logger.error(f"Error in query preparation: >>>{e}<<< ")
            continue


        #############################################
        ##### Generate Single-Table queries    ######
        #############################################
        if DO_SINGLE_TABLE:
            for rsc_id, pkg_id, rsc_name, df, sql_schema, notes, keywords, tags in [
                (r_rsc_id, r_pkg_id, r_rsc_name, r_df, r_sql_schema, r_pkg_notes, r_pkg_keywords, r_pkg_tags),
                (s_rsc_id, s_pkg_id, s_rsc_name, s_df, s_sql_schema, s_pkg_notes, s_pkg_keywords, s_pkg_tags)]:
                
                if rsc_id in already_seen_tables:
                    continue

                already_seen_tables.add(rsc_id)

                # bring the dataframe in the global scope 
                # to be used in the tool
                globals()['R'] = df
                
                query_question_data = {
                    "id"        : n_id,
                    "type"      : "single-table",
                    "rsc_id"    : rsc_id,                
                    "pkg_id"    : pkg_id,
                    "rsc_name"  : rsc_name,                
                }
                n_id += 1

                prev_sql = []
                prev_nl = []
                current_generations = {}            
                
                for nq, (difficulty, bird_q) in enumerate(bird_questions.items()):            
                    logger.info(f"{i=}, step {nq} - SINGLE TABLE")

                    # pass a formatted version of the tables to the completion client,
                    # only a small portion of the tables
                    with pl.Config(**pl_str_config):
                        df_str = str(df.select(df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))

                    sql = nl = "ERROR"
                    sql_time = nl_time = -1
                    sql_n_rev = nl_n_rev = -1
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
                                f"Be different from previous queries: {prev_sql}. "                                
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
                            prev_sql.append(sql)
                        if sql == 'ERROR' and sql_n_rev == -1:
                            raise ValueError('Agent output not correctly received')


                        logger.debug("Generating NL question")                    
                        runtime.start()
                        nl_start_t = time.time()
                        await runtime.publish_message(
                            NLGenerationTask(
                                nl_task=(                                                        
                                    "Consider the following information:\n"
                                    f"The table R is about: {notes}.\n"
                                    f"Some keywords and tags about it are: {keywords}, {tags}.\n"
                                    f"Example rows with schema:\n{df_str}"
                                    f"\n{'-' * 50}\n"
                                    f"Generate a natural language question which represents the SQL query {sql} on the given table."
                                    "Do not include any table name inside the question, and do not use 'dataset_r'. "
                                    "Do not use original column names if they are not human-like: try to figure out what an "
                                    "abbreviation means based on the given context (like 'geo' --> 'geography' --> 'region'). "
                                    "The question must be human-like: do not use SQL-like words, such as null or select. "
                                    "Try to generate fluent question as human."
                                    "If keyowrds or notes are meaningful, insert remiders to them. "
                                    f'Be different from previous questions: {prev_nl}. '
                                    "Your response must be only the question, nothing else."
                                )
                            ),
                            topic_id=nl_generation_topic_id
                        )
                        
                        # wait the agent response
                        await runtime.stop_when_idle()
                        nl_time = round(time.time() - nl_start_t, 3)
                        while not queue.empty():
                            nl, nl_n_rev, nl_intok, nl_outok, nl_review = await queue.get()                        
                            prev_nl.append(nl)
                        if nl == 'ERROR' and nl_n_rev == -1:
                            raise ValueError('Agent output not correctly received')

                        logger.debug("Idle state after NL generation")
                    except Exception as e:
                        logger.error(f'Error in generation: {e}')
                        if sql == 'ERROR': sql = f'ERROR: {str(e)}, SQL: {nl}'
                        if nl == 'ERROR' : nl = f'ERROR: {str(e)}, NL: {nl}'

                    success = not nl.startswith('ERROR') and not sql.startswith('ERROR') and not nl.startswith('FAILURE') and not sql.startswith('FAILURE')
                        
                    current_generations[difficulty] = {
                        "nq"        : nq,
                        "sql"       : sql,
                        "nl"        : nl,
                        "success"   : success,

                        "sql_success": sql_success,
                        "sql_time"  : sql_time,
                        "sql_n_rev" : sql_n_rev,
                        "sql_review": sql_review,
                        "sql_intok" : sql_intok,
                        "sql_outok" : sql_outok,

                        "nl_time"   : nl_time,
                        "nl_n_rev"  : nl_n_rev,
                        "nl_review" : nl_review,
                        "nl_intok"  : nl_intok,
                        "nl_outok"  : nl_outok,
                        
                        "tot_time"  : nl_time + sql_time,
                    }
                
                query_question_data['query_question'] = current_generations
                results.append(query_question_data)



        #############################################
        ##### Generate Multi-Table JOIN queries #####
        #############################################

        if DO_JOIN and not set(r_columns_dtypes) == set(s_columns_dtypes):
            query_question_data = {
                "id"        : n_id,
                "type"      : "multi-table-join",
                "r_rsc_id"  : r_rsc_id,
                "s_rsc_id"  : s_rsc_id,
                "r_pkg_id"  : r_pkg_id,
                "s_pkg_id"  : s_pkg_id,
                "r_rsc_name": r_rsc_name,
                "s_rsc_name": s_rsc_name,
                "r_col_name": r_col_name,
                "s_col_name": s_col_name,
            }
            n_id += 1

            prev_sql = []
            prev_nl = []
            current_generations = {}

            # bring the dataframe in the global scope 
            # to be used in the tool
            globals()['R'] = r_df
            globals()['S'] = s_df
            
            
            for nq, (difficulty, bird_q) in enumerate(bird_questions.items()):            
                logger.info(f"{i=}, step {nq} - MULTI TABLE -JOIN")

                # pass a formatted version of the tables to the completion client,
                # only a small portion of the tables
                with pl.Config(**pl_str_config):
                    r_df_str = str(r_df.select(r_df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))
                    s_df_str = str(s_df.select(s_df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))

                sql_examples, nl_examples = zip(*[(q['SQL'], q['question']) for q in random.sample(bird_q, k=3)])
                
                sql = nl = "ERROR"
                sql_time = nl_time = -1
                sql_n_rev = nl_n_rev = -1
                sql_success = False
                sql_review = nl_review = "NOREVIEW"

                try:
                    logger.debug("Generating SQL query")
                    runtime.start()
                    sql_start_t = time.time()
                    await runtime.publish_message(                    
                        SQLGenerationTask(sql_task=(
                            "Given the following information:\n"                        
                            "Use 'R' to indicate the first table. "
                            f'Its schema is: {r_sql_schema}\n'
                            f"Example rows of R table:\n{r_df_str}"
                            f"\n{'-' * 50}\n"
                            "Use 'S' to indicate the second table. "
                            f"Its schema is: {s_sql_schema}\n"
                            f"Example rows of S table: {s_df_str}"
                            f"\n{'-' * 50}\n"
                            f"Generate a {difficulty} SQL query based on the given tables. "
                            f"The query must include a JOIN on the R column {r_col_name} and on the S column {s_col_name}. "
                            f"Be different from previous queries: {prev_sql}. "                            
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
                        prev_sql.append(sql)
                    if sql == 'ERROR' and sql_n_rev == -1:
                        raise ValueError('Agent output not correctly received')
                            
                    logger.debug("Generating NL question")
                    
                    runtime.start()
                    nl_start_t = time.time()
                    await runtime.publish_message(
                        NLGenerationTask(
                            nl_task=(                                                        
                                "Consider the following information:\n"
                                f"The table R is about: {r_pkg_notes}.\n"
                                f"Some keywords and tags about it are: {r_pkg_keywords}, {r_pkg_tags}.\n"
                                f"Example rows with schema:\n{r_df_str}"
                                f"\n{'-' * 50}\n"
                                f"The table S is about: {s_pkg_notes}.\n"
                                f"Some keywords and tags about it are: {s_pkg_keywords}, {s_pkg_tags}.\n"
                                f"Example rows: {s_df_str}"
                                f"\n{'-' * 50}\n"
                                f"Generate a natural language question which represents the SQL query {sql} on the given tables."
                                "Do not include any table name inside the question, and do not use 'dataset_r' or 'dataset_s'. "
                                "Do not use original column names if they are not human-like: try to figure out what an "
                                "abbreviation means based on the given context (like 'geo' --> 'geography' --> 'region'). "
                                "The question must be human-like: do not use SQL-like words, such as null or select. "
                                "If keyowrds or notes are meaningful, insert some remiders to them. "
                                f"Keep focus on R column {r_col_name} and on S column {s_col_name}. "
                                f'Be different from previous questions: {prev_nl}. '
                                "Your response must be only the question, nothing else."
                            )
                        ),
                        topic_id=nl_generation_topic_id
                    )
                    
                    # wait the agent response
                    await runtime.stop_when_idle()
                    nl_time = round(time.time() - nl_start_t, 3)
                    while not queue.empty():
                        nl, nl_n_rev, nl_intok, nl_outok, nl_review = await queue.get()
                        prev_nl.append(nl)  
                    if nl == 'ERROR' and nl_n_rev == -1:
                        raise ValueError('Agent output not correctly received')                        

                    logger.debug("Idle state after NL generation")                
                except Exception as e:
                    logger.error(f'Error in generation: {e}')
                    if sql == 'ERROR': sql = f'ERROR: {str(e)}, SQL: {nl}'
                    if nl == 'ERROR' : nl = f'ERROR: {str(e)}, NL: {nl}'

                success = not nl.startswith('ERROR') and not sql.startswith('ERROR') and not nl.startswith('FAILURE') and not sql.startswith('FAILURE')
                        
                current_generations[difficulty] = {
                    "nq"        : nq,
                    "sql"       : sql,
                    "nl"        : nl,
                    "success"   : success,

                    "sql_success": sql_success,
                    "sql_time"  : sql_time,
                    "sql_n_rev" : sql_n_rev,
                    "sql_review": sql_review,
                    "sql_intok" : sql_intok,
                    "sql_outok" : sql_outok,

                    "nl_time"   : nl_time,
                    "nl_n_rev"  : nl_n_rev,
                    "nl_review" : nl_review,
                    "nl_intok"  : nl_intok,
                    "nl_outok"  : nl_outok,
                    
                    "tot_time"  : nl_time + sql_time,
                }
            
            query_question_data['query_question'] = current_generations
            results.append(query_question_data)
            
        
        #############################################
        ##### Generate Multi-Table UNION queries ####
        #############################################

        if DO_UNION:
            # For the UNION query generation, we can now try a very
            # simple schema alignment process, checking if the two tables
            # share at most a high percentage of attributes

            # get pairs of (colname, dtype) in common
            matches = set(r_columns_dtypes) & set(s_columns_dtypes)

            # if for both tables the number of matches is more 
            # than the 70% if the total attributes, these are
            # likely unionable
            # are_unionable = (len(matches) / len(r_columns_dtypes) >= 0.7) \
            #     & (len(matches) / len(s_columns_dtypes) >= 0.7) \
            #     & (len(r_columns_dtypes) >= 3)
            unionable = set(r_columns_dtypes) == set(s_columns_dtypes)
            if not unionable:
                continue

            query_question_data = {
                "id"        : n_id,
                "type"      : "multi-table-union",
                "r_rsc_id"  : r_rsc_id,
                "s_rsc_id"  : s_rsc_id,
                "r_pkg_id"  : r_pkg_id,
                "s_pkg_id"  : s_pkg_id,
                "r_rsc_name": r_rsc_name,
                "s_rsc_name": s_rsc_name,
                "r_col_name": r_col_name,
                "s_col_name": s_col_name,
            }

            n_id += 1

            prev_sql = []
            prev_nl = []
            current_generations = {}

            # bring the dataframe in the global scope 
            # to be used in the tool
            globals()['R'] = r_df
            globals()['S'] = s_df
            
            
            for nq, (difficulty, bird_q) in enumerate(bird_questions.items()):            
                logger.info(f"{i=}, step {nq} - MULTI TABLE - UNION")

                # pass a formatted version of the tables to the completion client,
                # only a small portion of the tables
                with pl.Config(**pl_str_config):
                    r_df_str = str(r_df.select(r_df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))
                    s_df_str = str(s_df.select(s_df.columns[:MAX_N_COLUMNS]).sample(N_ROWS_SAMPLE, with_replacement=True))

                sql_examples, nl_examples = zip(*[(q['SQL'], q['question']) for q in random.sample(bird_q, k=3)])
                
                sql = nl = "ERROR"
                sql_time = nl_time = -1
                sql_n_rev = nl_n_rev = -1
                sql_success = False
                sql_review = nl_review = "NOREVIEW"

                #TODO search for Jinja templates for prompting
                try:
                    logger.debug("Generating SQL query")
                    runtime.start()
                    sql_start_t = time.time()
                    await runtime.publish_message(                    
                        SQLGenerationTask(sql_task=(
                            "Given the following information:\n"                        
                            "Use 'R' to indicate the first table. "
                            f'Its schema is: {r_sql_schema}\n'
                            f"Example rows of R table:\n{r_df_str}"
                            f"\n{'-' * 50}\n"
                            "Use 'S' to indicate the second table. "
                            f"Its schema is: {s_sql_schema}\n"
                            f"Example rows of S table: {s_df_str}"
                            f"\n{'-' * 50}\n"
                            f"Generate a {difficulty} UNION SQL query based on the given tables. "
                            f"The two tables have in common these attribtues: {list(zip(*matches))[0]}. "
                            f"Be different from previous queries: {prev_sql}. "
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
                        prev_sql.append(sql)
                    if sql == 'ERROR' and sql_n_rev == -1:
                        raise ValueError('Agent output not correctly received')
                            
                    logger.debug("Generating NL question")
                    
                    runtime.start()
                    nl_start_t = time.time()
                    await runtime.publish_message(
                        NLGenerationTask(
                            nl_task=(                                                        
                                "Consider the following information:\n"
                                f"The table R is about: {r_pkg_notes}.\n"
                                f"Some keywords and tags about it are: {r_pkg_keywords}, {r_pkg_tags}.\n"
                                f"Example rows with schema:\n{r_df_str}"
                                f"\n{'-' * 50}\n"
                                f"The table S is about: {s_pkg_notes}.\n"
                                f"Some keywords and tags about it are: {s_pkg_keywords}, {s_pkg_tags}.\n"
                                f"Example rows: {s_df_str}"
                                f"\n{'-' * 50}\n"
                                f"Generate a natural language question which represents the SQL query {sql} on the given tables."
                                "Do not include any table name inside the question, and do not use 'dataset_r' or 'dataset_s'. "
                                "Do not use original column names if they are not human-like: try to figure out what an "
                                "abbreviation means based on the given context (like 'geo' --> 'geography' --> 'region'). "
                                "The question must be human-like: do not use SQL-like words, such as null or select. "
                                "If keyowrds or notes are meaningful, insert some remiders to them. "
                                f'Be different from previous questions: {prev_nl}. '
                                "Your response must be only the question, nothing else."
                            )
                        ),
                        topic_id=nl_generation_topic_id
                    )
                    
                    # wait the agent response
                    await runtime.stop_when_idle()
                    nl_time = round(time.time() - nl_start_t, 3)
                    while not queue.empty():
                        nl, nl_n_rev, nl_intok, nl_outok, nl_review = await queue.get()
                        prev_nl.append(nl)  
                    if nl == 'ERROR' and nl_n_rev == -1:
                        raise ValueError('Agent output not correctly received')                        

                    logger.debug("Idle state after NL generation")                
                except Exception as e:
                    logger.error(f'Error in generation: {e}')
                    if sql == 'ERROR': sql = f'ERROR: {str(e)}, SQL: {nl}'
                    if nl == 'ERROR' : nl = f'ERROR: {str(e)}, NL: {nl}'

                success = not nl.startswith('ERROR') and not sql.startswith('ERROR') and not nl.startswith('FAILURE') and not sql.startswith('FAILURE')
                        
                current_generations[difficulty] = {
                    "nq"        : nq,
                    "success"   : success,
                    "sql"       : sql,
                    "nl"        : nl,

                    "sql_success": sql_success,
                    "sql_time"  : sql_time,
                    "sql_n_rev" : sql_n_rev,
                    "sql_review": sql_review,
                    "sql_intok" : sql_intok,
                    "sql_outok" : sql_outok,

                    "nl_time"   : nl_time,
                    "nl_n_rev"  : nl_n_rev,
                    "nl_review" : nl_review,
                    "nl_intok"  : nl_intok,
                    "nl_outok"  : nl_outok,
                    
                    "tot_time"  : nl_time + sql_time,
                }
                
                query_question_data['query_question'] = current_generations
                results.append(query_question_data)


        if len(results) % 10 == 0:
            with jsonlines.open(queries_path, 'a') as file:
                file.write_all(results)
            results = []

        logger.info(f"Run {i}/{len(rows)} ({round(i * 100 / len(rows), 3)}%), (step_time: {round(time.time() - start_step_t, 3)}s, total_time:{round(time.time() - start_t, 3)}s)")

    try:
        await runtime.close()
    except Exception as e:
        logger.error(f"Error on closing: {e}")

    if len(results) % 10 == 0:
        with jsonlines.open(queries_path, 'a') as file:
            file.write_all(results)
        results = []


    logger.info("Done")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
