import os
import re
import csv
import sys
import time
import sqlite3
import asyncio
import logging
import logging.handlers
from logging.handlers import TimedRotatingFileHandler

from typing import List
from typing_extensions import Annotated

import jsonlines
import polars as pl

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool
from autogen_core import SingleThreadedAgentRuntime, DefaultTopicId
from autogen_core.models import ModelFamily

from orqa.utils import sanitize_string
from orqa.agents.reviewer import ReviewerAgent
from orqa.agents.question_generator import SQLQueryGeneratorAgent
from orqa.agents.utils import SQLGenerationTask, NLGenerationTask


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




async def verify_sql(sql_query: Annotated[str, "A SQL query which represents the natural language question."]):
    con, cur, result, error = None, None, None, None
    try:
        con = sqlite3.connect("tables.db")
        cur = con.cursor()
        cur.execute(sql_query)
        
        result = cur.fetchmany()
    except sqlite3.Error as e:
        error = str(e)
    finally:
        if con: con.close()

    if error:
        return {
            "status": "error",
            "error_description": error,
            "sql_query": sql_query,
            "result": result if result else "N/A"            
        }
    elif len(result) == 0:
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
            "result": str(result)
        }



async def amain(tag, from_, to_):     
    data_path           = f"{os.path.dirname(__file__)}/../data"
    tables_path         = f"{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}"
    metadata_path       = f"{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl"
    log_path            = f"{data_path}/log/{tag}_QuestGen.log"
    
    evaluated_path      = f"{data_path}/outputs/{tag}_evaluated_joins.csv"
    queries_path        = f"{data_path}/outputs/{tag}_generated_queries.csv"

    UP_TO_ROW           = 30

    MAX_RETRIES         = 3

    # number of queries to generate from each join
    N_GEN_QUERIES       = 2

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 500
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 10

    # number of values in common between the joinable columns
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    # the model name (here we will use LiteLLM and Ollama models)
    gen_model           = "ollama/llama3.3:latest"
    reviewer_model      = "ollama/llama3.1:8b"
    # model               = "ollama/qwen2.5:14b"
    # model               = "ollama/deepseek-r1:70b"
    # reviewer_model        = "ollama/deepseek-r1:14b"
    
    base_url            = "http://localhost:4000"
    api_key             = "NotRequiredSinceWeAreLocal"
    temperature         = 0
    
    model_info          = {
        "json_output"       : False,
        "vision"            : False,
        "function_calling"  : True,
        "family"            : ModelFamily.UNKNOWN,
        "keep_alive"        : "6h", # to keep the model in memory more time
        "num_ctx"           : 8192, # to increase the context size (not sure)
        "num_ctx_per_seq"   : 4096   
    }

    # sql_tokens          = ['GROUP BY', 'ORDER BY', 'AVG', 'BETWEEN', 'COUNT', 'LIMIT', 'HAVING', 'WHERE', 'HAVING']
    
    tmp_results         = []

    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("agentJobLogger")
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=3)
    handler.suffix = "%y-%m-%d_%H:%M:%S.log"
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    stdout_hanlder = logging.StreamHandler()
    logger.addHandler(stdout_hanlder)

    
    logger.info("Reading Table IDs")
    table_ids = list(sorted(os.listdir(tables_path), reverse=True))

    logger.info("Loading Resources Metadata")
    with jsonlines.open(metadata_path) as fr:
        metadata = {rsc['id']: md for md in fr.iter() for rsc in md['resources'] if rsc['format'] == 'CSV'}

    logger.info("Loading evaluated JOIN pairs")
    joins = pl.read_csv(evaluated_path).filter((pl.col('ollama/qwen2.5:14b_score') >= 4)) # & (pl.col('ollama/deepseek-r1:14b_score') >= 4))

    with open(queries_path, 'w') as file:
        csv.writer(file).writerow(['r_rsc_id', 's_rsc_id', 'r_pkg_id', 's_pkg_id', 'r_rsc_name', 's_rsc_name', 'r_col_name', 's_col_name', 'num_query', 'num_sql_reviews', 'sql', 'num_nl_reviews', 'nl'])
            

    # create the Natural Language-SQL Query Generator Agent
    # We use LiteLLM, so ollama and other models should not 
    # give problems with this client type anyway 
    logger.info(f"Definine Chat Completion Client ({gen_model=}, {reviewer_model=})")
    qgen_model_client = OpenAIChatCompletionClient(
        model=gen_model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info
    )

    reviewer_model_client = OpenAIChatCompletionClient(
        model=reviewer_model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        model_info=model_info
    )

    qgen_system_message = """
        You are a smart AI assistant. 
        Your task is to generate SQL queries relative to joining tables.        
    """

    reviewer_system_message = """
        You are a query reviewer. You focus on the correctness of a proposed query.
        Do not focus on formatting.
        Pay attention that the new query must be different from old ones.
    """

    # create the agent runtime
    runtime = SingleThreadedAgentRuntime()

    # create the tools used by the agent
    tools: List[Tool] = [
        FunctionTool(
            verify_sql, 
            description="Checks if the proposed SQL query works."
        )
    ]

    # register the query generator agent
    logger.debug("Registering Query Generator Agent")
    await SQLQueryGeneratorAgent.register(
        runtime, "query_generator_agent",
        lambda: SQLQueryGeneratorAgent(qgen_model_client, qgen_system_message, tools, tmp_results, MAX_RETRIES)
    )

    logger.debug("Registering Reviewer Agent")
    await ReviewerAgent.register(
        runtime, "reviewer_agent",
        lambda: ReviewerAgent(reviewer_model_client, reviewer_system_message)
    )

    # start processing messages and create one agent
    logger.info("Started Queries Generation")
    
    for i, row in enumerate(joins.rows()[:UP_TO_ROW], start=1):
            
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, _, _, r_col_name, s_col_name, _, _, r_pkg_id, s_pkg_id = row[:10]
            # r_rsc_id, r_rsc_name, _, r_pkg_note = get_resource_metadata(r_tab_id, table_ids, metadata)
            # s_rsc_id, s_rsc_name, _, s_pkg_note = get_resource_metadata(s_tab_id, table_ids, metadata)
            r_rsc_id, r_rsc_name, _, r_pkg_title, r_pkg_notes, r_pkg_keywords, r_pkg_tags = get_resource_metadata(r_tab_id, table_ids, metadata)
            s_rsc_id, s_rsc_name, _, s_pkg_title, s_pkg_notes, s_pkg_keywords, s_pkg_tags = get_resource_metadata(s_tab_id, table_ids, metadata)
            
            if r_rsc_name == s_rsc_name:
                r_rsc_name += '_r'
                s_rsc_name += '_s'

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_notes = re.sub(r"(\n|\r|\t)", " ", r_pkg_notes)[:MAX_LENGTH_NOTES]
            s_pkg_notes = re.sub(r"(\n|\r|\t)", " ", s_pkg_notes)[:MAX_LENGTH_NOTES]

            # read the tables (use pandas to *try to* convert the dtypes after cleaning)
            r_df = pl.read_parquet(f'{tables_path}/{table_ids[r_tab_id]}').select(pl.all().map_elements(sanitize_string, return_dtype=pl.String)).to_pandas().convert_dtypes()
            s_df = pl.read_parquet(f'{tables_path}/{table_ids[s_tab_id]}').select(pl.all().map_elements(sanitize_string, return_dtype=pl.String)).to_pandas().convert_dtypes()
            
            # rename tables and columns names (to fix that, this should be coherent with the previous steps?)
            r_df.rename(sanitize_string, axis=1, inplace=True)
            s_df.rename(sanitize_string, axis=1, inplace=True)
            r_col_name, s_col_name = sanitize_string(r_col_name), sanitize_string(s_col_name)
            r_rsc_name, s_rsc_name = sanitize_string(r_rsc_name), sanitize_string(s_rsc_name)
            
            # drop null columns
            r_df.dropna(axis=1, how='all', inplace=True)
            s_df.dropna(axis=1, how='all', inplace=True)
            
            # get a small sample from the dataframes
            r_df_sample = r_df.sample(N_ROWS_SAMPLE, replace=True, ignore_index=True).dropna(axis=1, how='any')
            s_df_sample = s_df.sample(N_ROWS_SAMPLE, replace=True, ignore_index=True).dropna(axis=1, how='any')

            # load tables into the database for tool check on sql
            try: r_df.to_sql(name='Rtable', con="sqlite:///tables.db", index=False, if_exists="replace")
            except ValueError: pass
            try: s_df.to_sql(name='Stable', con="sqlite:///tables.db", index=False, if_exists="replace")
            except ValueError: pass
            
        except Exception as e:
            logger.error(f"Error in query preparation: {e} ")


        old_questions = []
        success = True
        for nq in range(N_GEN_QUERIES):
            if not success: break
            success = False
            logger.info(f"Iteration {i=}, step {nq}:")

            try:
                logger.debug("Starting/Resuming Runtime")
                runtime.start()
           
                await runtime.publish_message(                    
                    SQLGenerationTask(sql_task= \
                                        f"""
                        Given the following information: 
                        
                        The r table is:
                        "Rtable"

                        Example rows: 
                        {r_df_sample},
                        ############################

                        The s table is:
                        "Stable"

                        Example rows:
                        {s_df_sample},
                        ############################

                        Create a SQL query which requires joining the r column \"{r_col_name}\" and the s column \"{s_col_name}\".
                        
                        Use the tool \"verify_sql\" to check if the SQL query is correct.

                        Do not modify tables and columns names.
                        Use tokens and others to create different and more complex queries with respect to old questions: 
                        {old_questions}.
                        Do not use "SELECT *", if possible, use one or more functions in selection to create more insteresting queries.
                        Use also other attributes that are present in the two schema and if they do not have null values.

                        In the SQL put only column names inside ``, not table names.
                        """
                    ),
                    topic_id=DefaultTopicId()                  
                )
                
                # wait the agent response
                await runtime.stop_when_idle()

                logger.debug("Idle state after SQL generation.")
                sql_n_rev, sql = tmp_results.pop()
                old_questions.append(sql)
                logger.info(f"SQL generation: {sql_n_rev=}, {sql=}")
            except Exception as e:
                logger.error(f"Error in SQL generation: {e}")
                continue

            try:
                logger.debug("Generating NL")
                runtime.start()
                await runtime.publish_message(
                    NLGenerationTask(
                        nl_task=f"""
                            Given the following SQL query:
                            {sql}
                            
                            Generate a Natural Language question that represents it. 
                            Consider also the following information:

                            The table "{r_rsc_name}" is about:
                            {r_pkg_notes}.

                            Some keywords and tags about it are:
                            {r_pkg_keywords}, {r_pkg_tags}.

                            #########################################################

                            The table "{s_rsc_name}" is about:
                            {s_pkg_notes}.
                            
                            Some keywords and tags about it are:
                            {s_pkg_keywords}, {s_pkg_tags}.

                            #########################################################

                            The question must be at high level, and must not include any table name inside.
                            The question must be human-like, so do not use SQL-like words, such as null or select.
                            Don't use tools.

                            Return only the question.

                        """
                    ),
                    topic_id=DefaultTopicId()
                )

                # wait the agent response
                await runtime.stop_when_idle()

                logger.debug("Idle state after NL generation")
                nl_n_rev, nl = tmp_results.pop()
                logger.info(f"NL generation: {nl_n_rev=}, {nl=}")
            except Exception as e:
                logger.error(f'Error in NL generation: {e}')
                continue
            
            logger.info("#" * 100)
            success = True
            with open(queries_path, 'a') as file:
                csv.writer(file).writerow([r_rsc_id, s_rsc_id, r_pkg_id, s_pkg_id, r_rsc_name, s_rsc_name, r_col_name, s_col_name, nq, sql_n_rev, sql, nl_n_rev, nl])
                
    try:
        await runtime.stop_when_idle()
        await runtime.close()
    except:
        pass
    
    logger.info("Done")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
