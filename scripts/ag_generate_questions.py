import os
import re
import csv
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


def get_package_id(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    return metadata[rsc_id]['id']
    

def get_resource_metadata(rsc_id, table_ids, metadata):
    rsc_id = re.sub(r'(_\d+)?.parquet$', '', table_ids[rsc_id])
    md = next(
        filter(
            lambda r: r['id'] == rsc_id, metadata[rsc_id]['resources']))
    return rsc_id, md['name'], metadata[rsc_id]['title'], metadata[rsc_id]['notes']




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



async def amain():
    tag                 = "CAN"
    from_               = 0
    to_                 = 10_000

    data_path           = f"{os.path.dirname(__file__)}/../data"
    tables_path         = f"{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}"
    metadata_path       = f"{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl"
    log_path            = f"{data_path}/log/{tag}_QuestGen.log"
    
    evaluated_path      = f"{data_path}/outputs/{tag}_evaluated_joins.csv"
    queries_path        = f"{data_path}/outputs/{tag}_generated_queries.csv"

    UP_TO_ROW           = 100

    MAX_RETRIES         = 3

    # number of queries to generate from each join
    N_GEN_QUERIES       = 2

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 500
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 3

    # number of values in common between the joinable columns
    # passed to the LLM into the question context
    MAX_COMM_CELLS      = 10

    # the model name (here we will use LiteLLM and Ollama models)
    model               = "ollama/llama3.3:latest"
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
        "num_ctx"           : 8192 # to increase the context size (not sure)         
    }

    sql_tokens          = ['GROUP BY', 'ORDER BY', 'AVG', 'LIMIT', 'HAVING', 'WHERE', 'HAVING']
    
    tmp_results         = []

    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("agentJobLogger")
    logger.setLevel(logging.DEBUG)
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
    joins = pl.read_csv(evaluated_path).filter(pl.col('size_r_col') < 20)

    with open(queries_path, 'w') as file:
        csv.writer(file).writerow(['r_rsc_id', 's_rsc_id', 'r_pkg_id', 's_pkg_id', 'r_rsc_name', 's_rsc_name', 'r_col_name', 's_col_name', 'num_query', 'num_sql_reviews', 'sql', 'num_nl_reviews', 'nl'])
            

    # create the Natural Language-SQL Query Generator Agent
    # We use LiteLLM, so ollama and other models should not 
    # give problems with this client type anyway 
    logger.info(f"Definine Chat Completion Client ({model=})")
    qgen_model_client = OpenAIChatCompletionClient(
        model=model,
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
            r_rsc_id, r_rsc_name, _, r_pkg_note = get_resource_metadata(r_tab_id, table_ids, metadata)
            s_rsc_id, s_rsc_name, _, s_pkg_note = get_resource_metadata(s_tab_id, table_ids, metadata)

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_note = re.sub(r"(\n|\r|\t)", " ", r_pkg_note)[:MAX_LENGTH_NOTES]
            s_pkg_note = re.sub(r"(\n|\r|\t)", " ", s_pkg_note)[:MAX_LENGTH_NOTES]

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

            # load tables into the database for tool check
            try: r_df.to_sql(name=r_rsc_name, con="sqlite:///tables.db", index=False, if_exists="fail")
            except ValueError: pass
            try: s_df.to_sql(name=s_rsc_name, con="sqlite:///tables.db", index=False, if_exists="fail")
            except ValueError: pass
            
            # get a small sample from the dataframes
            r_df = r_df.sample(N_ROWS_SAMPLE, replace=True, ignore_index=True)
            s_df = s_df.sample(N_ROWS_SAMPLE, replace=True, ignore_index=True)

            # hide index when printing
            r_df.style.hide(axis='index')
            s_df.style.hide(axis='index')

        except Exception as e:
            logger.error(f"Error in query preparation: {e} ")


        old_questions = []
        success = True
        for nq in range(N_GEN_QUERIES):
            if not success: break
            success = False

            try:
                logger.debug("Starting/Resuming Runtime")
                runtime.start()
           
                await runtime.publish_message(                    
                    SQLGenerationTask(sql_task= \
                                        f"""
                        Given the following information: 
                        
                        The r table is:
                        \"{r_rsc_name}\"

                        Example rows: 
                        {r_df},
                        ############################

                        The s table is:
                        \"{s_rsc_name}\",

                        Example rows:
                        {s_df},
                        ############################

                        Create a SQL query which requires joining the r column \"{r_col_name}\" and the s column \"{s_col_name}\".
                        
                        Use the tool \"verify_sql\" to check if the SQL query is correct.

                        Do not modify tables and columns names.                        
                        Use tokens like: {sql_tokens} and others to create different and more complex queries with respect to old questions: {old_questions}.

                        In the SQL put column and table names inside ``.
                        """
                    ),
                    topic_id=DefaultTopicId()                  
                )
                
                # wait the agent response
                await runtime.stop_when_idle()

                logger.debug("Idle state after SQL generation.")
                sql_n_rev, sql = tmp_results.pop()
                old_questions.append(sql)
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
                            {r_pkg_note}.

                            The table "{s_rsc_name}" is about:
                            {s_pkg_note}.

                            The question must be at high level, and must not include any table name inside.
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
            except Exception as e:
                logger.error(f'Error in NL generation: {e}')
                continue
            
            logger.info(f"Iteration {i=}, output: {sql_n_rev=}, {len(tmp_results)=}, {sql=}")
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
    asyncio.run(amain())
