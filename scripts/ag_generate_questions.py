import json
import os
import random
import warnings
import re
import sys
import time
import sqlite3
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler

from typing import List
from typing_extensions import Annotated

import jsonlines
import polars as pl

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool, Tool
from autogen_core import ClosureAgent, ClosureContext, MessageContext, SingleThreadedAgentRuntime, TypeSubscription

from orqa.utils import sanitize_string
from orqa.agents.reviewer import ReviewerAgent
from orqa.agents.question_generator import NLQuestionGeneratorAgent, SQLQueryGeneratorAgent
from orqa.agents.utils import (
    NLGenerationResult, SQLGenerationResult, SQLGenerationTask, NLGenerationTask, 
    SQL_GENERATION_TOPIC_TYPE, NL_GENERATION_TOPIC_TYPE, 
    SQL_INTERMEDIATE_TOPIC_TYPE, NL_INTERMEDIATE_TOPIC_TYPE, 
    SQL_RESULT_TOPIC_TYPE, NL_RESULT_TOPIC_TYPE, 
    sql_generation_topic_id, nl_generation_topic_id
)


warnings.filterwarnings("ignore")

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
    R: pl.DataFrame = globals()['R']
    S: pl.DataFrame = globals()['S']
    
    error = result  = None
    try:
        result = pl.sql(sql_query).head(10).collect()
    except Exception as e:
        error = str(e)
        print(f'{sql_query=}, {error=}')
    # con, cur, result, error = None, None, None, None
    # try:
    #     con = sqlite3.connect("tables.db")
    #     cur = con.cursor()
    #     cur.execute(sql_query)
    #     
    #     result = cur.fetchmany()
    # except sqlite3.Error as e:
    #     error = str(e)
    # finally:
    #     if con: con.close()

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


def get_model_client(model, family: str = "unknown") -> OpenAIChatCompletionClient:  # type: ignore
    "Mimic OpenAI API using Local LLM Server."
    return OpenAIChatCompletionClient(
        model=model,
        api_key="NotRequiredSinceWeAreLocal",
        base_url="http://localhost:4000/",
        model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
            "family": family
        },
    )    


async def amain(tag, from_, to_):     
    data_path           = f"{os.path.dirname(__file__)}/../data"
    tables_path         = f"{data_path}/datasets/{tag}/tables/tables_from{from_}_to{to_}"
    metadata_path       = f"{data_path}/datasets/{tag}/metadata/metadata_from{from_}_to{to_}.jsonl"
    log_path            = f"{data_path}/log/{tag}/QuestGen.log"
    
    bird_dev_path       = f"{data_path}/bird-mini-dev/dev.json"

    evaluated_path      = f"{data_path}/outputs/{tag}/evaluated_joins.csv"
    # queries_path        = f"{data_path}/outputs/{tag}/generated_queries.csv"
    queries_path        = f"{data_path}/outputs/{tag}/generated_queries.json"


    UP_TO_ROW           = 5

    MAX_REVIEWS         = 3

    # to limit the context passed to the LLM-agent (the "notes" field may be very very long...)
    MAX_LENGTH_NOTES    = 500
    
    # number of sampled rows passed to the LLM into the question context
    N_ROWS_SAMPLE       = 10

    # the model name (here we will use LiteLLM and Ollama models)
    sql_gen_model       = "qwen2.5-coder-32b"
    sql_rev_model       = "qwen2.5-coder-32b"
    nl_gen_model        = "qwen2.5-7b"
    nl_rev_model        = "qwen2.5-7b"
    

    # set up the logging
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("agentJobLogger")
    logger.setLevel(logging.DEBUG)
    handler = TimedRotatingFileHandler(log_path, when="midnight", interval=1, backupCount=2)
    handler.suffix = "%y-%m-%d_%H:%M:%S.log"
    log_formatter = logging.Formatter("%(asctime)s,[%(process)d],[%(levelname)s],%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    joins = pl.read_csv(evaluated_path).drop_nans(subset=["score"]).filter((pl.col('score') >= 8)).unique(subset=['r_col_name', 's_col_name'])

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
        lambda: ReviewerAgent(get_model_client(sql_rev_model), logger)
    )
    
    await ReviewerAgent.register(
        runtime, "nl_reviewer_agent",
        lambda: ReviewerAgent(get_model_client(nl_rev_model), logger)
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
            await queue.put((message.sql_query, message.n_rev, message.review))
        elif isinstance(message, NLGenerationResult):
            await queue.put((message.nl_question, message.n_rev, message.review))

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

    results = []
    
    for i, row in enumerate(joins.rows()[:UP_TO_ROW]):
        try:
            # take relevant information for the llm-agent
            r_tab_id, s_tab_id, _, _, original_r_col_name, original_s_col_name, _, _, r_pkg_id, s_pkg_id = row[:10]
            r_rsc_id, r_rsc_name, _, r_pkg_title, r_pkg_notes, r_pkg_keywords, r_pkg_tags = get_resource_metadata(r_tab_id, table_ids, metadata)
            s_rsc_id, s_rsc_name, _, s_pkg_title, s_pkg_notes, s_pkg_keywords, s_pkg_tags = get_resource_metadata(s_tab_id, table_ids, metadata)
            
            # in some cases the name is the same (e.g. "dataset") thus to avoid issues
            # when writing tables to database and in SQL add these suffixes
            if r_rsc_name == s_rsc_name:
                r_rsc_name += '_r'
                s_rsc_name += '_s'

            # limit the length of the notes and remove some chars (needed?)
            r_pkg_notes = re.sub(r"(\n|\r|\t)", " ", r_pkg_notes)[:MAX_LENGTH_NOTES]
            s_pkg_notes = re.sub(r"(\n|\r|\t)", " ", s_pkg_notes)[:MAX_LENGTH_NOTES]

            r_col_name, r_rsc_name = sanitize_string(original_r_col_name), sanitize_string(r_rsc_name)
            s_col_name, s_rsc_name = sanitize_string(original_s_col_name), sanitize_string(s_rsc_name)

            # read the tables (use pandas to *try to* convert the dtypes after cleaning)
            r_df = (
                pl
                .scan_parquet(f'{tables_path}/{table_ids[r_tab_id]}')
                .select(
                    pl.all().map_elements(sanitize_string, return_dtype=pl.String))
                .rename(sanitize_string)
                .collect()
            )
            
            s_df = (
                pl
                .scan_parquet(f'{tables_path}/{table_ids[s_tab_id]}')
                .select(pl.all().map_elements(sanitize_string, return_dtype=pl.String))
                .rename(sanitize_string)
                .collect()
            )
            
            # drop all nulls columns
            r_df = r_df[[s.name for s in r_df if not (s.null_count() == r_df.height)]]
            s_df = s_df[[s.name for s in s_df if not (s.null_count() == s_df.height)]]
            
            # drop first rows which sometimes yield misformatted headers
            r_df = pl.from_pandas(r_df.to_pandas().iloc[5:].convert_dtypes())
            s_df = pl.from_pandas(s_df.to_pandas().iloc[5:].convert_dtypes())
            
            # get a small sample from the dataframes
            r_df_sample = r_df.sample(N_ROWS_SAMPLE, with_replacement=True)
            s_df_sample = s_df.sample(N_ROWS_SAMPLE, with_replacement=True)
            
            # load tables into the database for tool check on sql
            # (maybe this could be done more easily with 
            # pandas/polars/duckdb SQL interfaces without storing on a temp database...)
            globals()['R'] = r_df
            globals()['S'] = s_df
            # try: r_df.to_sql(name='Rtable', con="sqlite:///tables.db", index=False, if_exists="replace")
            # except ValueError: pass
            # try: s_df.to_sql(name='Stable', con="sqlite:///tables.db", index=False, if_exists="replace")
            # except ValueError: pass
            
        except Exception as e:
            logger.error(f"Error in query preparation: {e} ")

        old_queries = []

        data = {
            "r_rsc_id"  : r_rsc_id,
            "s_rsc_id"  : s_rsc_id,
            "r_pkg_id"  : r_pkg_id,
            "s_pkg_id"  : s_pkg_id,
            "r_rsc_name": r_rsc_name,
            "s_rsc_name": s_rsc_name,
            "r_col_name": r_col_name,
            "s_col_name": s_col_name,
        }

        current_generations = {}
        
        for nq, (difficulty, bird_q) in enumerate(bird_questions.items()):            
            logger.info(f"Iteration {i=}, step {nq}:")

            sql_examples, nl_examples = zip(*[(q['SQL'], q['question']) for q in random.sample(bird_q, k=4)])
            sql_time = nl_time = -1
            sql = nl = "ERROR"
            sql_n_rev = nl_n_rev = -1

            #TODO search for Jinja templates for prompting
            #TODO define token counts for input/output chat
            try:
                logger.debug("Generating SQL query")
                logger.debug("Starting/Resuming Runtime")
                runtime.start()
                sql_start_t = time.time()
                await runtime.publish_message(                    
                    SQLGenerationTask(sql_task=(
                        "Given the following information:\n"                        
                        "Use 'R' to indicate the first table.\n"
                        f"Example rows:\n{r_df_sample},"
                        "\n############################\n"
                        "Use 'S' to indicate the second table.\n"
                        f"Example rows: {s_df_sample},"
                        "\n############################\n"
                        f"Create a {difficulty} SQL query which requires joining the R column '{r_col_name}' and the S column '{s_col_name}'.\n"                    
                        f"Some examples of good and {difficulty} SQL queries are:\n{'\n'.join(sql_examples)}\n"
                        "Enclose the only the column names into backticks, ``, not the table names."
                        "Use the given tool to validate your SQL query. "
                        )
                        # "In the SQL put only column names inside ``."
                    ),
                    topic_id=sql_generation_topic_id
                )
                
                # wait the agent response
                await runtime.stop_when_idle()
                sql_time = round(time.time() - sql_start_t, 3)

                logger.debug("Idle state after SQL generation.")
                
                while not queue.empty():
                    sql, sql_n_rev, _ = await queue.get()

                if sql == 'ERROR':
                    continue

                logger.debug("Generating NL question")
                logger.debug("Resuming Runtime")
                runtime.start()
                nl_start_t = time.time()
                await runtime.publish_message(
                    NLGenerationTask(
                        nl_task=(
                            f"Given the following SQL query:\n{sql}\n"
                            "Generate a Natural Language question that represents it. "
                            "Consider also the following information:\n"
                            f"The table '{r_rsc_name}' is about:\n{r_pkg_notes}.\n"
                            f"Some keywords and tags about it are: {r_pkg_keywords}, {r_pkg_tags}."
                            "\n#########################################################\n"
                            f"The table '{s_rsc_name}' is about: {s_pkg_notes}.\n"
                            f"Some keywords and tags about it are: {s_pkg_keywords}, {s_pkg_tags}."
                            "\n#########################################################\n"
                            "The question must be at high level. "
                            "Do not include any table name inside the question, and do not use 'dataset_r' or 'dataset_s'. "
                            "Do not use original column names if they are not human-like: try to figure out what an "
                            "abbreviation means based on the given context (like 'geo' --> 'geography'). "
                            "The question must be human-like, so do not use SQL-like words, such as null or select. "
                            "If keyowrds or notes are meaningful, insert some remiders to them. "
                            "Some examples of interesting human-like natural language questions are:\n"
                            f"{'\n'.join(nl_examples)}\n"
                            "Don't use tools. Return only the question you have generated.\n"
                        )
                    ),
                    topic_id=nl_generation_topic_id
                )

                # wait the agent response
                await runtime.stop_when_idle()
                nl_time = round(time.time() - nl_start_t, 3)
                while not queue.empty():
                    nl, nl_n_rev, _ = await queue.get()                

                logger.debug("Idle state after NL generation")
            except Exception as e:
                logger.error(f'Error in NL generation: {e}')
                
            current_generations[difficulty] = {
                "nq"        : nq,
                "sql_n_rev" : sql_n_rev,
                "sql"       : sql,
                "sql_time"  : sql_time,
                "nl_n_rev"  : nl_n_rev,
                "nl"        : nl,
                "nl_time"   : nl_time
            }
        
        data['query_question'] = current_generations
        results.append(data)
    try:
        await runtime.stop_when_idle()
        await runtime.close()
    except:
        pass

    with open(queries_path, 'w') as file:
        json.dump(results, file, indent=3)
        
    
    logger.info("Done")


if __name__ == '__main__':
    asyncio.run(amain(*sys.argv[1:4]))
