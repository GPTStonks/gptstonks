import ast
import asyncio
import json
import os
import uuid
from functools import partial

import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import PythonREPL, SerpAPIWrapper
from llama_index.indices.postprocessor import SimilarityPostprocessor
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from openbb_terminal.sdk import openbb
from rich.progress import track
from transformers import BitsAndBytesConfig, GPTQConfig

from .utils import (
    get_default_classifier_model,
    get_default_llm,
    get_definitions_path,
    get_definitions_sep,
    get_embeddings_path,
    get_func_parameter_names,
    get_griffin_few_shot_template,
    get_griffin_general_template,
    get_keys_file,
    get_openbb_chat_output,
    get_openbb_chat_output_executed,
    get_wizardcoder_few_shot_template,
)

description = """
GPTStonks API allows interacting with [OpenBB](https://openbb.co/) using natural language.

# Features
The API supports the following features:
- OpenAI LLMs.
- Multiple text embedding models on Hugging Face.
- Asynchronous processing.

# API Operating Modes
The API can operate in two modes:
- **Programmer**: only OpenBB SDK is used to retrieve financial and economical data. It is the fastest and cheapest mode, but it is restricted to OpenBB SDK's functionalities.
- **Agent**: different tools are used, including OpenBB SDK and DuckDuckGo, that can look for general information and specific financial data. It is slower and slightly more expensive than Programmer, but also more powerful.
"""

app = FastAPI(
    title="GPTStonks API",
    description=description,
    summary="API to interact with OpenBB using natural language.",
    version="0.0.1",
    contact={
        "name": "GPTStonks",
        "email": "gptstonks@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

results_store = {}


@app.on_event("startup")
def init_data():
    """Initial function called during the application startup."""

    # Load AutoLlamaIndex
    app.auto_llama_index = AutoLlamaIndex(
        os.getenv("VSI_PATH"),
        os.getenv("EMBEDDING_MODEL_ID", "local:BAAI/bge-large-en-v1.5"),
        os.getenv("LLM_MODEL_ID", "openai:gpt-3.5-turbo"),
        context_window=os.getenv("LLM_CONTEXT_WINDOW", 4096),
        qa_template_str=os.getenv("QA_TEMPLATE", None),
        refine_template_str=os.getenv("REFINE_TEMPLATE", None),
        other_llama_index_vector_index_retriever_kwargs={
            "similarity_top_k": os.getenv("VIR_SIMILARITY_TOP_K", 3)
        },
        other_llama_index_bm25_retriever_kwargs={
            "similarity_top_k": os.getenv("BMR_SIMILARITY_TOP_K", 3)
        },
    )

    # Create agent
    search = DuckDuckGoSearchRun()
    app.python_repl_utility = PythonREPL()
    app.python_repl_utility.globals = globals()
    partial_get_openbb_chat_output_executed = partial(
        get_openbb_chat_output_executed,
        auto_llama_index=app.auto_llama_index,
        python_repl_utility=app.python_repl_utility,
    )
    app.tools = [
        Tool(
            name="Search",
            func=search.run,
            description=os.getenv("AGENT_SEARCH_TOOL_DESCRIPTION"),
        ),
        Tool(
            name="OpenBBChat",
            func=partial_get_openbb_chat_output_executed,
            description=os.getenv("OPENBBCHAT_TOOL_DESCRIPTION"),
            return_direct=True,
        ),
    ]
    app.agent = initialize_agent(
        tools=app.tools,
        llm=OpenAI(
            model_name=os.getenv("LLM_MODEL_ID", "openai:gpt-3.5-turbo").split(":")[1],
            temperature=0,
            request_timeout=os.getenv("AGENT_REQUEST_TIMEOUT", 20),
        ),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=2,
        early_stopping_method=os.getenv("AGENT_EARLY_STOPPING_METHOD", "generate"),
    )

    # Load API keys and set them in OpenBB
    if os.path.exists(get_keys_file()):
        with open(get_keys_file()) as keys_file:
            keys_list = json.load(keys_file)
        openbb.keys.set_keys(keys_dict=keys_list, persist=True, show_output=False)


@app.post("/process_query_async")
async def process_query_async(request: Request):
    """Asynchronous endpoint to start processing the given query. The processing runs in the
    background, and a unique job ID is returned for fetching the result.

    Args:
        request (Request): FastAPI request object containing the query to be processed.

    Returns:
        dict: Contains a message confirming the start of processing and a unique job ID.
    """
    data = await request.json()
    query = data.get("query")
    use_agent = data.get("use_agent", False)

    job_id = str(uuid.uuid4())
    response = {"message": "Processing started", "job_id": job_id}

    asyncio.create_task(run_model_in_background(job_id, query, use_agent))

    return response


async def run_model_in_background(job_id: str, query: str, use_agent: bool):
    """Background task to process the query using the STransformer and OpenBB. Stores the result in
    the global results_store dictionary.

    Args:
        job_id (str): Unique identifier for the job.
        query (str): User query to process.
        use_agent (bool): Whether to run in Agent mode or Programmer mode.

    Returns:
        None
    """

    try:
        if use_agent:
            # Run agent
            agent_output = app.agent({"input": query})
            print(f"{agent_output=}")
            agent_output_str = agent_output["output"]

            try:
                res = json.loads(agent_output_str)
                results_store[job_id] = {
                    "type": "data",
                    "result_data": res,
                    "body": [
                        [f"Observation {idx}: {step[1]}"]
                        for idx, step in enumerate(agent_output["intermediate_steps"])
                    ],
                }
            except Exception as e:
                results_store[job_id] = {
                    "type": "data",
                    "body": [
                        f"Observation {idx}: {step[1]}"
                        for idx, step in enumerate(agent_output["intermediate_steps"])
                    ]
                    + [f"Final answer: {agent_output_str}"],
                }
        else:
            # Run programmer
            openbbchat_output = get_openbb_chat_output(
                query_str=query, auto_llama_index=app.auto_llama_index
            )
            code_str = openbbchat_output.response.split("```python")[1].split("```")[0]
            executed_output_str = app.python_repl_utility.run(code_str)
            try:
                res = json.loads(executed_output_str)
                results_store[job_id] = {
                    "type": "data",
                    "result_data": res,
                    "body": openbbchat_output.response,
                }
            except Exception as e:
                results_store[job_id] = {
                    "type": "data",
                    "body": f"{openbbchat_output.response}\n\nResult: {executed_output_str}",
                }
    except Exception as e:
        print(e)
        results_store[job_id] = {
            "type": "error",
            "body": "Error processing the query. Please try again!",
        }
    finally:
        return


@app.get("/get_processing_result/{job_id}")
async def get_processing_result(job_id: str):
    """Endpoint to fetch the processing result using a job ID.

    Args:
        job_id (str): Unique identifier for the job.

    Returns:
        dict: Contains either the status of processing or the final result once completed.
    """
    result = results_store.get(job_id)
    if result is None:
        return {"status": "processing"}
    else:
        return {"status": "completed", "result": result}


@app.get("/get_api_keys")
async def get_api_keys():
    """Endpoint to retrieve the current API keys information.

    Args:
        None

    Returns:
        dict: Contains information about the current API keys.
    """

    keys = openbb.keys.get_keys_info()
    return {"result": keys}


@app.post("/set_api_keys")
async def set_api_keys(request: Request):
    """Endpoint to set new API keys. The new keys are set in the OpenBB SDK and also stored in a
    JSON file.

    Args:
        request (Request): FastAPI request object containing the keys to set.

    Returns:
        dict: Status message confirming if keys were set correctly or not.
    """

    data = await request.json()
    keys_list = data.get("keys_dict")
    persist = data.get("persist") or True
    show_output = data.get("show_output") or False
    if keys_list is not None:
        openbb.keys.set_keys(keys_dict=keys_list, persist=persist, show_output=show_output)
        with open(get_keys_file(), "w") as keysFile:
            json.dump(keys_list, keysFile)
        return {"status": "Keys set correctly", "keys": keys_list}
    else:
        return {"status": "Key Format was not valid. API Key list not uploaded"}
