import asyncio
import json
import os
import uuid
from functools import partial
from typing import Optional

import gdown
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun, YahooFinanceNewsTool
from langchain.utilities import PythonREPL, WikipediaAPIWrapper
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from openbb_terminal.sdk import openbb

from .utils import (
    arun_qa_over_tool_output,
    get_custom_gptstonks_prefix,
    get_keys_file,
    get_openbb_chat_output,
    get_openbb_chat_output_executed,
    yfinance_info_titles,
)

load_dotenv()

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

    gdown.download_folder(os.getenv("VSI_GDRIVE_URI"), output=os.getenv("VSI_PATH").split(":")[-1])

    embed_model = os.getenv("EMBEDDING_MODEL_ID", "local:BAAI/bge-large-en-v1.5")
    if embed_model == "default":
        embed_model = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
        )

    # Load AutoLlamaIndex
    app.auto_llama_index = AutoLlamaIndex(
        os.getenv("VSI_PATH"),
        embed_model,
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
        other_llama_index_llm_kwargs={
            "timeout": float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
            "temperature": 0,
        },
    )

    # Create agent
    ai_prefix = "AI"
    app.python_repl_utility = PythonREPL()
    app.python_repl_utility.globals = globals()
    app.node_postprocessor = (
        MetadataReplacementPostProcessor(target_metadata_key="window")
        if os.getenv("REMOVE_POSTPROCESSOR", None) is None
        else None
    )
    search_tool = DuckDuckGoSearchRun()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    yhfinance_tool = YahooFinanceNewsTool()
    llm = OpenAI(
        model_name=os.getenv("LLM_MODEL_ID", "openai:gpt-3.5-turbo").split(":")[1],
        temperature=0,
        request_timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=0, output_key="output", ai_prefix=ai_prefix
    )
    app.tools = [
        Tool(
            name=search_tool.name,
            func=None,
            coroutine=partial(
                arun_qa_over_tool_output,
                llm=llm,
                tool=search_tool,
            ),
            description=search_tool.description,
            return_direct=True,
        ),
        Tool(
            name="openbb_terminal",
            func=None,
            coroutine=partial(
                get_openbb_chat_output_executed,
                auto_llama_index=app.auto_llama_index,
                python_repl_utility=app.python_repl_utility,
                node_postprocessor=app.node_postprocessor,
            ),
            description=os.getenv("OPENBBCHAT_TOOL_DESCRIPTION"),
            return_direct=True,
        ),
        Tool(
            name=yhfinance_tool.name,
            func=yfinance_info_titles,
            description=yhfinance_tool.description,
            return_direct=True,
        ),
        Tool(
            name=wikipedia_tool.name,
            func=None,
            coroutine=partial(
                arun_qa_over_tool_output,
                llm=llm,
                tool=wikipedia_tool,
            ),
            description=wikipedia_tool.description,
            return_direct=True,
        ),
    ]
    app.agent_executor = initialize_agent(
        tools=app.tools,
        llm=llm,
        memory=memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=False,
        max_iterations=2,
        early_stopping_method=os.getenv("AGENT_EARLY_STOPPING_METHOD", "generate"),
        agent_kwargs={"ai_prefix": ai_prefix, "prefix": get_custom_gptstonks_prefix()},
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

    return await run_model_in_background(query, use_agent)


async def run_model_in_background(query: str, use_agent: bool) -> dict:
    """Background task to process the query using the STransformer and OpenBB. Stores the result in
    the global results_store dictionary.

    Args:
        job_id (str): Unique identifier for the job.
        query (str): User query to process.
        use_agent (bool): Whether to run in Agent mode or Programmer mode.

    Returns:
        dict: Response to the query.
    """

    try:
        if use_agent:
            # Run agent
            agent_output_str = await app.agent_executor.arun(query)

            try:
                res = json.loads(agent_output_str)
                return {
                    "type": "data",
                    "result_data": res,
                    "body": "> Data returned using `openbb`. Use with caution.",
                }
            except Exception as e:
                return {
                    "type": "data",
                    "body": agent_output_str,
                }
        else:
            # Run programmer
            openbbchat_output = await get_openbb_chat_output(
                query_str=query,
                auto_llama_index=app.auto_llama_index,
                node_postprocessor=app.node_postprocessor,
            )
            code_str = openbbchat_output.response.split("```python")[1].split("```")[0]
            executed_output_str = app.python_repl_utility.run(code_str)
            try:
                res = json.loads(executed_output_str)
                return {
                    "type": "data",
                    "result_data": res,
                    "body": openbbchat_output.response,
                }
            except Exception as e:
                return {
                    "type": "data",
                    "body": f"{openbbchat_output.response}\n\nResult: {executed_output_str}",
                }
    except Exception as e:
        print(e)
        return {
            "type": "error",
            "body": "Error processing the query. Please try again!",
        }


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
