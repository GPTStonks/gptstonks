import json
import os
from functools import partial

import gdown
import openai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.globals import set_debug
from langchain.llms import Bedrock, OpenAI, VertexAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun, YahooFinanceNewsTool
from langchain.utilities import PythonREPL, WikipediaAPIWrapper
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.llms import LangChainLLM
from llama_index.postprocessor import MetadataReplacementPostProcessor
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from openbb_terminal.sdk import openbb

from .utils import (
    arun_qa_over_tool_output,
    get_keys_file,
    get_openbb_chat_output,
    get_openbb_chat_output_executed,
    yfinance_info_titles,
)

description = """
GPTStonks API allows interacting with [OpenBB](https://openbb.co/) using natural language.

# Features
The API supports the following features:
- Bedrock LLMs.
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


@app.on_event("startup")
def init_data():
    """Initial function called during the application startup."""

    if os.getenv("DEBUG_API") is not None:
        set_debug(True)

    vsi_path = os.getenv("VSI_PATH").split(":")[-1]
    if not os.path.exists(vsi_path):
        gdown.download_folder(os.getenv("VSI_GDRIVE_URI"), output=vsi_path)
    else:
        print(f"{vsi_path} already exists, assuming it was already downloaded")

    embed_model = os.getenv("EMBEDDING_MODEL_ID", "local:BAAI/bge-large-en-v1.5")
    if embed_model == "default":
        embed_model = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
        )

    # Create LLM for both langchain agent and llama-index
    # In the future this logic could be moved to openbb-chat
    model_provider, llm_model_name = os.getenv(
        "LLM_MODEL_ID", "bedrock:anthropic.claude-instant-v1"
    ).split(":")
    llm_common_kwargs = {
        "model_name": llm_model_name,
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.1)),
        "request_timeout": float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 256)),
    }
    if model_provider == "openai":
        raise NotImplementedError(
            "OpenAI async API not working with latest llama-index and langchain"
        )
    elif model_provider == "anyscale":
        raise NotImplementedError("Anyscale does not support yet async API in langchain")
    elif model_provider == "bedrock":
        llm = Bedrock(
            credentials_profile_name=None,
            model_id=llm_model_name,
            model_kwargs={
                "temperature": llm_common_kwargs["temperature"],
                "top_p": float(os.getenv("LLM_TOP_P", 0.9)),
                "max_tokens_to_sample": llm_common_kwargs["max_tokens"],
            },
        )
    elif model_provider == "vertexai":
        llm_common_kwargs["max_output_tokens"] = llm_common_kwargs["max_tokens"]
        del llm_common_kwargs["max_tokens"]
        llm = VertexAI(location=os.getenv("LLM_CLOUD_LOCATION"), **llm_common_kwargs)
    else:
        raise NotImplementedError(f"Provider {model_provider} not implemented")
    llamaindex_llm = LangChainLLM(llm=llm)

    # Load AutoLlamaIndex
    app.auto_llama_index = AutoLlamaIndex(
        path=os.getenv("VSI_PATH"),
        embedding_model_id=embed_model,
        llm_model=llamaindex_llm,
        context_window=int(os.getenv("LLM_CONTEXT_WINDOW", 4096)),
        qa_template_str=os.getenv("QA_TEMPLATE", None),
        refine_template_str=os.getenv("REFINE_TEMPLATE", None),
        other_llama_index_vector_index_retriever_kwargs={
            "similarity_top_k": int(os.getenv("VIR_SIMILARITY_TOP_K", 3))
        },
    )

    # Create agent
    AI_PREFIX = "GPTSTONKS_RESPONSE"
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
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=0, output_key="output", ai_prefix=AI_PREFIX
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
        agent_kwargs={"ai_prefix": AI_PREFIX, "prefix": os.getenv("CUSTOM_GPTSTONKS_PREFIX")},
    )

    # Load API keys and set them in OpenBB
    if os.path.exists(get_keys_file()):
        with open(get_keys_file()) as keys_file:
            keys_list = json.load(keys_file)
        openbb.keys.set_keys(keys_dict=keys_list, persist=True, show_output=False)


@app.post("/process_query_async")
async def process_query_async(request: Request):
    """Asynchronous endpoint to start processing the given query. The processing runs in the
    background and the result is eventually returned.

    Args:
        request (Request): FastAPI request object containing the query to be processed.

    Returns:
        dict: containing the response.
    """
    data = await request.json()
    query = data.get("query")
    use_agent = data.get("use_agent", False)

    return await run_model_in_background(query, use_agent)


async def run_model_in_background(query: str, use_agent: bool) -> dict:
    """Background task to process the query using the `langchain` agent.

    Args:
        job_id (str): Unique identifier for the job.
        query (str): User query to process.
        use_agent (bool): Whether to run in Agent mode or Programmer mode.

    Returns:
        dict: Response to the query.
    """

    try:
        if use_agent:
            # Run agent. Best responses but high quality LLMs needed (e.g., Claude Instant or GPT-3.5)
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
            # Run programmer. Useful with LLMs of less quality (e.g., smaller open source LLMs)
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
        return {"type": "error", "body": "Sorry, something went wrong!"}


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
