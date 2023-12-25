import json
import os
from functools import partial

import gdown
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.globals import set_debug
from langchain.llms import Bedrock, LlamaCpp, OpenAI, VertexAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import (
    DuckDuckGoSearchAPIWrapper,
    PythonREPL,
    WikipediaAPIWrapper,
)
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.llms import LangChainLLM
from llama_index.postprocessor import MetadataReplacementPostProcessor
from openbb import obb
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex

from .utils import (
    arun_qa_over_tool_output,
    fix_frequent_code_errors,
    get_openbb_chat_output,
    get_openbb_chat_output_executed,
)

load_dotenv()

description = """
GPTStonks API allows interacting with financial data sources using natural language.

# Features
The API provides the following features to its users:
- Latest news search via [DuckDuckGo](https://duckduckgo.com/) and [Yahoo Finance](https://finance.yahoo.com/).
- Updated financial data via [OpenBB](https://openbb.co/): equities, cryptos, ETFs, currencies...
- General knowledge learned during the training of the LLM, dependable on the model.
- Run locally in an easy way with updated Docker images ([hub](https://hub.docker.com/r/gptstonks/api)).

# Supported AI models
The following models are supported:
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) optimized models: Llama 2, Mixtral, Zephyr...
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) LLMs.
- [OpenAI](https://platform.openai.com/docs/models) instruct LLMs.
- Multiple text embedding models on Hugging Face and OpenAI Ada 2 embeddings.
- [Vertex AI](https://cloud.google.com/vertex-ai) LLMs (alpha version).

# API Operating Modes
The API can operate in two modes:
- **Programmer**: only OpenBB is used to retrieve financial and economical data. It is the fastest and cheapest mode, but it is restricted to OpenBB's functionalities.
- **Agent**: different tools are used, including OpenBB and DuckDuckGo, that can look for general information and specific financial data. It is slower and slightly more expensive than Programmer, but also more powerful.
"""

app = FastAPI(
    title="GPTStonks API",
    description=description,
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
        "top_p": float(os.getenv("LLM_TOP_P", 1.0)),
    }
    if model_provider == "openai":
        app.llm = OpenAI(**llm_common_kwargs)
    elif model_provider == "anyscale":
        raise NotImplementedError("Anyscale does not support yet async API in langchain")
    elif model_provider == "bedrock":
        app.llm = Bedrock(
            credentials_profile_name=None,
            model_id=llm_common_kwargs["model_name"],
            model_kwargs={
                "temperature": llm_common_kwargs["temperature"],
                "top_p": llm_common_kwargs["top_p"],
                "max_tokens_to_sample": llm_common_kwargs["max_tokens"],
            },
        )
    elif model_provider == "vertexai":
        llm_common_kwargs["max_output_tokens"] = llm_common_kwargs["max_tokens"]
        del llm_common_kwargs["max_tokens"]
        app.llm = VertexAI(location=os.getenv("LLM_CLOUD_LOCATION"), **llm_common_kwargs)
    elif model_provider == "llamacpp":
        app.llm = LlamaCpp(
            model_path=llm_model_name,
            temperature=llm_common_kwargs["temperature"],
            max_tokens=llm_common_kwargs["max_tokens"],
            top_p=llm_common_kwargs["top_p"],
            n_ctx=int(os.getenv("LLM_LLAMACPP_CONTEXT_WINDOW")),
        )
    else:
        raise NotImplementedError(f"Provider {model_provider} not implemented")
    llamaindex_llm = LangChainLLM(llm=app.llm)

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
    app.AI_PREFIX = "GPTSTONKS_RESPONSE"
    app.python_repl_utility = PythonREPL()
    app.python_repl_utility.globals = globals()
    app.node_postprocessor = (
        MetadataReplacementPostProcessor(target_metadata_key="extra_context")
        if os.getenv("REMOVE_POSTPROCESSOR", None) is None
        else None
    )
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    app.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=0, output_key="output", ai_prefix=app.AI_PREFIX
    )
    app.tools = [
        Tool(
            name=search_tool.name,
            func=None,
            coroutine=partial(
                arun_qa_over_tool_output,
                llm=app.llm,
                tool=search_tool,
            ),
            description=search_tool.description,
            return_direct=True,
        ),
        Tool(
            name=wikipedia_tool.name,
            func=None,
            coroutine=partial(
                arun_qa_over_tool_output,
                llm=app.llm,
                tool=wikipedia_tool,
            ),
            description=wikipedia_tool.description,
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
    ]


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
    openbb_pat = data.get("openbb_pat")

    return await run_model_in_background(query, use_agent, openbb_pat)


async def run_model_in_background(query: str, use_agent: bool, openbb_pat: str | None) -> dict:
    """Background task to process the query using the `langchain` agent.

    Args:
        job_id (str): Unique identifier for the job.
        query (str): User query to process.
        use_agent (bool): Whether to run in Agent mode or Programmer mode.
        openbb_pat (str): OpenBB PAT to use with `openbb_terminal` tool.

    Returns:
        dict: Response to the query.
    """

    try:
        if use_agent:
            # update openbb tool with PAT (None if not provided)
            app.tools[-1].coroutine = partial(app.tools[-1].coroutine, openbb_pat=openbb_pat)
            # update QA tools to use original query to respond from the context
            app.tools[0].coroutine = partial(app.tools[0].coroutine, original_query=query)
            app.tools[1].coroutine = partial(app.tools[1].coroutine, original_query=query)
            agent_executor = initialize_agent(
                tools=app.tools,
                llm=app.llm,
                memory=app.memory,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                return_intermediate_steps=False,
                max_iterations=2,
                early_stopping_method=os.getenv("AGENT_EARLY_STOPPING_METHOD", "generate"),
                agent_kwargs={
                    "ai_prefix": app.AI_PREFIX,
                    "prefix": os.getenv("CUSTOM_GPTSTONKS_PREFIX"),
                },
            )
            # Run agent. Best responses but high quality LLMs needed (e.g., Claude Instant or GPT-3.5)
            agent_output_str = await agent_executor.arun(query)

            try:
                res = json.loads(agent_output_str)
                return {
                    "type": "data",
                    "result_data": res,
                    "body": "> Data retrieved using `openbb`. Use with caution.",
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
            code_str = (
                openbbchat_output.response.split("```python")[1].split("```")[0]
                if "```python" in openbbchat_output.response
                else openbbchat_output.response
            )
            executed_output_str = app.python_repl_utility.run(
                fix_frequent_code_errors(code_str, openbb_pat)
            )

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
    raise NotImplementedError("Keys are not stored, they are accessed using the PAT on-the-fly")


@app.post("/set_api_keys")
async def set_api_keys(request: Request):
    raise NotImplementedError("Keys are not stored, they are accessed using the PAT on-the-fly")
