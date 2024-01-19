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
from langchain_community.llms import Bedrock, LlamaCpp, OpenAI, VertexAI
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    PythonREPL,
    WikipediaAPIWrapper,
)
from langchain_openai import ChatOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.llms import LangChainLLM
from llama_index.postprocessor import (
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from openbb import obb
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from openbb_chat.llms.chat_model_llm_iface import ChatModelWithLLMIface
from pymongo import MongoClient

from .callbacks import ToolExecutionOrderCallback
from .explicability import add_context_to_output
from .models import TokenData
from .utils import (
    arun_qa_over_tool_output,
    fix_frequent_code_errors,
    get_openbb_chat_output,
    run_repl_over_openbb,
)

load_dotenv()

description = """
GPTStonks API allows interacting with financial data sources using natural language.

# Features
The API provides the following features to its users:
- Latest news search via [DuckDuckGo](https://duckduckgo.com/).
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

    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        app.db = client[os.getenv("MONGO_DBNAME")]
        print("Connected to MongoDB")
    except Exception as e:
        print(f"Error: {e}")
        print("Could not connect to MongoDB")

    vsi_path = os.getenv("AUTOLLAMAINDEX_VSI_PATH").split(":")[-1]
    if not os.path.exists(vsi_path):
        gdown.download_folder(os.getenv("AUTOLLAMAINDEX_VSI_GDRIVE_URI"), output=vsi_path)
    else:
        print(f"{vsi_path} already exists, assuming it was already downloaded")

    embed_model = os.getenv("AUTOLLAMAINDEX_EMBEDDING_MODEL_ID", "local:BAAI/bge-large-en-v1.5")
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
        if "instruct" in llm_model_name:
            llm = OpenAI(**llm_common_kwargs)
        else:
            top_p = llm_common_kwargs.pop("top_p")
            llm = ChatModelWithLLMIface(
                chat_model=ChatOpenAI(**llm_common_kwargs, model_kwargs={"top_p": top_p}),
                system_message=os.getenv(
                    "LLM_CHAT_MODEL_SYSTEM_MESSAGE", "You write concise and complete answers."
                ),
            )
    elif model_provider == "anyscale":
        raise NotImplementedError("Anyscale does not support yet async API in langchain")
    elif model_provider == "bedrock":
        llm = Bedrock(
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
        llm = VertexAI(location=os.getenv("LLM_CLOUD_LOCATION"), **llm_common_kwargs)
    elif model_provider == "llamacpp":
        llm = LlamaCpp(
            model_path=llm_model_name,
            temperature=llm_common_kwargs["temperature"],
            max_tokens=llm_common_kwargs["max_tokens"],
            top_p=llm_common_kwargs["top_p"],
            n_ctx=int(os.getenv("LLM_LLAMACPP_CONTEXT_WINDOW")),
        )
    else:
        raise NotImplementedError(f"Provider {model_provider} not implemented")
    llamaindex_llm = LangChainLLM(llm=llm)

    # Load AutoLlamaIndex
    app.auto_llama_index = AutoLlamaIndex(
        path=os.getenv("AUTOLLAMAINDEX_VSI_PATH"),
        embedding_model_id=embed_model,
        llm_model=llamaindex_llm,
        context_window=int(os.getenv("AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW", 4096)),
        qa_template_str=os.getenv("AUTOLLAMAINDEX_QA_TEMPLATE", None),
        refine_template_str=os.getenv("AUTOLLAMAINDEX_REFINE_TEMPLATE", None),
        other_llama_index_vector_index_retriever_kwargs={
            "similarity_top_k": int(os.getenv("AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K", 3))
        },
        use_hybrid_retriever=(not os.getenv("AUTOLLAMAINDEX_NOT_USE_HYBRID_RETRIEVER")),
    )

    # Create agent
    AI_PREFIX = "GPTSTONKS_RESPONSE"
    app.python_repl_utility = PythonREPL()
    app.python_repl_utility.globals = globals()
    app.node_postprocessors = [
        SimilarityPostprocessor(
            similarity_cutoff=os.getenv("AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF", 0.5)
        )
    ]
    if not os.getenv("AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR"):
        app.node_postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="extra_context")
        )
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool.description = os.getenv("SEARCH_TOOL_DESCRIPTION", search_tool.description)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [
        search_tool,
        wikipedia_tool,
        Tool(
            name="OpenBB",
            func=None,
            coroutine=partial(
                get_openbb_chat_output,
                auto_llama_index=app.auto_llama_index,
                node_postprocessors=app.node_postprocessors,
            ),
            description=os.getenv("OPENBBCHAT_TOOL_DESCRIPTION"),
            return_direct=True,
        ),
    ]
    app.agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=False,
        max_iterations=2,
        early_stopping_method=os.getenv("AGENT_EARLY_STOPPING_METHOD", "generate"),
        agent_kwargs={
            "ai_prefix": AI_PREFIX,
            "prefix": os.getenv("CUSTOM_GPTSTONKS_PREFIX"),
            "suffix": os.getenv("CUSTOM_GPTSTONKS_SUFFIX"),
            "input_variables": ["input", "agent_scratchpad"],
        },
    )


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
        openbb_pat (str): OpenBB PAT to use with `openbb_terminal` tool.

    Returns:
        dict: Response to the query.
    """

    try:
        openbb_pat_mongo = app.db.tokens.find_one({}, {"_id": 0, "openbb": 1}).get("openbb")
        openbb_pat = (
            str(openbb_pat_mongo) if openbb_pat_mongo is not None else openbb_pat_mongo
        )  # Retrieve OpenBB PAT from database
        print(f"Token: {openbb_pat}")
        if use_agent:
            # Run agent. Best responses but high quality LLMs needed (e.g., Claude Instant or GPT-3.5)
            tool_execution_order_callback = ToolExecutionOrderCallback()
            output_str = await app.agent_executor.arun(
                query, callbacks=[tool_execution_order_callback]
            )
            tools_executed = tool_execution_order_callback.tools_used.copy()
            output_str = add_context_to_output(output=output_str, tools_executed=tools_executed)
        else:
            # Run programmer. Useful with LLMs of less quality (e.g., smaller open source LLMs)
            # Already includes openbb context
            tools_executed = ["OpenBB"]
            output_str = await get_openbb_chat_output(
                query_str=query,
                auto_llama_index=app.auto_llama_index,
                node_postprocessors=app.node_postprocessors,
            )
        if "OpenBB" in tools_executed:
            output_str = run_repl_over_openbb(
                openbb_chat_output=output_str,
                python_repl_utility=app.python_repl_utility,
                openbb_pat=openbb_pat,
            )
            if "```json" in output_str:
                try:
                    result_data_str = output_str.split("```json")[1].split("```")[0].strip()
                    result_data = json.loads(result_data_str)
                    body_data_str = output_str.split("```json")[0].strip()

                    return {
                        "type": "data",
                        "result_data": result_data,
                        "body": body_data_str,
                    }
                except Exception as e:
                    return {
                        "type": "data",
                        "body": output_str,
                    }
        return {
            "type": "data",
            "body": output_str,
        }
    except Exception as e:
        print(e)
        return {"type": "error", "body": "Sorry, something went wrong!"}


@app.post("/tokens/")
async def update_token(token_data: TokenData):
    """Update the token used to access OpenBB.

    Args:
        token_data (TokenData): Token data.

    Returns:
        dict: Response to the query.
    """
    app.db.tokens.update_one({}, {"$set": token_data.dict()}, upsert=True)
    return {"message": "Token updated"}


@app.get("/tokens/")
async def get_token():
    """Get the token used to access OpenBB.

    Returns:
        dict: Response to the query.
    """
    token = app.db.tokens.find_one({}, {"_id": 0, "openbb": 1})
    return token if token else {"openbb": ""}
