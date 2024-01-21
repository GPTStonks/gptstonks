import os
from functools import partial

import gdown
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.globals import set_debug
from langchain_community.llms import Bedrock, LlamaCpp, OpenAI, VertexAI
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    PythonREPL,
    WikipediaAPIWrapper,
)
from langchain_core.language_models.llms import LLM
from langchain_openai import ChatOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.llms import LangChainLLM
from llama_index.postprocessor import (
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index.postprocessor.types import BaseNodePostprocessor
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from openbb_chat.llms.chat_model_llm_iface import ChatModelWithLLMIface
from pymongo import MongoClient
from pymongo.database import Database

from ..constants import AI_PREFIX
from ..databases import db
from ..models import AppData
from ..utils import get_openbb_chat_output


def set_api_debug():
    if os.getenv("DEBUG_API") is not None:
        set_debug(True)


def download_vsi(vsi_path: str):
    """Download Vector Store Index (VSI) if necessary."""

    if not os.path.exists(vsi_path):
        gdown.download_folder(os.getenv("AUTOLLAMAINDEX_VSI_GDRIVE_URI"), output=vsi_path)
    else:
        print(f"{vsi_path} already exists, assuming it was already downloaded")


def load_embed_model() -> str | OpenAIEmbedding:
    """Get LlamaIndex embedding model."""

    embed_model = os.getenv("AUTOLLAMAINDEX_EMBEDDING_MODEL_ID", "local:BAAI/bge-large-en-v1.5")
    if embed_model == "default":
        embed_model = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            timeout=float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
        )
    return embed_model


def create_openai_common_kwargs(llm_model_name: str) -> dict:
    return {
        "model_name": llm_model_name,
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.1)),
        "request_timeout": float(os.getenv("AGENT_REQUEST_TIMEOUT", 20)),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", 256)),
        "top_p": float(os.getenv("LLM_TOP_P", 1.0)),
    }


def load_llm_model() -> LLM:
    """Initialize the Langchain LLM to use."""

    model_provider, llm_model_name = os.getenv(
        "LLM_MODEL_ID", "bedrock:anthropic.claude-instant-v1"
    ).split(":")
    openai_common_kwargs = create_openai_common_kwargs(llm_model_name)
    if model_provider == "openai":
        if "instruct" in llm_model_name:
            return OpenAI(**openai_common_kwargs)
        else:
            top_p = openai_common_kwargs.pop("top_p")
            return ChatModelWithLLMIface(
                chat_model=ChatOpenAI(**openai_common_kwargs, model_kwargs={"top_p": top_p}),
                system_message=os.getenv(
                    "LLM_CHAT_MODEL_SYSTEM_MESSAGE", "You write concise and complete answers."
                ),
            )
    elif model_provider == "anyscale":
        raise NotImplementedError("Anyscale does not support yet async API in langchain")
    elif model_provider == "bedrock":
        return Bedrock(
            credentials_profile_name=None,
            model_id=openai_common_kwargs["model_name"],
            model_kwargs={
                "temperature": openai_common_kwargs["temperature"],
                "top_p": openai_common_kwargs["top_p"],
                "max_tokens_to_sample": openai_common_kwargs["max_tokens"],
            },
        )
    elif model_provider == "vertexai":
        openai_common_kwargs["max_output_tokens"] = openai_common_kwargs.pop("max_tokens")
        return VertexAI(location=os.getenv("LLM_CLOUD_LOCATION"), **openai_common_kwargs)
    elif model_provider == "llamacpp":
        return LlamaCpp(
            model_path=llm_model_name,
            temperature=openai_common_kwargs["temperature"],
            max_tokens=openai_common_kwargs["max_tokens"],
            top_p=openai_common_kwargs["top_p"],
            n_ctx=int(os.getenv("LLM_LLAMACPP_CONTEXT_WINDOW")),
        )
    else:
        raise NotImplementedError(f"Provider {model_provider} not implemented")


def init_openbb_async_tool(
    auto_llama_index: AutoLlamaIndex,
    node_postprocessors: list[BaseNodePostprocessor],
    name: str = "OpenBB",
    return_direct: bool = True,
) -> Tool:
    return Tool(
        name=name,
        func=None,
        coroutine=partial(
            get_openbb_chat_output,
            auto_llama_index=auto_llama_index,
            node_postprocessors=node_postprocessors,
        ),
        description=os.getenv("OPENBBCHAT_TOOL_DESCRIPTION"),
        return_direct=return_direct,
    )


def init_agent_tools(auto_llama_index: AutoLlamaIndex) -> list:
    node_postprocessors = [
        SimilarityPostprocessor(
            similarity_cutoff=os.getenv("AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF", 0.5)
        )
    ]
    if not os.getenv("AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR"):
        node_postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="extra_context")
        )
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool.description = os.getenv("SEARCH_TOOL_DESCRIPTION", search_tool.description)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return [
        search_tool,
        wikipedia_tool,
        init_openbb_async_tool(
            auto_llama_index=auto_llama_index, node_postprocessors=node_postprocessors
        ),
    ]


def init_api(app_data: AppData):
    """Initial function called during the application startup."""

    set_api_debug()

    vsi_path = os.getenv("AUTOLLAMAINDEX_VSI_PATH").split(":")[-1]
    download_vsi(vsi_path=vsi_path)

    embed_model = load_embed_model()

    # Create LLM for both langchain agent and llama-index
    # In the future this logic could be moved to openbb-chat
    llm = load_llm_model()
    llamaindex_llm = LangChainLLM(llm=llm)

    # Load AutoLlamaIndex
    auto_llama_index = AutoLlamaIndex(
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
    # REPL to execute code
    app_data.python_repl_utility = PythonREPL()
    app_data.python_repl_utility.globals = globals()

    # Create agent
    app_data.agent_executor = initialize_agent(
        tools=init_agent_tools(auto_llama_index=auto_llama_index),
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
