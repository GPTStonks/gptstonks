import os
from functools import partial

import gdown
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.globals import set_debug
from langchain_community.llms import (
    Bedrock,
    HuggingFacePipeline,
    LlamaCpp,
    OpenAI,
    VertexAI,
)
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
from transformers import GPTQConfig

from ..constants import (
    AGENT_EARLY_STOPPING_METHOD,
    AGENT_REQUEST_TIMEOUT,
    AI_PREFIX,
    AUTOLLAMAINDEX_EMBEDDING_MODEL_ID,
    AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW,
    AUTOLLAMAINDEX_NOT_USE_HYBRID_RETRIEVER,
    AUTOLLAMAINDEX_QA_TEMPLATE,
    AUTOLLAMAINDEX_REFINE_TEMPLATE,
    AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR,
    AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF,
    AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K,
    AUTOLLAMAINDEX_VSI_GDRIVE_URI,
    AUTOLLAMAINDEX_VSI_PATH,
    CUSTOM_GPTSTONKS_PREFIX,
    CUSTOM_GPTSTONKS_SUFFIX,
    DEBUG_API,
    LLM_CHAT_MODEL_SYSTEM_MESSAGE,
    LLM_HF_BITS,
    LLM_HF_DEVICE,
    LLM_HF_DEVICE_MAP,
    LLM_HF_DISABLE_EXLLAMA,
    LLM_HF_DISABLE_SAMPLING,
    LLM_HF_TRUST_REMOTE_CODE,
    LLM_LLAMACPP_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_VERTEXAI_CLOUD_LOCATION,
    OPENBBCHAT_TOOL_DESCRIPTION,
    SEARCH_TOOL_DESCRIPTION,
)
from ..databases import db
from ..models import AppData
from ..utils import get_openbb_chat_output


def set_api_debug():
    if DEBUG_API is not None:
        set_debug(True)


def download_vsi(vsi_path: str):
    """Download Vector Store Index (VSI) if necessary."""

    if not os.path.exists(vsi_path):
        gdown.download_folder(AUTOLLAMAINDEX_VSI_GDRIVE_URI, output=vsi_path)
    else:
        print(f"{vsi_path} already exists, assuming it was already downloaded")


def load_embed_model() -> str | OpenAIEmbedding:
    """Get LlamaIndex embedding model."""

    if AUTOLLAMAINDEX_EMBEDDING_MODEL_ID == "default":
        return OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            timeout=AGENT_REQUEST_TIMEOUT,
        )
    return AUTOLLAMAINDEX_EMBEDDING_MODEL_ID


def create_openai_common_kwargs(llm_model_name: str) -> dict:
    return {
        "model_name": llm_model_name,
        "temperature": LLM_TEMPERATURE,
        "request_timeout": AGENT_REQUEST_TIMEOUT,
        "max_tokens": LLM_MAX_TOKENS,
        "top_p": LLM_TOP_P,
    }


def load_llm_model() -> LLM:
    """Initialize the Langchain LLM to use."""

    model_provider, llm_model_name = LLM_MODEL_ID.split(":")
    openai_common_kwargs = create_openai_common_kwargs(llm_model_name)
    if model_provider == "openai":
        if "instruct" in llm_model_name:
            return OpenAI(**openai_common_kwargs)
        else:
            top_p = openai_common_kwargs.pop("top_p")
            return ChatModelWithLLMIface(
                chat_model=ChatOpenAI(**openai_common_kwargs, model_kwargs={"top_p": top_p}),
                system_message=LLM_CHAT_MODEL_SYSTEM_MESSAGE,
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
        return VertexAI(location=LLM_VERTEXAI_CLOUD_LOCATION, **openai_common_kwargs)
    elif model_provider == "llamacpp":
        return LlamaCpp(
            model_path=llm_model_name,
            temperature=openai_common_kwargs["temperature"],
            max_tokens=openai_common_kwargs["max_tokens"],
            top_p=openai_common_kwargs["top_p"],
            n_ctx=LLM_LLAMACPP_CONTEXT_WINDOW,
        )
    elif model_provider == "hf":
        return HuggingFacePipeline.from_model_id(
            model_id=llm_model_name,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": openai_common_kwargs["max_tokens"],
            },
            device=LLM_HF_DEVICE,
            model_kwargs={
                "temperature": openai_common_kwargs["temperature"],
                "top_p": openai_common_kwargs["top_p"],
                "do_sample": not LLM_HF_DISABLE_SAMPLING,
                "device_map": LLM_HF_DEVICE_MAP,
                "quantization_config": GPTQConfig(
                    bits=LLM_HF_BITS,
                    disable_exllama=LLM_HF_DISABLE_EXLLAMA,
                ),
                "trust_remote_code": LLM_HF_TRUST_REMOTE_CODE,
            },
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
        description=OPENBBCHAT_TOOL_DESCRIPTION,
        return_direct=return_direct,
    )


def init_agent_tools(auto_llama_index: AutoLlamaIndex) -> list:
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF)
    ]
    if not AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR:
        node_postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="extra_context")
        )
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool.description = SEARCH_TOOL_DESCRIPTION or search_tool.description
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

    vsi_path = AUTOLLAMAINDEX_VSI_PATH.split(":")[-1]
    download_vsi(vsi_path=vsi_path)

    embed_model = load_embed_model()

    # Create LLM for both langchain agent and llama-index
    # In the future this logic could be moved to openbb-chat
    llm = load_llm_model()
    llamaindex_llm = LangChainLLM(llm=llm)

    # Load AutoLlamaIndex
    auto_llama_index = AutoLlamaIndex(
        path=AUTOLLAMAINDEX_VSI_PATH,
        embedding_model_id=embed_model,
        llm_model=llamaindex_llm,
        context_window=AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW,
        qa_template_str=AUTOLLAMAINDEX_QA_TEMPLATE,
        refine_template_str=AUTOLLAMAINDEX_REFINE_TEMPLATE,
        other_llama_index_vector_index_retriever_kwargs={
            "similarity_top_k": AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K
        },
        use_hybrid_retriever=(not AUTOLLAMAINDEX_NOT_USE_HYBRID_RETRIEVER),
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
        early_stopping_method=AGENT_EARLY_STOPPING_METHOD,
        agent_kwargs={
            "ai_prefix": AI_PREFIX,
            "prefix": CUSTOM_GPTSTONKS_PREFIX,
            "suffix": CUSTOM_GPTSTONKS_SUFFIX,
            "input_variables": ["input", "agent_scratchpad"],
        },
    )
