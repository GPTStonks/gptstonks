import os
from functools import partial

import gdown
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
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
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI
from llama_index.core import PromptTemplate as LlamaIndexPromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.core.llms.llm import LLM as LlamaIndexLLM
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SimilarityPostprocessor,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from transformers import GPTQConfig

from gptstonks.wrappers.kernels import AutoMultiStepQueryEngine, AutoRag

from ..constants import (
    AGENT_EARLY_STOPPING_METHOD,
    AGENT_REQUEST_TIMEOUT,
    AUTOLLAMAINDEX_EMBEDDING_MODEL_ID,
    AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW,
    AUTOLLAMAINDEX_QA_TEMPLATE,
    AUTOLLAMAINDEX_REFINE_TEMPLATE,
    AUTOLLAMAINDEX_REMOTE_VECTOR_STORE_API_KEY,
    AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR,
    AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF,
    AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K,
    AUTOLLAMAINDEX_VSI_GDRIVE_URI,
    AUTOLLAMAINDEX_VSI_PATH,
    AUTOMULTISTEPQUERYENGINE_INDEX_SUMMARY,
    AUTOMULTISTEPQUERYENGINE_QA_TEMPLATE,
    AUTOMULTISTEPQUERYENGINE_REFINE_TEMPLATE,
    AUTOMULTISTEPQUERYENGINE_STEPDECOMPOSE_QUERY_PROMPT,
    CUSTOM_GPTSTONKS_PREFIX,
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
    WIKIPEDIA_TOOL_DESCRIPTION,
    WORLD_KNOWLEDGE_TOOL_DESCRIPTION,
)
from ..models import AppData
from ..utils import get_openbb_chat_output


def set_api_debug():
    """Set API in debug mode."""
    if DEBUG_API is not None:
        set_debug(True)


def download_vsi(vsi_path: str):
    """Download Vector Store Index (VSI) if necessary.

    Args:
        vsi_path (`str`):
            Path to VSI. If it doesn't exist already, it is downloaded from Google Drive's URI AUTOLLAMAINDEX_VSI_GDRIVE_URI.
    """

    if not os.path.exists(vsi_path):
        gdown.download_folder(AUTOLLAMAINDEX_VSI_GDRIVE_URI, output=vsi_path)
    else:
        print(f"{vsi_path} already exists, assuming it was already downloaded")


def load_embed_model() -> str | OpenAIEmbedding:
    """Get LlamaIndex embedding model.

    Returns:
        `str | OpenAIEmbedding`: ID of the model or the LangChain's OpenAI Embedding object.
    """

    if AUTOLLAMAINDEX_EMBEDDING_MODEL_ID == "default":
        return OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            timeout=AGENT_REQUEST_TIMEOUT,
        )
    return AUTOLLAMAINDEX_EMBEDDING_MODEL_ID


def create_openai_common_kwargs(llm_model_name: str) -> dict:
    """Create common parameters for OpenAI's LLM.

    These parameters include:
    - model_name (`str`): model ID of the LLM.
    - temperature (`float`): temperature to perform the sampling.
    - request_timeout (`float`): timeout in seconds before cancelling the request to OpenAI API.
    - max_tokens (`int`): max. number of tokens to generate with the LLM.
    - top_p (`float`): top-p value to use when sampling the LLM.
    Some of these parameters are reused for other model providers.

    Args:
        llm_model_name (`str`): model ID of the LLM.

    Returns:
        `dict`: containing the common parameters.
    """
    return {
        "model_name": llm_model_name,
        "temperature": LLM_TEMPERATURE,
        "request_timeout": AGENT_REQUEST_TIMEOUT,
        "max_tokens": LLM_MAX_TOKENS,
        "top_p": LLM_TOP_P,
    }


def load_llm_model() -> LLM:
    """Initialize the Langchain LLM to use.

    Several providers are currently supported:
    - OpenAI.
    - AWS Bedrock.
    - Llama.cpp.
    - HuggingFace.
    The provider is selected and configured based on env variables.

    Returns:
        `LLM`: LangChain's LLM object for the given provider.
    """

    model_provider, llm_model_name = LLM_MODEL_ID.split(":")
    openai_common_kwargs = create_openai_common_kwargs(llm_model_name)
    if model_provider == "openai":
        if "instruct" in llm_model_name:
            return OpenAI(**openai_common_kwargs)
        else:
            top_p = openai_common_kwargs.pop("top_p")
            return ChatOpenAI(**openai_common_kwargs, model_kwargs={"top_p": top_p})
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
    auto_rag: AutoRag,
    node_postprocessors: list[BaseNodePostprocessor],
    name: str = "OpenBB",
    return_direct: bool = True,
) -> Tool:
    """Initialize OpenBB asynchronous agent tool.

    Args:
        auto_rag (`AutoRag`):
            contains the necessary objects for performing RAG (i.e., vector store, embedding model, etc.).
        node_postprocessors (`list[BaseNodePostprocessor]`):
            list of LlamaIndex's postprocessors to apply to the retrieved nodes.
        name (`str`): name of the tool.
        return_direct (`bool`):
            whether or not to return directly from this tool, without going through the agent again.

    Returns:
        `Tool`: the custom agent tool.
    """
    return Tool(
        name=name,
        func=None,
        coroutine=partial(
            get_openbb_chat_output,
            auto_rag=auto_rag,
            node_postprocessors=node_postprocessors,
        ),
        description=OPENBBCHAT_TOOL_DESCRIPTION,
        return_direct=return_direct,
    )


def init_world_knowledge_tool(
    llamaindex_llm: LlamaIndexLLM,
    name: str = "world_knowledge",
    use_openai_agent: bool = False,
    return_direct: bool = True,
    verbose: bool = False,
) -> Tool:
    """Initialize World Knowledge tool.

    The World Knowledge tool can solve complex queries by applying [multi-step reasoning](https://arxiv.org/abs/2303.09014). It has several tools available,
    which include:

    - Search: to look up information on the Internet.
    - Wikipedia: to look up information about places, people, etc.
    - Request: to look up specific webpages on the Internet.

    In each step, the LLM can select any tool (or its own knowledge) to solve the target query. The final response is generated
    by combining the responses to each subquery.

    Args:
        llamaindex_llm (`llama_index.core.llms.llm.LLM`):
            LLM that will decompose the main query and answer the subqueries.
        name (`str`): name of the tool.
        return_direct (`bool`): whether or not the tool should return when the final answer is given.
        verbose (`bool`): whether or not the tool should write to stdout the intermediate information.

    Returns:
        `list[Tool]`: list of agent tools to be used by the agent.
    """
    # Prepare tools
    search_tool = DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper())
    search_tool.description = SEARCH_TOOL_DESCRIPTION or search_tool.description
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wikipedia_tool.description = WIKIPEDIA_TOOL_DESCRIPTION or wikipedia_tool.description

    def search_tool_func(x):
        return search_tool.run(x)

    def wikipedia_tool_func(x):
        return wikipedia_tool.run(x)

    async def search_tool_async_func(x):
        return await search_tool.arun(x)

    async def wikipedia_tool_async_func(x):
        return await wikipedia_tool.arun(x)

    # Load AutoMultiStepQueryEngine
    if use_openai_agent:
        query_engine = AutoMultiStepQueryEngine.from_simple_openai_agent(
            funcs=[search_tool_func, wikipedia_tool_func],
            async_funcs=[search_tool_async_func, wikipedia_tool_async_func],
            names=[search_tool.name, wikipedia_tool.name],
            descriptions=[search_tool.description, wikipedia_tool.description],
            llm=llamaindex_llm,
            verbose=verbose,
            index_summary=AUTOMULTISTEPQUERYENGINE_INDEX_SUMMARY,
        )
    else:
        query_engine = AutoMultiStepQueryEngine.from_simple_react_agent(
            funcs=[search_tool_func, wikipedia_tool_func],
            async_funcs=[search_tool_async_func, wikipedia_tool_async_func],
            names=[search_tool.name, wikipedia_tool.name],
            descriptions=[search_tool.description, wikipedia_tool.description],
            llm=llamaindex_llm,
            verbose=verbose,
            index_summary=AUTOMULTISTEPQUERYENGINE_INDEX_SUMMARY,
        )

    # Customize the prompts
    prompts_dict = {}
    if AUTOMULTISTEPQUERYENGINE_QA_TEMPLATE:
        prompts_dict.update(
            {
                "response_synthesizer:text_qa_template": LlamaIndexPromptTemplate(
                    AUTOMULTISTEPQUERYENGINE_QA_TEMPLATE
                )
            }
        )
    if AUTOMULTISTEPQUERYENGINE_REFINE_TEMPLATE:
        prompts_dict.update(
            {
                "response_synthesizer:refine_template": LlamaIndexPromptTemplate(
                    AUTOMULTISTEPQUERYENGINE_REFINE_TEMPLATE
                )
            }
        )
    if AUTOMULTISTEPQUERYENGINE_STEPDECOMPOSE_QUERY_PROMPT:
        prompts_dict.update(
            {
                "query_transform:step_decompose_query_prompt": LlamaIndexPromptTemplate(
                    AUTOMULTISTEPQUERYENGINE_STEPDECOMPOSE_QUERY_PROMPT
                )
            }
        )
    query_engine.update_prompts(prompts_dict)

    # Return LangChain Tool
    tool_config = IndexToolConfig(
        query_engine=query_engine,
        name=name,
        description=WORLD_KNOWLEDGE_TOOL_DESCRIPTION,
        tool_kwargs={"return_direct": return_direct},
    )

    return LlamaIndexTool.from_tool_config(tool_config)


def init_agent_tools(
    embed_model: str | OpenAIEmbedding, llm: LLM, use_openai_agent: bool = False
) -> list[Tool]:
    """Initialize the agent tools.

    These tools are by default:
    - World Knowledge: a multi-step reasoning tool to answer complex queries by looking on the Internet.
    - OpenBB: custom tool to retrieve financial data using OpenBB Platform.

    Args:
        embed_model (`str | OpenAIEmbedding`):
            embedding model to use for the RAG. It should be the same as in the Vector Store Index.
        llm (`langchain_core.language_models.llms.LLM`): LLM to use inside the tools that need one.

    Returns:
        `list[Tool]`: list of agent tools to be used by the agent.
    """
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF)
    ]
    if not AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR:
        node_postprocessors.append(
            MetadataReplacementPostProcessor(target_metadata_key="extra_context")
        )

    if not use_openai_agent:
        llamaindex_llm = LangChainLLM(llm=llm)
    else:
        llamaindex_llm = LlamaIndexOpenAI(model=llm.model_name, temperature=llm.temperature)

    # Initialize connection to Pinecone
    # NOTE: Modify to use a different vector store from LlamaIndex
    pc = Pinecone(api_key=AUTOLLAMAINDEX_REMOTE_VECTOR_STORE_API_KEY)
    vector_store = PineconeVectorStore(
        pinecone_index=pc.Index(AUTOLLAMAINDEX_VSI_PATH), add_sparse_vector=True
    )
    auto_rag = AutoRag(
        vsi=VectorStoreIndex.from_vector_store(vector_store=vector_store),
        embedding_model_id=embed_model,
        llm_model=llamaindex_llm,
        context_window=AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW,
        qa_template_str=AUTOLLAMAINDEX_QA_TEMPLATE,
        refine_template_str=AUTOLLAMAINDEX_REFINE_TEMPLATE,
        other_llama_index_vector_index_retriever_kwargs={
            "similarity_top_k": AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K,
            "vector_store_query_mode": "hybrid",
        },
        retriever_type="vector",
    )

    return [
        init_world_knowledge_tool(
            llamaindex_llm=llamaindex_llm,
            use_openai_agent=use_openai_agent,
            return_direct=False,
            verbose=True,
        ),
        init_openbb_async_tool(
            auto_rag=auto_rag,
            node_postprocessors=node_postprocessors,
            return_direct=False,
        ),
    ]


def init_api(app_data: AppData):
    """Initial function called during the application startup.

    Args:
        app_data (`AppData`): global application data.
    """

    set_api_debug()

    embed_model = load_embed_model()

    # Create LLM for both langchain agent and llama-index
    # In the future this logic could be moved to openbb-chat
    llm = load_llm_model()

    # REPL to execute code
    app_data.python_repl_utility = PythonREPL()
    app_data.python_repl_utility.globals = globals()

    # Create agent
    if "openai" in LLM_MODEL_ID:
        tools = init_agent_tools(embed_model=embed_model, llm=llm, use_openai_agent=True)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    LLM_CHAT_MODEL_SYSTEM_MESSAGE,
                ),
                ("user", CUSTOM_GPTSTONKS_PREFIX or "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm_with_tools = llm.bind_tools(tools)
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
    else:
        tools = init_agent_tools(embed_model=embed_model, llm=llm, use_openai_agent=False)
        prompt = (
            PromptTemplate.from_template(CUSTOM_GPTSTONKS_PREFIX)
            if CUSTOM_GPTSTONKS_PREFIX
            else hub.pull("hwchase17/react")
        )
        agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)

    app_data.agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        early_stopping_method=AGENT_EARLY_STOPPING_METHOD,
    )
