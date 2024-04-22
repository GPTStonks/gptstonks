"""## Constants defined via environment variables.

If the constant is defined with environ it means it is required, as it will returned an error if not provided.

/// tip | List of env variables
The list of env variables is provided in [Configuration with environment variables ⚙️](/api/#configuration-with-environment-variables).
///
"""

import os
import warnings

try:
    MONGO_URI: str = os.environ["MONGO_URI"]
except KeyError:
    warnings.warn("MONGO_URI env variable not provided")
    MONGO_URI = None
try:
    MONGO_DBNAME: str = os.environ["MONGO_DBNAME"]
except KeyError:
    warnings.warn("MONGO_DBNAME env variable not provided")
    MONGO_DBNAME = None
DEBUG_API: str | None = os.getenv("DEBUG_API")
AUTOLLAMAINDEX_VSI_GDRIVE_URI: str | None = os.getenv("AUTOLLAMAINDEX_VSI_GDRIVE_URI")
AUTOLLAMAINDEX_EMBEDDING_MODEL_ID: str = os.getenv(
    "AUTOLLAMAINDEX_EMBEDDING_MODEL_ID", "local:BAAI/bge-large-en-v1.5"
)
AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF: float = float(
    os.getenv("AUTOLLAMAINDEX_SIMILARITY_POSTPROCESSOR_CUTOFF", 0.5)
)
AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR: str | None = os.getenv(
    "AUTOLLAMAINDEX_REMOVE_METADATA_POSTPROCESSOR"
)
try:
    AUTOLLAMAINDEX_VSI_PATH: str = os.environ["AUTOLLAMAINDEX_VSI_PATH"]
except KeyError:
    warnings.warn("AUTOLLAMAINDEX_VSI_PATH env variable not provided")
    AUTOLLAMAINDEX_VSI_PATH = None
AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW: int = int(os.getenv("AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW", 4096))
AUTOLLAMAINDEX_QA_TEMPLATE: str | None = os.getenv("AUTOLLAMAINDEX_QA_TEMPLATE")
AUTOLLAMAINDEX_REFINE_TEMPLATE: str | None = os.getenv("AUTOLLAMAINDEX_REFINE_TEMPLATE")
AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K: int = int(os.getenv("AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K", 3))
AUTOLLAMAINDEX_RETRIEVER_TYPE: str | None = os.getenv("AUTOLLAMAINDEX_RETRIEVER_TYPE")
AUTOMULTISTEPQUERYENGINE_QA_TEMPLATE: str | None = os.getenv(
    "AUTOMULTISTEPQUERYENGINE_QA_TEMPLATE"
)
AUTOMULTISTEPQUERYENGINE_REFINE_TEMPLATE: str | None = os.getenv(
    "AUTOMULTISTEPQUERYENGINE_REFINE_TEMPLATE"
)
AUTOMULTISTEPQUERYENGINE_STEPDECOMPOSE_QUERY_PROMPT: str | None = os.getenv(
    "AUTOMULTISTEPQUERYENGINE_STEPDECOMPOSE_QUERY_PROMPT"
)
AUTOMULTISTEPQUERYENGINE_INDEX_SUMMARY: str = os.getenv(
    "AUTOMULTISTEPQUERYENGINE_INDEX_SUMMARY", "Useful to search information on the Internet."
)
AGENT_REQUEST_TIMEOUT: float = float(os.getenv("AGENT_REQUEST_TIMEOUT", 20))
AGENT_EARLY_STOPPING_METHOD: str = os.getenv("AGENT_EARLY_STOPPING_METHOD", "force")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 256))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 1.0))
try:
    LLM_MODEL_ID: str = os.environ["LLM_MODEL_ID"]
except KeyError:
    warnings.warn("LLM_MODEL_ID env variable not provided")
    LLM_MODEL_ID = None
LLM_CHAT_MODEL_SYSTEM_MESSAGE: str = os.getenv(
    "LLM_CHAT_MODEL_SYSTEM_MESSAGE", "You write concise and complete answers."
)
LLM_VERTEXAI_CLOUD_LOCATION: str | None = os.getenv("LLM_VERTEXAI_CLOUD_LOCATION")
LLM_LLAMACPP_CONTEXT_WINDOW: int | None = int(os.getenv("LLM_LLAMACPP_CONTEXT_WINDOW", 4000))
LLM_HF_DEVICE: int = int(os.getenv("LLM_HF_DEVICE", -1))
LLM_HF_DISABLE_SAMPLING: str | None = bool(os.getenv("LLM_HF_DISABLE_SAMPLING", False))
LLM_HF_DEVICE_MAP: str | None = os.getenv("LLM_HF_DEVICE_MAP")
LLM_HF_BITS: int = int(os.getenv("LLM_HF_GPTQ_BITS", 4))
LLM_HF_DISABLE_EXLLAMA: bool = bool(os.getenv("LLM_HF_DISABLE_EXLLAMA", False))
LLM_HF_TRUST_REMOTE_CODE: bool = bool(os.getenv("LLM_HF_TRUST_REMOTE_CODE"))
try:
    OPENBBCHAT_TOOL_DESCRIPTION: str = os.environ["OPENBBCHAT_TOOL_DESCRIPTION"]
except KeyError:
    warnings.warn("OPENBBCHAT_TOOL_DESCRIPTION env variable not provided")
    OPENBBCHAT_TOOL_DESCRIPTION = None
SEARCH_TOOL_DESCRIPTION: str | None = os.getenv("SEARCH_TOOL_DESCRIPTION")
WIKIPEDIA_TOOL_DESCRIPTION: str | None = os.getenv("WIKIPEDIA_TOOL_DESCRIPTION")
CUSTOM_GPTSTONKS_PREFIX: str | None = os.getenv("CUSTOM_GPTSTONKS_PREFIX")
try:
    WORLD_KNOWLEDGE_TOOL_DESCRIPTION: str = os.environ["WORLD_KNOWLEDGE_TOOL_DESCRIPTION"]
except KeyError:
    warnings.warn("WORLD_KNOWLEDGE_TOOL_DESCRIPTION env variable not provided")
    WORLD_KNOWLEDGE_TOOL_DESCRIPTION = None
