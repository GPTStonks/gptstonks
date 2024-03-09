"""## Constants defined via environment variables.

If the constant is defined with environ it means it is required, as it will returned an error if not provided.

/// tip | List of env variables
The list of env variables is provided in [Configuration with environment variables ⚙️](/api/#configuration-with-environment-variables).
///
"""
import os

from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL

MONGO_URI: str = os.environ["MONGO_URI"]
MONGO_DBNAME: str = os.environ["MONGO_DBNAME"]
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
AUTOLLAMAINDEX_VSI_PATH: str = os.environ["AUTOLLAMAINDEX_VSI_PATH"]
AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW: int = int(os.getenv("AUTOLLAMAINDEX_LLM_CONTEXT_WINDOW", 4096))
AUTOLLAMAINDEX_QA_TEMPLATE: str | None = os.getenv("AUTOLLAMAINDEX_QA_TEMPLATE")
AUTOLLAMAINDEX_REFINE_TEMPLATE: str | None = os.getenv("AUTOLLAMAINDEX_REFINE_TEMPLATE")
AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K: int = int(os.getenv("AUTOLLAMAINDEX_VIR_SIMILARITY_TOP_K", 3))
AUTOLLAMAINDEX_NOT_USE_HYBRID_RETRIEVER: str | None = os.getenv(
    "AUTOLLAMAINDEX_NOT_USE_HYBRID_RETRIEVER"
)
AGENT_REQUEST_TIMEOUT: float = float(os.getenv("AGENT_REQUEST_TIMEOUT", 20))
AGENT_EARLY_STOPPING_METHOD: str = os.getenv("AGENT_EARLY_STOPPING_METHOD", "force")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 256))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", 1.0))
LLM_MODEL_ID: str = os.environ["LLM_MODEL_ID"]
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
OPENBBCHAT_TOOL_DESCRIPTION: str = os.environ["OPENBBCHAT_TOOL_DESCRIPTION"]
SEARCH_TOOL_DESCRIPTION: str | None = os.getenv("SEARCH_TOOL_DESCRIPTION")
CUSTOM_GPTSTONKS_PREFIX: str | None = os.getenv("CUSTOM_GPTSTONKS_PREFIX")
