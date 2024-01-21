API_DESCRIPTION = """GPTStonks API allows interacting with financial data sources using natural language.

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

AI_PREFIX = "GPTSTONKS_RESPONSE"
