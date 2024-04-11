"""## Constants defined manually.

They include:

- **API_DESCRIPTION:** rendered when /docs is called on the API.
- **AI_PREFIX:** prefix to use by the agent when generating the response.
"""
API_DESCRIPTION = """GPTStonks API allows interacting with financial data sources using natural language.

# Features
The API provides the following features to its users:

- Latest news search via [DuckDuckGo](https://duckduckgo.com/).
- Updated financial data via [OpenBB](https://openbb.co/): equities, cryptos, ETFs, currencies...
- General knowledge learned during the training of the LLM, dependable on the model.
- Fast local deployment with updated Docker images ([DockerHub](https://hub.docker.com/r/gptstonks/api)).

# Supported AI models
The following models are supported:

- [Llama.cpp](https://github.com/ggerganov/llama.cpp) optimized models: Llama 2, Mixtral, Zephyr...
- [HuggingFace](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) models, including quantized versions, such as Mixtral GPTQ, Phi-2, etc.
- [Amazon Bedrock](https://aws.amazon.com/bedrock/) LLMs.
- [OpenAI](https://platform.openai.com/docs/models) instruct and chat LLMs (e.g., gpt-3.5-turbo-1106 or gpt-4-turbo-preview).
- Multiple text embedding models on Hugging Face and OpenAI Ada 2 embeddings.
- [Vertex AI](https://cloud.google.com/vertex-ai) LLMs (alpha version).
"""

AI_PREFIX = "GPTSTONKS_RESPONSE"
