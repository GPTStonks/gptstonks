<p align="center">
  <img src="./docs/assets/logo-chatbot.png" alt="Logo">
</p>
<p align="center">
  <!-- Waitlist Badge -->
  <a href="https://gptstonks.net/login"><img src="./docs/assets/waitlist_badge.png" alt="Join Waitlist Badge"></a>
  <!-- YT Badge -->
  <a href="https://www.youtube.com/@GPTStonks"><img src="https://img.shields.io/badge/channel-ff0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Youtube Channel Badge"></a>
  <!-- X Badge -->
  <a href="https://twitter.com/GPTStonks"><img src="https://img.shields.io/badge/follow_us-000000?style=for-the-badge&logo=x&logoColor=white" alt="X Follow Us Badge"></a>
  <!-- Discord Badge -->
  <a href="https://discord.gg/MyDDGuEd"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord Badge"></a>
  <!-- Docker Badge -->
  <a href="https://hub.docker.com/u/gptstonks">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Badge">
  </a>
</p>
<p align="center">
  <!-- Hugging Face Badge -->
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Hugging%20Face-F58025?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Badge"></a>
  <!-- LangChain Badge -->
  <a href="https://langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-005A9C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain Badge">
  </a>
  <!-- FastAPI Badge -->
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI Badge">
  </a>
  <!-- OpenBB Badge -->
  <a href="https://openbb.co/">
    <img src="https://img.shields.io/badge/OpenBB-FFA500?style=for-the-badge&logo=openbb&logoColor=white" alt="OpenBB Badge">
  </a>
</p>

# GPTStonks Core

Welcome to the GPTStonks Core documentation! This project allows you to interact with a powerful financial chatbot built on top of the latest AI and financial tools. Whether you're a developer looking to integrate financial chat capabilities into your application or a trader seeking automated financial insights, GPTStonks is designed to provide you with a seamless and customizable experience.

## Table of Contents

- [GPTStonks Core](#gptstonks-core)
  - [Table of Contents](#table-of-contents)
  - [Introduction 🌟](#introduction-)
  - [Features 🚀](#features-)
  - [Supported LLM Providers](#supported-llm-providers)
  - [Supported Embeddings Providers](#supported-embeddings-providers)
  - [Getting Started 🛠️](#getting-started-%EF%B8%8F)
  - [Contributing 🤝](#contributing-)
  - [License 📃](#license-)
  - [Disclaimer](#disclaimer)

## Introduction 🌟

GPTStonks is a financial chatbot powered by LLMs and enhanced with data frameworks. It provides natural language conversation capabilities for financial topics, making it an ideal choice for a wide range of financial applications, including:

- Learning about the financial markets
- Improving trading strategies
- Financial news analysis: sentiment, trends, etc.
- Customer support for financial institutions

## Features 🚀

- **Real-time Financial Chat**: Engage in natural language conversations about financial topics.
- **Customizable Responses**: Tailor the chatbot's responses to suit your specific use case.
- **Easy Integration**: FastAPI implementation for straightforward integration into your application or platform.
- **Extensive Documentation**: Detailed documentation and examples to help you get started quickly.

## Supported LLM Providers

- **[Llama.cpp](https://github.com/ggerganov/llama.cpp)**: optimized implementations of the most popular open source LLMs for inference over CPU and GPU. See their docs for more details on supported models, which include Mixtral, Llama 2 and Zephyr among others. Many quantized models (GGUF) can be found in Hugging Face under the user [TheBloke](https://huggingface.co/TheBloke).
- **[Amazon Bedrock](https://aws.amazon.com/bedrock/)**: foundation models from a variety of providers, including Anthropic and Amazon.
- **[OpenAI](https://platform.openai.com/docs/models)**: GPT family of foundation models, including chat and instruct versions.
- **[Vertex AI](https://cloud.google.com/vertex-ai)**: similar to Amazon Bedrock but provided by Google. This integration is in alpha version, not recommended for now.

## Supported Embeddings Providers

- **[OpenAI Embeddings](https://platform.openai.com/docs/models/embeddings)**: includes models such as Ada 2.
- **[Hugging Face](https://huggingface.co/)**: including providers such as BAAI, see their general embedding (BGE) [model list](https://huggingface.co/BAAI/bge-large-en-v1.5#model-list).

## Getting Started 🛠️

See [GPTStonks API](projects/api/README.md) to find the main implementation of the API behind https://gptstonks.net.

## Contributing 🤝

We welcome contributions from the community! If you have any suggestions, bug reports, or want to contribute to the project, feel free to open issues or propose changes.

## License 📃

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

GPTStonks Chat serves as an interface for accessing financial data and general knowledge. It is not intended to provide financial or investment advice.
