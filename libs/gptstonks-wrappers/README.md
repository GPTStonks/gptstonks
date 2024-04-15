<p align="center">
  <img src="../../docs/assets/logo-chatbot.png" alt="Logo">
</p>
<p align="center">
  <!-- Waitlist Badge -->
  <a href="https://gptstonks.net/login"><img src="../../docs/assets/waitlist_badge.png" alt="Join Waitlist Badge"></a>
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

# GPTStonks Wrappers

## Description

GPTStonks Wrappers provides Auto models, similar to the `transformers` library, but for common AI tools instead of models: LangChain, LlamaIndex, etc.

## Development

  1. Install [PDM](https://pdm.fming.dev/latest/#installation).

  2. Clone the project and install necessary packages:
```bash
# clone project
git clone https://github.com/GPTStonks/gptstonks.git
cd gptstonks

# install pdm
pip install pdm

# install package
pdm install -dG default
```

## Sample usage with pre-trained models

In the [API project](../../projects/gptstonks_api/), `AutoLlamaIndex` is used to perform [retrieval-augmented generation](https://arxiv.org/abs/2005.11401) (RAG) with [OpenBB](https://openbb.co)'s official documentation and with pre-trained models (e.g., OpenAI, Anthropic, Llama.cpp, etc.). Additionally, `AutoMultiStepQueryEngine` plans and executes Internet searches to solve complex queries.
