# GPTStonks Chatbot API

Welcome to the GPTStonks Chatbot API documentation! This API allows you to interact with a powerful financial chatbot built on top of the `openbb` framework. Whether you're a developer looking to integrate financial chat capabilities into your application or a trader seeking automated financial insights, this API is designed to provide you with a seamless and customizable experience.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Introduction

GPTStonks is a financial chatbot powered by LLMs and enhanced with the `openbb` framework. It provides natural language conversation capabilities for financial topics, making it an ideal choice for a wide range of financial applications, including:

- Learning about the financial markets
- Improving trading strategies
- Financial news analysis: sentiment, trends, etc.
- Customer support for financial institutions

This API allows you to integrate the GPTStonks financial chatbot into your projects, enabling real-time financial chat interactions with users.

## Features

- **Real-time Financial Chat**: Engage in natural language conversations about financial topics.
- **Customizable Responses**: Tailor the chatbot's responses to suit your specific use case.
- **Easy Integration**: Built on FastAPI, this API is designed for straightforward integration into your application or platform.
- **Extensive Documentation**: Detailed documentation and examples to help you get started quickly.

## Getting Started

### Prerequisites

Before you begin, make sure you have [Docker](https://docs.docker.com/engine/install/) installed on your system.

### Installation

- **\[Recommended\] Option 1:** use the latest Docker image in [ghcr.io](https://github.com/features/packages).

  ```bash
  docker run --rm --gpus all -it -v $HOME/.cache/huggingface/:/root/.cache/huggingface -p 8000:8000 -e OPENSSL_CONF=/api/gptstonks_api/openssl.cnf ghcr.io/gptstonks/api:main
  ```

- **Option 2:** build from source.

  0. Install [PDM](https://pdm.fming.dev/latest/#installation) for a faster install.

  1. Clone this repository to your local machine:

  ```bash
  git clone https://github.com/GPTStonks/api.git
  ```

  2. Navigate to the project directory:

  ```bash
  cd gptstonks_api
  ```

  3. Install the required dependencies:

  ```bash
  pdm install --no-editable --no-self
  ```

  4. Create `openssl.cnf` to allow legacy TLS renegotiation, needed for OECD data in OpenBB.

  ```bash
  echo 'openssl_conf = openssl_init\n\
  \n\
  [openssl_init]\n\
  ssl_conf = ssl_sect\n\
  \n\
  [ssl_sect]\n\
  system_default = system_default_sect\n\
  \n\
  [system_default_sect]\n\
  Options = UnsafeLegacyRenegotiation'\
  > openssl.cnf
  ```

  5. Start the API:

  ```bash
  uvicorn gptstonks_api.main:app --host 0.0.0.0 --port 8000
  ```

Now your GPTStonks Financial Chatbot API is up and running!

## Usage

To use the GPTStonks Financial Chatbot API, send HTTP requests to the provided endpoints. You can interact with the chatbot by sending messages and receiving responses in real-time.

## API Endpoints

Check http://localhost:8000/docs once the API is started to access the endpoints' documentation.

## Contributing

We welcome contributions from the community! If you have any suggestions, bug reports, or want to contribute to the project, feel free to open issues or propose changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
