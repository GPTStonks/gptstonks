#!/bin/bash
# Env variables for testing. Only obligatory variables are specified.
export MONGO_URI=mongodb://localhost:27017
export MONGO_DBNAME=mongodb
export AUTOLLAMAINDEX_VSI_PATH="vsi:./gptstonks/api/data/openbb_v4.1.0_historical_vectorstoreindex_bgebaseen"
export OPENBBCHAT_TOOL_DESCRIPTION="useful to get financial and investing data. Input should be the concrete data to retrieve, in natural language."
export WORLD_KNOWLEDGE_TOOL_DESCRIPTION="useful to solve complex or incomplete financial questions and to search on the Internet current events, news and concrete financial datapoints."
export LLM_MODEL_ID="llamacpp:./gptstonks/api/zephyr-7b-beta.Q4_K_M.gguf"
