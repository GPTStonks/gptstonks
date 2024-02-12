#!/bin/bash
# Env variables for testing. Only obligatory variables are specified.
export MONGO_URI=mongodb://localhost:27017
export MONGO_DBNAME=mongodb
export AUTOLLAMAINDEX_VSI_PATH="vsi:./gptstonks_api/data/openbb_v4.1.0_historical_vectorstoreindex_bgebaseen"
export OPENBBCHAT_TOOL_DESCRIPTION="useful to get financial and investing data. Input should be the concrete data to retrieve, in natural language."
export LLM_MODEL_ID="llamacpp:./gptstonks_api/zephyr-7b-beta.Q4_K_M.gguf"
