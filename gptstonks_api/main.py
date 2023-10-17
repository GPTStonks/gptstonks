import ast
import asyncio
import json
import os
import uuid

import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openbb_chat.classifiers.stransformer import STransformerZeroshotClassifier
from openbb_chat.llms.guidance_wrapper import GuidanceWrapper
from openbb_terminal.sdk import openbb
from rich.progress import track
from transformers import BitsAndBytesConfig, GPTQConfig

from .utils import (
    get_default_classifier_model,
    get_default_llm,
    get_definitions_path,
    get_definitions_sep,
    get_embeddings_path,
    get_func_parameter_names,
    get_griffin_few_shot_template,
    get_griffin_general_template,
    get_wizardcoder_few_shot_template,
)

description = """
GPTStonks API allows interacting with [OpenBB](https://openbb.co/) using natural language.

# Features
The API supports the following features:
- All LLMs on [Hugging Face](https://huggingface.co/).
- Multiple text embedding models on Hugging Face.
- Asynchronous processing.
"""

app = FastAPI(
    title="GPTStonks API",
    description=description,
    summary="API to interact with OpenBB using natural language.",
    version="0.0.1",
    contact={
        "name": "GPTStonks",
        "email": "gptstonks@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

results_store = {}


@app.on_event("startup")
def init_data():
    """Initial function called during the application startup.

    It initializes the descriptions, definitions, classifiers, and templates for guidance.
    """
    df = pd.read_csv(get_definitions_path(), sep=get_definitions_sep())
    df = df.dropna()
    app.definitions = df["Definitions"].tolist()

    with torch.inference_mode():
        app.stransformer = STransformerZeroshotClassifier(
            get_embeddings_path(),
            os.environ.get("CLASSIFIER_MODEL", get_default_classifier_model()),
        )

    llm_kwargs = {
        "device_map": {"": 0},
    }
    if "gptq" not in os.environ.get("LLM_MODEL", get_default_llm()).lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        llm_kwargs.update({"quantization_config": bnb_config})
    guidance_wrapper = GuidanceWrapper(
        model_id=os.environ.get("LLM_MODEL", get_default_llm()),
        model_kwargs=llm_kwargs,
    )

    # Code template
    code_template = get_wizardcoder_few_shot_template()
    if "griffin" in os.environ.get("LLM_MODEL", get_default_llm()).lower():
        code_template = get_griffin_few_shot_template()
    app.get_code = guidance_wrapper(code_template)

    # General knowledge template
    # For now all models use Griffin template, it should be simple enough
    general_template = get_griffin_general_template()
    app.get_answer = guidance_wrapper(general_template)


@app.post("/process_query_async")
async def process_query_async(request: Request):
    """Asynchronous endpoint to start processing the given query. The processing runs in the
    background, and a unique job ID is returned for fetching the result.

    Args:
        request (Request): FastAPI request object containing the query to be processed.

    Returns:
        dict: Contains a message confirming the start of processing and a unique job ID.
    """
    data = await request.json()
    query = data.get("query")

    job_id = str(uuid.uuid4())
    response = {"message": "Processing started", "job_id": job_id}

    asyncio.create_task(run_model_in_background(job_id, query))

    return response


async def run_model_in_background(job_id: str, query: str):
    """Background task to process the query using the STransformer and OpenBB. Stores the result in
    the global results_store dictionary.

    Args:
        job_id (str): Unique identifier for the job.
        query (str): User query to process.

    Returns:
        None
    """
    with torch.inference_mode():
        _, scores, indices = app.stransformer.rank_k(query, k=3)
        if scores[0] < 0.45:
            results_store[job_id] = {
                "type": "general_response",
                "body": app.get_answer(query=query)["answer"],
            }
            return
        func_defs = [app.definitions[idx] for idx in indices]
        func_names = [func_def.split("(")[0] for func_def in func_defs]

        code = app.get_code(func_def=func_defs[0], func_name=func_names[0], query=query)
        final_func_call = f"{func_names[0]}({code['params'].strip()})"

    # Prevent memory leakage
    torch.cuda.empty_cache()

    # Call OpenBB SDK
    try:
        res = eval(final_func_call)
        if isinstance(res, pd.DataFrame):
            res = res.dropna().to_dict(orient="dict")
        elif res is None:
            res = "Nothing returned."
        elif isinstance(res, list):
            res = pd.DataFrame(res).dropna().to_dict(orient="dict")
        else:
            res = str(res)
        results_store[job_id] = {
            "type": "data",
            "result_data": res,
            "body": code["answer"],
        }
    except Exception as e:
        print(e)
        results_store[job_id] = {
            "type": "error",
            "body": "Error processing the query. Please try again!",
        }
    finally:
        return


@app.get("/get_processing_result/{job_id}")
async def get_processing_result(job_id: str):
    """Endpoint to fetch the processing result using a job ID.

    Args:
        job_id (str): Unique identifier for the job.

    Returns:
        dict: Contains either the status of processing or the final result once completed.
    """
    result = results_store.get(job_id)
    if result is None:
        return {"status": "processing"}
    else:
        return {"status": "completed", "result": result}


@app.get("/get_api_keys")
async def get_api_keys():
    """Endpoint to retrieve the current API keys information.

    Args:
        None

    Returns:
        dict: Contains information about the current API keys.
    """

    keys = openbb.keys.get_keys_info()
    return {"result": keys}


@app.post("/set_api_keys")
async def set_api_keys(request: Request):
    """Endpoint to set new API keys. The new keys are set in the OpenBB SDK and also stored in a
    JSON file.

    Args:
        request (Request): FastAPI request object containing the keys to set.

    Returns:
        dict: Status message confirming if keys were set correctly or not.
    """

    data = await request.json()
    keys_list = data.get("keys_dict")
    persist = data.get("persist") or True
    show_output = data.get("show_output") or False
    if keys_list is not None:
        openbb.keys.set_keys(keys_dict=keys_list, persist=persist, show_output=show_output)
        with open("./apikeys_list.json", "w") as keysFile:
            json.dump(keys_list, keysFile)
        return {"status": "Keys set correctly", "keys": keys_list}
    else:
        return {"status": "Key Format was not valid. API Key list not uploaded"}
