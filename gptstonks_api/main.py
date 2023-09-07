import ast
import asyncio
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
    get_func_parameter_names,
    get_griffin_template,
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
    df = pd.read_csv(os.environ["CSV_PATH"], sep=os.environ["CSV_SEPARATOR"])
    df = df.dropna()
    app.descriptions = df["Descriptions"].tolist()
    app.definitions = df["Definitions"].tolist()

    app.keys = []
    for idx, descr in track(
        enumerate(app.descriptions), total=len(app.descriptions), description="Processing..."
    ):
        topics = app.definitions[idx][: app.definitions[idx].index("(")].split(".")[1:]
        if descr.find("[") != -1:
            descr = descr[: descr.find("[")].strip()
        if descr.strip()[-1] != ".":
            search_str = f"{descr.strip()}. Topics: {', '.join(topics)}."
        else:
            search_str = f"{descr.strip()} Topics: {', '.join(topics)}."
        app.keys.append(search_str)
    app.stransformer = STransformerZeroshotClassifier(app.keys, os.environ["CLASSIFIER_MODEL"])

    llm_kwargs = {
        "device_map": {"": 0},
    }
    if "gptq" not in os.environ["LLM_MODEL"].lower():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        llm_kwargs.update({"quantization_config": bnb_config})
    app.guidance_wrapper = GuidanceWrapper(
        model_id=os.environ["LLM_MODEL"],
        model_kwargs=llm_kwargs,
    )


@app.post("/process_query_async")
async def process_query_async(request: Request):
    data = await request.json()
    query = data.get("query")

    job_id = str(uuid.uuid4())
    response = {"message": "Processing started", "job_id": job_id}

    asyncio.create_task(run_model_in_background(job_id, query))

    return response


async def run_model_in_background(job_id: str, query: str):
    with torch.inference_mode():
        keys, _, indices = app.stransformer.rank_k(query, k=3)
        func_defs = [app.definitions[idx] for idx in indices]
        func_names = [func_def.split("(")[0] for func_def in func_defs]

        template = get_wizardcoder_few_shot_template()
        if "griffin" in os.environ["LLM_MODEL"].lower():
            template = get_griffin_template()
        get_code = app.guidance_wrapper(template)
        code = get_code(func_def=func_defs[0], func_name=func_names[0], query=query)
        final_func_call = f"{func_names[0]}({code['params'].strip()})"

    # Call OpenBB SDK
    try:
        res = eval(final_func_call)
        if isinstance(res, pd.DataFrame):
            res = res.dropna().to_dict(orient="dict")
        elif res is None:
            res = "Nothing returned."
        else:
            res = str(res)
        results_store[job_id] = {"result": res}
    except Exception as e:
        print(e)
        results_store[job_id] = {"error": "Error processing the query. Please try again!"}
    finally:
        return


@app.get("/get_processing_result/{job_id}")
async def get_processing_result(job_id: str):
    result = results_store.get(job_id)
    if result is None:
        return {"status": "processing"}
    else:
        return {"status": "completed", "result": result}
