from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import shlex
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

results_store = {}

@app.post("/process_query_async")
async def process_query_async(request: Request):
    data = await request.json()
    query = data.get('query')

    job_id = str(uuid.uuid4())
    response = {"message": "Processing started", "job_id": job_id}

    asyncio.create_task(run_command_in_background(job_id, query))

    return response

async def run_command_in_background(job_id: str, query: str):
    base_cmd = "poetry run python /openbb-chat/scripts/example_chat.py -kc /openbb-chat/data/openbb-doc-22Jul2023.csv -q "
    final_cmd = base_cmd + query

    process = await asyncio.create_subprocess_shell(final_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await process.communicate()

    stdout = stdout.decode()
    stderr = stderr.decode()
    print(f"stdout: {stdout}")
    print(f"stderr: {stderr}")

    if process.returncode != 0:
        results_store[job_id] = {"error": stderr}
    else:
        results_store[job_id] = {"result": stdout}

@app.get("/get_processing_result/{job_id}")
async def get_processing_result(job_id: str):
    result = results_store.get(job_id)
    if result is None:
        return {"status": "processing"}
    else:
        return {"status": "completed", "result": result}

