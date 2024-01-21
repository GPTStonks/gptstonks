import json

from ..callbacks import ToolExecutionOrderCallback
from ..databases import db
from ..explicability import add_context_to_output
from ..initialization import init_api
from ..models import AppData
from ..utils import run_repl_over_openbb


async def run_agent_in_background(query: str, app_data: AppData) -> dict:
    """Background task to process the query using the `langchain` agent.

    Args:
        query (str): User query to process.
        app_data (AppData): Objects needed to run the agent successfully.

    Returns:
        dict: Response to the query.
    """

    try:
        openbb_pat_mongo = db.tokens.find_one({}, {"_id": 0, "openbb": 1}).get("openbb")
        openbb_pat = (
            str(openbb_pat_mongo) if openbb_pat_mongo is not None else openbb_pat_mongo
        )  # Retrieve OpenBB PAT from database

        # Run agent. Best responses but high quality LLMs needed (e.g., Claude Instant or GPT-3.5)
        tool_execution_order_callback = ToolExecutionOrderCallback()
        output_str = await app_data.agent_executor.arun(
            query, callbacks=[tool_execution_order_callback]
        )
        tools_executed = tool_execution_order_callback.tools_used.copy()
        output_str = add_context_to_output(output=output_str, tools_executed=tools_executed)

        if "OpenBB" in tools_executed:
            output_str = run_repl_over_openbb(
                openbb_chat_output=output_str,
                python_repl_utility=app_data.python_repl_utility,
                openbb_pat=openbb_pat,
            )
            if "```json" in output_str:
                try:
                    result_data_str = output_str.split("```json")[1].split("```")[0].strip()
                    result_data = json.loads(result_data_str)
                    body_data_str = output_str.split("```json")[0].strip()

                    return {
                        "type": "data",
                        "result_data": result_data,
                        "body": body_data_str,
                    }
                except Exception as e:
                    return {
                        "type": "data",
                        "body": output_str,
                    }
        return {
            "type": "data",
            "body": output_str,
        }
    except Exception as e:
        print(e)
        return {"type": "error", "body": "Sorry, something went wrong!"}
