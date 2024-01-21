import os
from typing import List, Optional
from urllib.error import HTTPError

import yfinance as yf
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from langchain_community.utilities import PythonREPL
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from requests.exceptions import ReadTimeout


async def get_openbb_chat_output(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
) -> str:
    nodes = await auto_llama_index.aretrieve(query_str)
    if node_postprocessors is not None:
        for node_postprocessor in node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(nodes)
    return (await auto_llama_index.asynth(str_or_query_bundle=query_str, nodes=nodes)).response


def fix_frequent_code_errors(prev_code: str, openbb_pat: Optional[str] = None) -> str:
    if "import pandas as pd" not in prev_code:
        prev_code = f"import pandas as pd\n{prev_code}"
    if "obb." in prev_code and "from openbb import obb" not in prev_code:
        prev_code = f"from openbb import obb\n{prev_code}"
    # login to openbb hub from inside the REPL if a PAT is provided
    if openbb_pat is not None:
        prev_code = prev_code.replace(
            "from openbb import obb\n",
            f"from openbb import obb\nobb.account.login(pat='{openbb_pat}')\n",
            1,
        )
    # convert generic openbb output to JSON
    prev_code = f'{prev_code}\nprint(pd.DataFrame.from_records([dict(r) for r in res.results]).to_json(orient="records"))'
    return prev_code


def run_repl_over_openbb(
    openbb_chat_output: str, python_repl_utility: PythonREPL, openbb_pat: Optional[str] = None
) -> str:
    if "```python" not in openbb_chat_output:
        # no code available to execute
        return openbb_chat_output
    code_str = (
        openbb_chat_output.split("```python")[1].split("```")[0]
        if "```python" in openbb_chat_output
        else openbb_chat_output
    )
    fixed_code_str = fix_frequent_code_errors(code_str, openbb_pat)
    # run Python and get output
    repl_output = python_repl_utility.run(fixed_code_str)
    # get OpenBB's functions called for explicability
    openbb_funcs_called = set()
    for code_line in code_str.split("\n"):
        if "obb." in code_line:
            openbb_funcs_called.add(code_line.split("obb.")[1].strip())
    openbb_platform_ref_uri = "https://docs.openbb.co/platform/reference/"
    openbb_funcs_called_str = "".join(
        [
            f"- {x} [[documentation]({openbb_platform_ref_uri}{x.split('(')[0].replace('.', '/')})]\n"
            for x in openbb_funcs_called
        ]
    )

    return (
        "> Context retrieved using OpenBB. "
        f"OpenBB's functions called:\n{openbb_funcs_called_str.strip()}\n\n"
        f"```json\n{repl_output.strip()}\n```"
    )


async def get_openbb_chat_output_executed(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    python_repl_utility: PythonREPL,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    openbb_pat: Optional[str] = None,
) -> str:
    output_res = await get_openbb_chat_output(query_str, auto_llama_index, node_postprocessors)
    return run_repl_over_openbb(output_res, python_repl_utility, openbb_pat)


def run_qa_over_tool_output(tool_input: str | dict, llm: BaseLLM, tool: BaseTool) -> str:
    tool_output: str = tool.run(tool_input)
    model_prompt: str = PromptTemplate(
        input_variables=["context_str", "query_str"],
        template=os.getenv("CUSTOM_GPTSTONKS_QA", DEFAULT_TEXT_QA_PROMPT_TMPL),
    ).format(query_str=tool_input, context_str=tool_output)
    answer: str = llm(model_prompt)

    return f"> Context retrieved using {tool.name}.\n\n" f"{answer}"


async def arun_qa_over_tool_output(
    tool_input: str | dict, llm: BaseLLM, tool: BaseTool, original_query: Optional[str] = None
) -> str:
    tool_output: str = await tool.arun(tool_input)
    model_prompt = PromptTemplate(
        input_variables=["context_str", "query_str"],
        template=os.getenv("CUSTOM_GPTSTONKS_QA", DEFAULT_TEXT_QA_PROMPT_TMPL),
    )
    if original_query is not None:
        answer: str = await llm.apredict(
            model_prompt.format(query_str=original_query, context_str=tool_output)
        )
    else:
        answer: str = await llm.apredict(
            model_prompt.format(query_str=tool_input, context_str=tool_output)
        )

    return f"> Context retrieved using {tool.name}.\n\n" f"{answer}"
