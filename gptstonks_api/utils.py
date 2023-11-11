from typing import List, Optional

from langchain.utilities import PythonREPL
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex


def get_func_parameter_names(func_def: str) -> List[str]:
    # E.g. stocks(symbol: str, time: int) would be ["symbol", "time"]
    inner_func = func_def[func_def.index("(") + 1 : func_def.index(")")].strip()
    if inner_func == "":
        return []
    return [param.strip().split(":")[0].strip() for param in inner_func.split(",")]


def get_wizardcoder_select_template():
    return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given the request "{{query}}" and the following Python functions:
{{#each func_defs_and_descrs}}- {{this.def}}: {{this.descr}}
{{/each}}
Choose the best function and use it to solve the request.

### Response:
Here is the code you asked for:
```python
import openbb
return {{select 'func' options=func_names}}({{gen 'params' stop=')'}}
```"""


def get_wizardcoder_few_shot_template():
    return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given a query and a Python function, use that function with the correct parameters to return the queried information. Here are some examples:
---
Query: load the information about Apple from 2019 to 2023
Function: openbb.stocks.load(symbol: str, start_date: Union[datetime.datetime, str, NoneType] = None, interval: int = 1440, end_date: Union[datetime.datetime, str, NoneType] = None, prepost: bool = False, source: str = "YahooFinance", weekly: bool = False, monthly: bool = False, verbose: bool = True)
Code: openbb.stocks.load(symbol="AAPL", start_date="2019-01-01", end_date="2023-01-01")
---
Query: get historical prices for Amazon from 2010-01-01 to 2023-01-01
Function: openbb.stocks.ca.hist(similar: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, candle_type: str = "a")
Code: openbb.stocks.ca.hist(similar=["AMZN"], start_date="2010-01-01", end_date="2023-01-01")
---
Query: {{query}}
Function: {{func_def}}

### Response:
Code: {{func_name}}({{gen 'params' stop=')'}}"""


def get_openllama_template():
    return """The Python function `{{func_def}}` is used to "{{func_descr}}". Given the prompt "{{query}}", write the correct parameters for the function using Python:
```python
{{param_str}}
```"""


def get_griffin_few_shot_template():
    # Template follows Griffin GPTQ format: https://huggingface.co/TheBloke/Griffin-3B-GPTQ
    return """You are a financial personal assistant called GPTStonks and you are having a conversation with a human. You are not allowed to give advice or opinions. You must decline to answer questions not related to finance. Please respond briefly, objectively and politely.
### HUMAN:
Given a query and a Python function, use that function with the correct parameters to return the queried information. Here are some examples:
1. Query: load the information about Apple from 2019 to 2023
Function: openbb.stocks.load(symbol: str, start_date: Union[datetime.datetime, str, NoneType] = None, interval: int = 1440, end_date: Union[datetime.datetime, str, NoneType] = None, prepost: bool = False, source: str = "YahooFinance", weekly: bool = False, monthly: bool = False, verbose: bool = True)
Answer: Sure! Here is the information about Apple (AAPL) from 2019-01-01 to 2023-01-01.
Code: openbb.stocks.load(symbol="AAPL", start_date="2019-01-01", end_date="2023-01-01")
2. Query: get historical prices for Amazon from 2010-01-01 to 2023-01-01
Function: openbb.stocks.ca.hist(similar: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, candle_type: str = "a")
Answer: These are the historical prices for Amazon (AMZN) for the requested dates.
Code: openbb.stocks.ca.hist(similar=["AMZN"], start_date="2010-01-01", end_date="2023-01-01")
3. Query: {{query}}
Function: {{func_def}}

### RESPONSE:
Answer: {{gen 'answer' stop='\n' temperature=0.9 top_p=0.9}}
Code: {{func_name}}({{gen 'params' stop=')'}}
"""


def get_griffin_general_template():
    return """You are a financial personal assistant called GPTStonks and you are having a conversation with a human. You are not allowed to give advice or opinions. You must decline to answer questions not related to finance. Please respond briefly, objectively and politely.
### HUMAN:
{{query}}

### RESPONSE:
{{gen 'answer' stop='.\n' temperature=0.9 top_p=0.9}}"""


def get_definitions_path():
    return "/api/gptstonks_api/data/openbb-docs-v3.2.2-funcs.csv"


def get_definitions_sep():
    return "@"


def get_embeddings_path():
    return "/api/gptstonks_api/data/openbb-docs-v3.2.2.pt"


def get_default_classifier_model():
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_default_llm():
    return "daedalus314/Griffin-3B-GPTQ"


def get_keys_file():
    return "./apikeys_list.json"


def get_openbb_chat_output(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    node_postprocessor: Optional[BaseNodePostprocessor] = None,
) -> str:
    nodes = auto_llama_index.retrieve(query_str)
    if node_postprocessor is not None:
        nodes = node_postprocessor.postprocess_nodes(nodes)
    return auto_llama_index.synth(query_str, nodes)


def get_openbb_chat_output_executed(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    python_repl_utility: PythonREPL,
    node_postprocessor: Optional[BaseNodePostprocessor] = None,
) -> str:
    output_res = get_openbb_chat_output(query_str, auto_llama_index, node_postprocessor)
    code_str = output_res.response.split("```python")[1].split("```")[0]
    return python_repl_utility.run(code_str)
