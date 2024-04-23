import json

import pytest
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from gptstonks.multiagents.tools import DuckDuckGoSearchResultsJson


@pytest.mark.asyncio
async def test_ddg_json():
    search_tool = DuckDuckGoSearchResultsJson(api_wrapper=DuckDuckGoSearchAPIWrapper())

    tool_output = await search_tool.ainvoke("lorem ipsum")
    # ensure json format
    tool_dict = json.loads(tool_output)
    assert isinstance(tool_dict, dict) or isinstance(tool_dict, list)
