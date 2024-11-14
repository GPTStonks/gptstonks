from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain.tools import StructuredTool

from gptstonks.multiagents.tools import MitheithelSearchTool


@patch.object(
    httpx._client.AsyncClient,
    "get",
    AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "type": "string",
                "body": "string",
                "follow_up_questions": ["string"],
                "references": ["string"],
                "intermediate_steps": "string",
                "keywords": ["string"],
                "subqueries_answered": ["string"],
                "subqueries_responses": ["string"],
            },
        )
    ),
)
@pytest.mark.asyncio
async def test_mith_search():
    mith_search_tool = MitheithelSearchTool.create()
    assert isinstance(mith_search_tool, StructuredTool)
    res = await mith_search_tool.ainvoke("gptstonks")
    assert isinstance(res, str)
    assert "References:" in res
