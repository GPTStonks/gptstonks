import json
from unittest.mock import patch

import pytest
from langchain.tools import StructuredTool
from websockets.asyncio.client import connect


@pytest.mark.asyncio
async def test_mith_search():
    with (patch.object(connect, "create_connection") as mock_connect,):
        from gptstonks.multiagents.tools import MitheithelSearchTool

        mith_search_tool = MitheithelSearchTool.create()
        assert isinstance(mith_search_tool, StructuredTool)
        with patch.object(
            json,
            "loads",
            return_value={
                "type": "string",
                "body": "string",
                "follow_up_questions": ["string"],
                "references": ["string"],
                "intermediate_steps": "string",
                "keywords": ["string"],
                "subqueries_answered": ["string"],
                "subqueries_responses": ["string"],
            },
        ):
            res = await mith_search_tool.ainvoke("gptstonks")
            assert isinstance(res, str)
            assert "References:" in res
