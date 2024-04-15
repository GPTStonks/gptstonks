from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gptstonks.api.agent import run_agent_in_background
from gptstonks.api.databases import db
from gptstonks.api.initialization import init_api
from gptstonks.api.main import app_data

# client = TestClient(app)


@pytest.mark.asyncio
async def test_run_model_bg():
    try:
        db.command("ping")
    except Exception as e:
        pytest.skip("No database to connect to. Test skipped")
    init_api(app_data)
    result_json = await run_agent_in_background("get the sentiment analysis of TSLA", app_data)

    assert "type" in result_json
    assert "body" in result_json
