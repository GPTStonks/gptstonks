from unittest.mock import patch

from fastapi.testclient import TestClient

from ..gptstonks_api.main import app

client = TestClient(app)


def test_process_query_async():
    result = client.post(
        "/process_query_async", json={"query": "get the sentiment analysis of TSLA"}
    )
    result_json = result.json()

    assert result.status_code == 200
    assert "type" in result_json
    assert "body" in result_json
