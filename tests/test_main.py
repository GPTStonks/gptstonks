from unittest.mock import patch

from fastapi.testclient import TestClient

from ..gptstonks_api.main import app

client = TestClient(app)


def test_get_processing_result():
    result = client.get("/get_processing_result/not_a_valid_job_id")

    assert result.status_code == 200
    assert "status" in result.json()
    assert result.json()["status"] == "processing"


@patch("asyncio.create_task")
def test_process_query_async(mocked_create_task):
    result = client.post(
        "/process_query_async", json={"query": "get the sentiment analysis of TSLA"}
    )

    assert result.status_code == 200
    mocked_create_task.assert_called_once()
    assert "job_id" in result.json()
