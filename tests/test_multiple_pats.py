import asyncio

import pytest
from httpx import AsyncClient, codes

from gptstonks_api.main import app, init_data


async def run_multiple_async_queries():
    async with AsyncClient(app=app, base_url="http://localhost:8000") as ac:
        res = await asyncio.gather(
            *[
                ac.post(
                    "/process_query_async",
                    json={
                        "query": "get the balance sheets for AAPL",
                        "use_agent": True,
                        "openbb_pat": str(i),
                    },
                )
                for i in range(5)
            ]
        )
    return res


@pytest.mark.asyncio
async def test_multiple_pats():
    try:
        init_data()
    except AttributeError as e:
        pytest.skip("No .env file provided, or not all env variables specified")

    res = await run_multiple_async_queries()
    for r in res:
        assert r.status_code == codes.OK
