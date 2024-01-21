import asyncio

import pytest
from httpx import AsyncClient, codes

from gptstonks_api.databases import db
from gptstonks_api.initialization import init_api
from gptstonks_api.main import app, app_data


async def run_multiple_async_queries():
    async with AsyncClient(app=app, base_url="http://localhost:8000") as ac:
        res = await asyncio.gather(
            *[
                ac.post(
                    "/process_query_async",
                    json={
                        "query": "get the balance sheets for AAPL",
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
        db.command("ping")
    except Exception as e:
        pytest.skip("No database to connect to. Test skipped")
    init_api(app_data)
    res = await run_multiple_async_queries()
    for r in res:
        assert r.status_code == codes.OK
