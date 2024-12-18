from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gptstonks.wrappers.kernels import MitheithelAPI


@patch.object(
    httpx._client.AsyncClient,
    "post",
    AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "message": "OK",
            },
        )
    ),
)
@pytest.mark.asyncio
async def test_mitheithel_register():
    mitheithel_api = MitheithelAPI()
    assert await mitheithel_api.register(
        email="whatever@e.com", password="1234", name="whatever"
    )  # nosec B106


@patch.object(
    httpx._client.AsyncClient,
    "post",
    AsyncMock(
        return_value=httpx.Response(
            401,
        )
    ),
)
@pytest.mark.asyncio
async def test_mitheithel_register_fail():
    mitheithel_api = MitheithelAPI()
    assert not await mitheithel_api.register(
        email="whatever@e.com", password="1234", name="whatever"
    )  # nosec B106


@patch.object(
    httpx._client.AsyncClient,
    "post",
    AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "type": "bearer",
                "access_token": "sometoken",
            },
        )
    ),
)
@pytest.mark.asyncio
async def test_mitheithel_login():
    mitheithel_api = MitheithelAPI()
    assert await mitheithel_api.login(email="whatever", password="1234")  # nosec B106


@patch.object(
    httpx._client.AsyncClient,
    "post",
    AsyncMock(
        return_value=httpx.Response(
            401,
        )
    ),
)
@pytest.mark.asyncio
async def test_mitheithel_login_unauthorized():
    mitheithel_api = MitheithelAPI()
    assert not await mitheithel_api.login(email="whatever", password="1234")  # nosec B106
