import os
from getpass import getpass

import httpx
from pydantic import BaseModel, Field

MITHEITHEL_AUTH_API: str = os.getenv("MITHEITHEL_AUTH_API", "https://api.gptstonks.net")
MITHEITHEL_SERVICE_API: str = os.getenv("MITHEITHEL_AUTH_API", "https://service.gptstonks.net")


class MitheithelAPI(BaseModel):
    """Wrapper around Mitheithel API for management calls: login, API key retrieval, etc."""

    timeout: float = Field(default=20, description="seconds for the requests to expire")
    bearer_token: str | None = Field(default=None, description="bearer token to access service")

    async def login(self, username: str, password: str | None = None) -> bool:
        """Login to Mitheithel API.

        Args:
            username (`str`): Mitheithel username.
            password (`str | None`): Mitheithel password, or None to prompt for it.

        Returns:
            `bool`: whether or not the login is successful.
        """
        service_login_url: str = f"{MITHEITHEL_AUTH_API}/login/password?is_service_account=true"
        if password is None:
            password = getpass(prompt="Mitheithel Account Password: ")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(
                service_login_url, json={"username": username, "password": password}
            )
        self.bearer_token = (
            res.json().get("access_token") if res.status_code == httpx.codes.OK else None
        )
        return res.status_code == httpx.codes.OK

    async def register(self, username: str, password: str | None = None) -> bool:
        """Register to Mitheithel API.

        Args:
            username (`str`): Mitheithel username.
            password (`str | None`): Mitheithel password, or None to prompt for it.

        Returns:
            `bool`: whether or not the register is successful.
        """
        service_register_url: str = (
            f"{MITHEITHEL_AUTH_API}/register?plan=free&is_service_account=true"
        )
        if password is None:
            password = getpass(prompt="Mitheithel Account Password: ")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(
                service_register_url,
                json={"username": username, "password": password},
            )
        return res.status_code == httpx.codes.OK

    async def get_api_key(self) -> list[str] | None:
        """Get the API key with the bearer token to Mitheithel API.

        Returns:
            `list[str] | None`: list of API keys to use with Mitheithel API. None if the status code is not OK (e.g., not authorized).
        """
        read_service_account_url: str = f"{MITHEITHEL_SERVICE_API}/service/accounts/me"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(
                read_service_account_url, headers={"Authorization": f"Bearer {self.bearer_token}"}
            )
        return res.json() if res.status_code == httpx.codes.OK else None
