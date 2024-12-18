import os

import httpx
from pydantic import BaseModel, Field

MITHEITHEL_AUTH_API: str = os.getenv("MITHEITHEL_AUTH_API", "https://api.gptstonks.net")


class MitheithelAPI(BaseModel):
    """Wrapper around Mitheithel API for management calls: login, API key retrieval, etc."""

    timeout: float = Field(default=20, description="seconds for the requests to expire")
    bearer_token: str | None = Field(default=None, description="bearer token to access service")

    async def login(self, email: str, password: str) -> bool:
        """Login to Mitheithel API.

        Args:
            email (`str`): email to identify the account.
            password (`str`): Mitheithel password, or None to prompt for it.

        Returns:
            `bool`: whether or not the login is successful.
        """
        service_login_url: str = f"{MITHEITHEL_AUTH_API}/login/password?is_service_account=true"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(
                service_login_url, data={"username": email, "password": password}
            )
        self.bearer_token = (
            res.json().get("access_token") if res.status_code == httpx.codes.OK else None
        )
        return res.status_code == httpx.codes.OK

    async def register(self, email: str, password: str, name: str, language: str = "en") -> bool:
        """Register to Mitheithel API.

        Args:
            email (`str`): email to identify the account.
            password (`str`): Mitheithel password, or None to prompt for it.
            name (`str`): Mitheithel username.
            language (`str`): language to generate the responses. English by default.

        Returns:
            `bool`: whether or not the register is successful.
        """
        service_register_url: str = (
            f"{MITHEITHEL_AUTH_API}/register?plan=free&is_service_account=true"
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(
                service_register_url,
                json={"email": email, "password": password, "name": name, "language": language},
            )
        return res.status_code == httpx.codes.OK

    async def get_api_key(self) -> list[str] | None:
        """Get the API key with the bearer token to Mitheithel API.

        Returns:
            `list[str] | None`: list of API keys to use with Mitheithel API. None if the status code is not OK (e.g., not authorized).
        """
        read_service_account_url: str = f"{MITHEITHEL_AUTH_API}/service/accounts/me"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(
                read_service_account_url, headers={"Authorization": f"Bearer {self.bearer_token}"}
            )
        return res.json() if res.status_code == httpx.codes.OK else None
