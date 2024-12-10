import os
import warnings
from functools import partial

import httpx
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

try:
    MITHEITHEL_API_KEY = os.environ["MITHEITHEL_API_KEY"]
    MITHEITHEL_SEARCH_URI = os.getenv(
        "MITHEITHEL_SEARCH_URI", "https://service.gptstonks.net/service/agents/research"
    )
except KeyError:
    warnings.warn("Mitheithel API key missing, so Mitheithel API cannot be used")


class MitheithelSearchTool(StructuredTool):
    """StructuredTool that searches the Internet."""

    class MitheithelSearchInput(BaseModel):
        query: str = Field(description="query for the queries to search with Mitheithel.")

    @classmethod
    async def search(cls, query: str, timeout: float = 180) -> str:
        """Searches videos on Youtube related to a query."""

        async with httpx.AsyncClient(timeout=timeout) as client:
            mith_res = await client.get(
                url=MITHEITHEL_SEARCH_URI,
                params={
                    "query": query,
                },
                headers={"Authorization": f"Bearer {MITHEITHEL_API_KEY}"},
            )
            mith_res_data: dict = mith_res.json()
        references: str = "\n".join(f"- {x}" for x in mith_res_data["references"])
        return f'Content: {mith_res_data["body"]}\nReferences:\n{references}\n'

    @classmethod
    def create(
        cls,
        name: str = "MitheithelSearch",
        description: str = "Useful to search anything on the Internet",
        return_direct: bool = False,
        timeout: float = 180,
    ) -> StructuredTool:
        return cls.from_function(
            func=None,
            coroutine=partial(cls.search, timeout=timeout),
            name=name,
            description=description,
            args_schema=cls.MitheithelSearchInput,
            return_direct=return_direct,
        )
