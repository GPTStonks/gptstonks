import json
import os
import urllib.parse
import warnings
from functools import partial

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from websockets.asyncio.client import connect

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
    async def search(
        cls,
        query: str,
        timeout: float = 180,
        use_quality: bool = False,
        timelimit: str = "w",
        return_json: bool = False,
    ) -> str:
        """Searches videos on Youtube related to a query.

        Args:
            query (`str`): what to search using Mitheithel agents. It works better with full requests in natural language, instead of keywords.
            timeout (`float`): maximum time to wait for a response, in seconds. Speed takes 15-30s, and Quality 120-150s.
            use_quality (`bool`): whether to use quality or speed mode when searching.
            timelimit (`str`): how far into the past our search engine can look for data. One of d, w, m, y. Defaults to one week old.
            return_json (`bool`): whether to return the complete JSON or only the content.

        Returns:
            `str`: the complete JSON from Mitheithel API or only its content, depending on `return_json`.
        """

        params: dict = {
            "token": MITHEITHEL_API_KEY,
            "use_quality": use_quality,
            "timelimit": timelimit,
        }
        websocket_uri: str = (
            f"{MITHEITHEL_SEARCH_URI.replace('https', 'wss')}?{urllib.parse.urlencode(params)}"
        )
        async with connect(websocket_uri) as websocket:
            await websocket.send(json.dumps({"query": query}))
            mith_res: str = await websocket.recv()
        if return_json:
            return mith_res
        mith_res_data: dict = json.loads(mith_res)
        references: str = "\n".join(f"- {x}" for x in mith_res_data["references"])
        return f'Content: {mith_res_data["body"]}\nReferences:\n{references}\n'

    @classmethod
    def create(
        cls,
        name: str = "MitheithelSearch",
        description: str = "Useful to search anything on the Internet",
        return_direct: bool = False,
        timeout: float = 180,
        use_quality: bool = False,
        timelimit: str = "w",
        return_json: bool = False,
    ) -> StructuredTool:
        """Creates a LangChain tool to interact with Mitheithel API.

        Args:
            name (`str`): tool name for LangChain.
            name (`str`): the purpose of the tool, according to LangChain's documentation.
            return_direct (`bool`): Whether to return the result directly or as a callback.
            timeout (`float`): maximum time to wait for a response, in seconds. Speed takes 15-30s, and Quality 120-150s.
            use_quality (`bool`): whether to use quality or speed mode when searching.
            timelimit (`str`): how far into the past our search engine can look for data. One of d, w, m, y. Defaults to one week old.
            return_json (`bool`): whether to return the complete JSON or only the content.

        Returns:
            `StructuredTool`: the tool."""

        return cls.from_function(
            func=None,
            coroutine=partial(
                cls.search,
                timeout=timeout,
                use_quality=use_quality,
                timelimit=timelimit,
                return_json=return_json,
            ),
            name=name,
            description=description,
            args_schema=cls.MitheithelSearchInput,
            return_direct=return_direct,
        )
