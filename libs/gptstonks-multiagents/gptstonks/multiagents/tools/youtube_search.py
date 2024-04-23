import json

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool

try:
    from pytube import Search
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTube needs to be installed: `pip install pytube`")


class YoutubeSearchTool(StructuredTool):
    """StructuredTool that searches videos related to a query."""

    class YoutubeSearchInput(BaseModel):
        query: str = Field(description="query for the videos to search on Youtube.")

    @classmethod
    def search_videos(cls, query: str) -> str:
        """Searches videos on Youtube related to a query."""

        s = Search(query)
        # return top URL
        return json.dumps({"query": query, "top_video": s.results[0].watch_url})

    @classmethod
    def create(
        cls,
        name: str = "YoutubeSearch",
        description: str = "Useful to search Youtube videos",
        return_direct: bool = False,
    ) -> StructuredTool:
        return cls.from_function(
            func=cls.search_videos,
            name=name,
            description=description,
            args_schema=cls.YoutubeSearchInput,
            return_direct=return_direct,
        )
