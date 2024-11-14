import json
import warnings
from functools import partial

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

try:
    from pytube import Search
except ModuleNotFoundError:
    warnings.warn("PyTube needs to be installed to use YoutubeSearchTool: `pip install pytube`")


class YoutubeSearchTool(StructuredTool):
    """StructuredTool that searches videos related to a query."""

    class YoutubeSearchInput(BaseModel):
        query: str = Field(description="query for the videos to search on Youtube.")

    @classmethod
    def search_videos(cls, query: str, include_description: bool) -> str:
        """Searches videos on Youtube related to a query."""

        s = Search(query)
        # return top URL
        top_video_data = {
            "query": query,
            "top_video": s.results[0].watch_url,
            "title": s.results[0].title,
        }
        if include_description:
            top_video_data["description"] = s.results[0].description
        return json.dumps(top_video_data)

    @classmethod
    def create(
        cls,
        name: str = "YoutubeSearch",
        description: str = "Useful to search Youtube videos",
        return_direct: bool = False,
        include_description: bool = False,
    ) -> StructuredTool:
        return cls.from_function(
            func=partial(cls.search_videos, include_description=include_description),
            name=name,
            description=description,
            args_schema=cls.YoutubeSearchInput,
            return_direct=return_direct,
        )
