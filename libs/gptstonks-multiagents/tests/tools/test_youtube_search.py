import json

from langchain.tools import StructuredTool

from gptstonks.multiagents.tools import YoutubeSearchTool


def test_yt_search():
    youtube_search_tool = YoutubeSearchTool.create()
    assert isinstance(youtube_search_tool, StructuredTool)
    res = json.loads(youtube_search_tool.invoke("gptstonks"))
    assert "query" in res
    assert "top_video" in res
    assert "title" in res
    assert "description" not in res


def test_yt_search_description():
    youtube_search_tool = YoutubeSearchTool.create(include_description=True)
    assert isinstance(youtube_search_tool, StructuredTool)
    res = json.loads(youtube_search_tool.invoke("gptstonks"))
    assert "query" in res
    assert "top_video" in res
    assert "title" in res
    assert "description" in res
