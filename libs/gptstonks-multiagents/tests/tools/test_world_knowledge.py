from llama_index.core.langchain_helpers.agents import LlamaIndexTool
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from gptstonks.multiagents.tools import WorldKnowledgeTool


def test_world_knowledge_factory():
    llamaindex_llm = LlamaIndexOpenAI(model="gpt-3.5-turbo")
    world_knowledge_tool = WorldKnowledgeTool.from_llamaindex_llm(
        llamaindex_llm=llamaindex_llm,
        use_openai_agent=True,
        return_direct=False,
        verbose=False,
    )
    assert isinstance(world_knowledge_tool, LlamaIndexTool)
