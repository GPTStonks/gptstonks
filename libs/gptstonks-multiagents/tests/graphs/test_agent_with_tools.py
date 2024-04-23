from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from gptstonks.multiagents.graphs import GraphAgentWithTools
from gptstonks.multiagents.tools import WorldKnowledgeTool, YoutubeSearchTool


def test_agent_with_tools_def():
    # define tools
    world_knowledge_tool = WorldKnowledgeTool.from_llamaindex_llm(
        llamaindex_llm=LlamaIndexOpenAI(model="gpt-3.5-turbo-0125", temperature=0),
        use_openai_agent=True,
        verbose=True,
    )
    youtube_search_tool = YoutubeSearchTool.create()

    # define graph
    graph = GraphAgentWithTools(
        model=ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0, max_tokens=2048, model_kwargs={"top_p": 0.8}
        ),
        tools=[world_knowledge_tool, youtube_search_tool],
        prompt_main_agent=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Forget everything you knew. You are an expert in creating music playlists for a wide variety of users. You provide very detailed and clearly structured answers. You always start by searching with world knowledge.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        ),
    ).define_basic_graph()
