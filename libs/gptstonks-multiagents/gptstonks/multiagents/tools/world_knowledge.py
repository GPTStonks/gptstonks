from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from llama_index.core import PromptTemplate as LlamaIndexPromptTemplate
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool
from llama_index.core.llms.llm import LLM as LlamaIndexLLM

from gptstonks.wrappers.kernels import AutoMultiStepQueryEngine

from .ddg_results_json import DuckDuckGoSearchResultsJson


class WorldKnowledgeTool(LlamaIndexTool):
    """Adds factories for the World Knowledge tool."""

    @classmethod
    def from_llamaindex_llm(
        cls,
        llamaindex_llm: LlamaIndexLLM,
        tool_description: str = "Useful to extract complex insights from the information available on the Internet.",
        name: str = "world_knowledge",
        use_openai_agent: bool = False,
        return_direct: bool = False,
        verbose: bool = False,
        search_tool_description: str | None = None,
        wikipedia_tool_description: str | None = None,
        auto_multistep_query_engine_index_summary: str = "Useful to search any information on the Internet.",
        auto_multistep_query_engine_qa_template: str | None = None,
        auto_multistep_query_engine_refine_template: str | None = None,
        auto_multistep_query_engine_stepdecompose_query_prompt: str | None = None,
    ) -> LlamaIndexTool:
        """Initialize World Knowledge tool.

        The World Knowledge tool can solve complex queries by applying [multi-step reasoning](https://arxiv.org/abs/2303.09014). It has several tools available,
        which include:

        - Search: to look up information on the Internet.
        - Wikipedia: to look up information about places, people, etc.
        - Request: to look up specific webpages on the Internet (not implemented).

        In each step, the LLM can select any tool (or its own knowledge) to solve the target query. The final response is generated
        by combining the responses to each subquery.

        Args:
            llamaindex_llm (`llama_index.core.llms.llm.LLM`):
                LLM that will decompose the main query and answer the subqueries.
            name (`str`): name of the tool.
            return_direct (`bool`): whether or not the tool should return when the final answer is given.
            verbose (`bool`): whether or not the tool should write to stdout the intermediate information.
            search_tool_description (`str | None`): description of the search tool. Defaults to the one provided by LangChain.
            wikipedia_tool_description (`str | None`): description of the Wikipedia tool. Defaults to the one provided by LangChain.
            auto_multistep_query_engine_index_summary (`str`): summary for the auto multistep query engine index. Seen internally by LlamaIndex agent, important to set how the sub-questions are asked."
            auto_multistep_query_engine_qa_template (`str | None`): template for QA in the auto multistep query engine. Defaults to the one provided by LlamaIndex.
            auto_multistep_query_engine_refine_template (`str | None`): template for refining queries in the auto multistep query engine. Defaults to the one provided by LlamaIndex.
            auto_multistep_query_engine_stepdecompose_query_prompt (`str | None`): prompt for decomposing steps in the auto multistep query engine. Defaults to the one provided by LlamaIndex.

        Returns:
            `LlamaIndexTool`: LangChain and LlamaIndex compatible tool.
        """
        # Prepare tools
        search_tool = DuckDuckGoSearchResultsJson(api_wrapper=DuckDuckGoSearchAPIWrapper())
        search_tool.description = search_tool_description or search_tool.description
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        wikipedia_tool.description = wikipedia_tool_description or wikipedia_tool.description

        def search_tool_func(x):
            return search_tool.run(x)

        def wikipedia_tool_func(x):
            return wikipedia_tool.run(x)

        async def search_tool_async_func(x):
            return await search_tool.arun(x)

        async def wikipedia_tool_async_func(x):
            return await wikipedia_tool.arun(x)

        # Load AutoMultiStepQueryEngine
        if use_openai_agent:
            query_engine = AutoMultiStepQueryEngine.from_simple_openai_agent(
                funcs=[search_tool_func, wikipedia_tool_func],
                async_funcs=[search_tool_async_func, wikipedia_tool_async_func],
                names=[search_tool.name, wikipedia_tool.name],
                descriptions=[search_tool.description, wikipedia_tool.description],
                llm=llamaindex_llm,
                verbose=verbose,
                index_summary=auto_multistep_query_engine_index_summary,
            )
        else:
            query_engine = AutoMultiStepQueryEngine.from_simple_react_agent(
                funcs=[search_tool_func, wikipedia_tool_func],
                async_funcs=[search_tool_async_func, wikipedia_tool_async_func],
                names=[search_tool.name, wikipedia_tool.name],
                descriptions=[search_tool.description, wikipedia_tool.description],
                llm=llamaindex_llm,
                verbose=verbose,
                index_summary=auto_multistep_query_engine_index_summary,
            )

        # Customize the prompts
        prompts_dict = {}
        if auto_multistep_query_engine_qa_template:
            prompts_dict.update(
                {
                    "response_synthesizer:text_qa_template": LlamaIndexPromptTemplate(
                        auto_multistep_query_engine_qa_template
                    )
                }
            )
        if auto_multistep_query_engine_refine_template:
            prompts_dict.update(
                {
                    "response_synthesizer:refine_template": LlamaIndexPromptTemplate(
                        auto_multistep_query_engine_refine_template
                    )
                }
            )
        if auto_multistep_query_engine_stepdecompose_query_prompt:
            prompts_dict.update(
                {
                    "query_transform:step_decompose_query_prompt": LlamaIndexPromptTemplate(
                        auto_multistep_query_engine_stepdecompose_query_prompt
                    )
                }
            )
        query_engine.update_prompts(prompts_dict)

        # Return LangChain Tool
        tool_config = IndexToolConfig(
            query_engine=query_engine,
            name=name,
            description=tool_description,
            tool_kwargs={"return_direct": return_direct},
        )

        return cls.from_tool_config(tool_config)
