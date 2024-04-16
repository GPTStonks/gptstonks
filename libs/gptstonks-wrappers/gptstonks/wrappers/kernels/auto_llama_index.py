from typing import List, Optional

from llama_index.core import (
    PromptTemplate,
    QueryBundle,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.query.schema import QueryType
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

try:
    from llama_index.llms.huggingface import HuggingFaceLLM
except Exception as e:
    print("Warning: HuggingFaceLLM not imported, so HuggingFace models are not available")
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from pydantic import BaseModel

from ..retrievers.hybrid_or_retriever import HybridORRetriever


class AutoLlamaIndex(BaseModel):
    """Wrapper around `llama-index` that fixes its possibilities to the ones needed for `openbb-
    chat`.

    Args:
        path (`str`):
            There are two possibilities, depending of the starting sequence:
            1. `files:{path_str}` (default): path to files to compute embeddings. The folder is processed recursively.
            2. `vsi:{path_str}`: path to a persisted vector store index.
        embedding_model_id (`str`):
            Name of the Embedding model to use following `llama-index` convention.
        llm_model (`str | llama_index.llms.base.LLM`):
            It can be specified in two possible ways:
            - Name of the LLM to use. For now, only OpenAI and Hugging Face models are supported.
                The model should be in the format `openai:{model_name}` or `hf:{model_name}`.
            - Instance of a `llama-index` compatible LLM, for models other than OpenAI and Hugging Face.
        context_window (`int`):
            Context window to use with Hugging Face models.
        tokenizer_name (`Optional[str]`):
            For Hugging Face models. By default set to the llm_model id.
        generate_kwargs (`Optional[dict]`):
            For Hugging Face models. These kwargs are passed directly to `AutoModelForCausalLM.generate` method.
        tokenizer_kwargs (`Optional[dict]`):
            For Hugging Face models. These kwargs are passed directly to `AutoTokenizer`.
        model_kwargs (`Optional[dict]`):
            For Hugging Face models. These kwargs are passed directly to `AutoModelForCausalLM.from_pretrained`, apart from
            `device_map` which should be specified in `other_llama_index_llm_kwargs`.
        qa_template_str (`Optional[str]`):
            String representation of the LlamaIndex's QA template to use.
        refine_template_str (`Optional[str]`):
            String representation of the LlamaIndex's refine template to use.
        retriever_type (`str`):
            One of 'hybrid', 'vector' or 'bm25'. Default 'hybrid'.
        other_llama_index_llm_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `LLM`
        other_llama_index_simple_directory_reader_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `SimpleDirectoryReader`.
        other_llama_index_storage_context_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `StorageContext.from_defaults`.
        other_llama_index_vector_store_index_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `VectorStoreIndex`.
        other_llama_index_vector_index_retriever_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `VectorIndexRetriever`.
        other_llama_index_response_synthesizer_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `get_response_synthesizer`.
        other_llama_index_retriever_query_engine_kwargs (`dict`):
            Overrides the default values in LlamaIndex's `RetrieverQueryEngine`.
    """

    def __init__(
        self,
        path: str,
        embedding_model_id: str,
        llm_model: str | LLM | None = None,
        context_window: int = 1024,
        tokenizer_name: Optional[str] = None,
        generate_kwargs: Optional[dict] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        qa_template_str: Optional[str] = None,
        refine_template_str: Optional[str] = None,
        retriever_type: str = "hybrid",
        other_llama_index_llm_kwargs: dict = {},
        other_llama_index_simple_directory_reader_kwargs: dict = {},
        other_llama_index_storage_context_kwargs: dict = {},
        other_llama_index_vector_store_index_kwargs: dict = {},
        other_llama_index_vector_index_retriever_kwargs: dict = {},
        other_llama_index_bm25_retriever_kwargs: dict = {},
        other_llama_index_response_synthesizer_kwargs: dict = {},
        other_llama_index_retriever_query_engine_kwargs: dict = {},
    ):
        """Init method."""
        super().__init__()

        # create LLM from configuration
        if llm_model:
            self._llm = self._create_llama_index_llm(
                llm_model=llm_model,
                context_window=context_window,
                generate_kwargs=generate_kwargs,
                tokenizer_name=tokenizer_name,
                tokenizer_kwargs=tokenizer_kwargs,
                model_kwargs=model_kwargs,
                other_llama_index_llm_kwargs=other_llama_index_llm_kwargs,
            )
            Settings.llm = self._llm
        else:
            self._llm = None

        # global default
        Settings.embed_model = embedding_model_id

        # create index
        self._set_index_from_path(
            path=path,
            other_llama_index_simple_directory_reader_kwargs=other_llama_index_simple_directory_reader_kwargs,
            other_llama_index_storage_context_kwargs=other_llama_index_storage_context_kwargs,
            other_llama_index_vector_store_index_kwargs=other_llama_index_vector_store_index_kwargs,
        )

        # configure retriever
        if retriever_type == "hybrid":
            vector_retriever = VectorIndexRetriever(
                index=self._index, **other_llama_index_vector_index_retriever_kwargs
            )
            bm25_retriever = BM25Retriever.from_defaults(
                index=self._index, **other_llama_index_bm25_retriever_kwargs
            )
            self._retriever = HybridORRetriever(vector_retriever, bm25_retriever)
        elif retriever_type == "vector":
            self._retriever = VectorIndexRetriever(
                index=self._index, **other_llama_index_vector_index_retriever_kwargs
            )
        elif retriever_type == "bm25":
            self._retriever = BM25Retriever.from_defaults(
                index=self._index, **other_llama_index_bm25_retriever_kwargs
            )
        else:
            raise ValueError(
                f"`retriever_type` must be 'hybrid', 'vector' or 'bm25'. Current value: {retriever_type}"
            )

        self._qa_template_str = (
            PromptTemplate(qa_template_str) if qa_template_str is not None else None
        )
        self._refine_template_str = (
            PromptTemplate(refine_template_str) if refine_template_str is not None else None
        )

        # configure response synthesizer
        self._response_synthesizer = get_response_synthesizer(
            text_qa_template=self._qa_template_str,
            refine_template=self._refine_template_str,
            **other_llama_index_response_synthesizer_kwargs,
        )

        # assemble query engine
        self._query_engine = RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=self._response_synthesizer,
            **other_llama_index_retriever_query_engine_kwargs,
        )

    @property
    def index(self) -> VectorStoreIndex:
        return self._index

    @property
    def query_engine(self) -> RetrieverQueryEngine:
        return self._query_engine

    @property
    def llm(self) -> LLM | None:
        return self._llm

    def _set_index_from_path(
        self,
        path: str,
        other_llama_index_simple_directory_reader_kwargs: dict = {},
        other_llama_index_storage_context_kwargs: dict = {},
        other_llama_index_vector_store_index_kwargs: dict = {},
    ):
        try:
            path_type, path_str = path.split(":")
        except Exception as e:
            # `:` is not present in the string, assuming `files` as default
            path_type = "files"
            path_str = path

        if path_type == "files":
            # index loaded from files in a folder
            docs_sdk = SimpleDirectoryReader(
                path_str, recursive=True, **other_llama_index_simple_directory_reader_kwargs
            ).load_data()

            # storage context to customize
            self._storage_context = StorageContext.from_defaults(
                **other_llama_index_storage_context_kwargs
            )

            # create vector store index
            self._index = VectorStoreIndex.from_documents(
                docs_sdk,
                storage_context=self._storage_context,
                **other_llama_index_vector_store_index_kwargs,
            )
        elif path_type == "vsi":
            # index loaded from a persisted index
            self._storage_context = StorageContext.from_defaults(
                persist_dir=path_str,
            )
            self._index = load_index_from_storage(storage_context=self._storage_context)
        else:
            raise ValueError(
                f"`path` type {path_type} undefined. Check documentation for valid values."
            )

    def _create_llama_index_llm(
        self,
        llm_model: str | LLM,
        context_window: int = 1024,
        generate_kwargs: Optional[dict] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        other_llama_index_llm_kwargs: dict = {},
    ) -> LLM:
        if isinstance(llm_model, str):
            llm_provider, llm_id = llm_model.split(":")
            if llm_provider == "openai":
                return OpenAI(model=llm_id, **other_llama_index_llm_kwargs)
            elif llm_provider == "hf":
                return HuggingFaceLLM(
                    context_window=context_window,
                    generate_kwargs=generate_kwargs,
                    tokenizer_name=llm_id if tokenizer_name is None else tokenizer_name,
                    model_name=llm_id,
                    tokenizer_kwargs=tokenizer_kwargs,
                    model_kwargs=model_kwargs,
                    **other_llama_index_llm_kwargs,
                )
            else:
                raise NotImplementedError(
                    f"LLM provider {llm_provider} is not implemented yet. Please check the documentation to see implemented providers."
                )
        elif isinstance(llm_model, LLM):
            return llm_model
        else:
            raise ValueError(
                f"`llm_model` is not a `str` nor a `llama_index.llms.base.LLM` object: {llm_model}"
            )

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        """Calls query method on the RetrieverQueryEngine.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle with the query to run.

        Returns:
            `llama_index.response.schema.RESPONSE_TYPE`: response from the LLM.
        """

        return self._query_engine.query(str_or_query_bundle)

    async def aquery(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        """Calls aquery method on the RetrieverQueryEngine.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle with the query to run.

        Returns:
            `llama_index.response.schema.RESPONSE_TYPE`: response from the LLM.
        """

        return await self._query_engine.aquery(str_or_query_bundle)

    def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Obtains the closest nodes to the query, computing the similarity of the query embedding
        and the index embeddings.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle to get most similar stored nodes.

        Returns:
            `List[llama_index.schema.NodeWithScore]`: list with most similar nodes and their similarity score.
        """

        return self._retriever.retrieve(str_or_query_bundle)

    async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]:
        """Obtains the closest nodes to the query, computing the similarity of the query embedding
        and the index embeddings. Async interface.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle to get most similar stored nodes.

        Returns:
            `List[llama_index.schema.NodeWithScore]`: list with most similar nodes and their similarity score.
        """

        return await self._retriever.aretrieve(str_or_query_bundle)

    def query_with_model(
        self,
        str_or_query_bundle: QueryType,
        llm_model: str | LLM,
        context_window: int = 1024,
        generate_kwargs: Optional[dict] = None,
        tokenizer_name: Optional[str] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        other_llama_index_llm_kwargs: dict = {},
        other_llama_index_response_synthesizer_kwargs={},
        other_llama_index_retriever_query_engine_kwargs={},
    ) -> RESPONSE_TYPE:
        """Calls query method on the RetrieverQueryEngine with the specified model. It is slower
        than `self.query` because the LLM, response synthesizer and query engine need to be
        rebuilt.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle with the query to run.
            llm_model (`str | llama_index.llms.base.LLM`):
                It can be specified in two possible ways:
                - Name of the LLM to use. For now, only OpenAI and Hugging Face models are supported.
                    The model should be in the format `openai:{model_name}` or `hf:{model_name}`.
                - Instance of a `llama-index` compatible LLM, for models other than OpenAI and Hugging Face.
            context_window (`int`):
                Context window to use with Hugging Face models.
            generate_kwargs (`Optional[dict]`):
                For Hugging Face models. These kwargs are passed directly to `AutoModelForCausalLM.generate` method.
            tokenizer_name (`Optional[str]`):
                For Hugging Face models. By default set to the llm_model id.
            tokenizer_kwargs (`Optional[dict]`):
                For Hugging Face models. These kwargs are passed directly to `AutoTokenizer`.
            model_kwargs (`Optional[dict]`):
                For Hugging Face models. These kwargs are passed directly to `AutoModelForCausalLM.from_pretrained`, apart from
                `device_map` which should be specified in `other_llama_index_llm_kwargs`.
            other_llama_index_llm_kwargs (`dict`):
                Overrides the default values in LlamaIndex's `LLM`.
            other_llama_index_response_synthesizer_kwargs (`dict`):
                Overrides the default values in LlamaIndex's `get_response_synthesizer`.
            other_llama_index_retriever_query_engine_kwargs (`dict`):
                Overrides the default values in LlamaIndex's `RetrieverQueryEngine`.

        Returns:
            `llama_index.response.schema.RESPONSE_TYPE`: response from the LLM.
        """

        llm = self._create_llama_index_llm(
            llm_model=llm_model,
            context_window=context_window,
            generate_kwargs=generate_kwargs,
            tokenizer_name=tokenizer_name,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
            other_llama_index_llm_kwargs=other_llama_index_llm_kwargs,
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            text_qa_template=self._qa_template_str,
            refine_template=self._refine_template_str,
            llm=llm,
            **other_llama_index_response_synthesizer_kwargs,
        )

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=response_synthesizer,
            **other_llama_index_retriever_query_engine_kwargs,
        )

        return query_engine.query(str_or_query_bundle)

    def synth(self, str_or_query_bundle: QueryType, nodes: List[NodeWithScore]) -> RESPONSE_TYPE:
        """Calls synth method on the RetrieverQueryEngine.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle with the query to run.

        Returns:
            `llama_index.response.schema.RESPONSE_TYPE`: response from the LLM.
        """

        return self._query_engine.synthesize(
            (
                QueryBundle(str_or_query_bundle)
                if isinstance(str_or_query_bundle, str)
                else str_or_query_bundle
            ),
            nodes,
        )

    async def asynth(
        self, str_or_query_bundle: QueryType, nodes: List[NodeWithScore]
    ) -> RESPONSE_TYPE:
        """Calls asynth method on the RetrieverQueryEngine.

        Args:
            str_or_query_bundle (`llama_index.indices.query.schema.QueryType`):
                String or query bundle with the query to run.

        Returns:
            `llama_index.response.schema.RESPONSE_TYPE`: response from the LLM.
        """

        return await self._query_engine.asynthesize(
            (
                QueryBundle(str_or_query_bundle)
                if isinstance(str_or_query_bundle, str)
                else str_or_query_bundle
            ),
            nodes,
        )
