from unittest.mock import patch

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from gptstonks.wrappers.kernels import AutoLlamaIndex, AutoRag


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index(mocked_query, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = autollamaindex.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = autollamaindex.query(query)
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index_persistance(mocked_query, mocked_retrieve):
    # persist an index to test loading
    docs_sdk = SimpleDirectoryReader("../../docs", recursive=True).load_data()
    nodes_sdk = SentenceSplitter().get_nodes_from_documents(docs_sdk)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes_sdk)

    # create vector store index
    index = VectorStoreIndex(
        nodes_sdk,
        show_progress=True,
        embed_model="local:sentence-transformers/all-MiniLM-L6-v2",
        storage_context=storage_context,
    )
    index.storage_context.persist(persist_dir="./.vsi_cache")

    # load testing models
    autollamaindex = AutoLlamaIndex(
        "vsi:./.vsi_cache",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = autollamaindex.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = autollamaindex.query(query)
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index_query_with_model(mocked_query, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test query
    response = autollamaindex.query_with_model(query, "hf:sshleifer/tiny-gpt2")
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "synthesize")
def test_auto_llama_index_synth(mocked_synthesize, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test query
    node_list = autollamaindex.retrieve(query)
    response = autollamaindex.synth(query, node_list)
    mocked_synthesize.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index_vector_retriever(mocked_query, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        retriever_type="vector",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = autollamaindex.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = autollamaindex.query(query)
    mocked_query.assert_called_once()


@patch.object(BM25Retriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_llama_index_bm25_retriever(mocked_query, mocked_retrieve):
    # load testing models
    autollamaindex = AutoLlamaIndex(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        retriever_type="bm25",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = autollamaindex.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = autollamaindex.query(query)
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_rag(mocked_query, mocked_retrieve):
    # load testing models
    auto_rag = AutoRag(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = auto_rag.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = auto_rag.query(query)
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_rag_persistance(mocked_query, mocked_retrieve):
    # persist an index to test loading
    docs_sdk = SimpleDirectoryReader("../../docs", recursive=True).load_data()
    nodes_sdk = SentenceSplitter().get_nodes_from_documents(docs_sdk)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes_sdk)

    # create vector store index
    index = VectorStoreIndex(
        nodes_sdk,
        show_progress=True,
        embed_model="local:sentence-transformers/all-MiniLM-L6-v2",
        storage_context=storage_context,
    )
    index.storage_context.persist(persist_dir="./.vsi_cache")

    # load testing models
    auto_rag = AutoRag(
        "vsi:./.vsi_cache",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = auto_rag.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = auto_rag.query(query)
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_rag_query_with_model(mocked_query, mocked_retrieve):
    # load testing models
    auto_rag = AutoRag(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test query
    response = auto_rag.query_with_model(query, "hf:sshleifer/tiny-gpt2")
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "synthesize")
def test_auto_rag_synth(mocked_synthesize, mocked_retrieve):
    # load testing models
    auto_rag = AutoRag(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test query
    node_list = auto_rag.retrieve(query)
    response = auto_rag.synth(query, node_list)
    mocked_synthesize.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_rag_vector_retriever(mocked_query, mocked_retrieve):
    # load testing models
    auto_rag = AutoRag(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        retriever_type="vector",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = auto_rag.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = auto_rag.query(query)
    mocked_query.assert_called_once()


@patch.object(BM25Retriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
def test_auto_rag_bm25_retriever(mocked_query, mocked_retrieve):
    # load testing models
    auto_rag = AutoRag(
        "../../docs",
        "local:sentence-transformers/all-MiniLM-L6-v2",
        "hf:sshleifer/tiny-gpt2",
        retriever_type="bm25",
        context_window=100,
        other_llama_index_response_synthesizer_kwargs={"response_mode": "simple_summarize"},
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = auto_rag.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = auto_rag.query(query)
    mocked_query.assert_called_once()


@patch.object(VectorIndexRetriever, "retrieve")
@patch.object(RetrieverQueryEngine, "query")
@patch("llama_index.vector_stores.pinecone.base.PineconeVectorStore")
@patch("pinecone.control.pinecone.Pinecone")
def test_auto_rag_hybrid_pinecone(mocked_pc, mocked_pc_vs, mocked_query, mocked_retrieve):
    pc = mocked_pc(api_key="prueba")
    vector_store = mocked_pc_vs(pinecone_index=pc.Index("prueba"), add_sparse_vector=True)
    embed_model = OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002)
    llamaindex_llm = OpenAI(llm="gpt-3.5-turbo")
    auto_rag = AutoRag(
        vsi=VectorStoreIndex.from_vector_store(vector_store=vector_store),
        embedding_model_id=embed_model,
        llm_model=llamaindex_llm,
        other_llama_index_vector_index_retriever_kwargs={
            "similarity_top_k": 5,
            "vector_store_query_mode": "hybrid",
        },
        retriever_type="vector",
    )

    query = "What is the purpose of Index.md"

    # test retrieval
    node_list = auto_rag.retrieve(query)
    mocked_retrieve.assert_called_once()

    # test query
    response = auto_rag.query(query)
    mocked_query.assert_called_once()
