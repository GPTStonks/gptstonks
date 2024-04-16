from llama_index.core.indices.query.schema import QueryType
from llama_index.core.retrievers import BaseRetriever


class HybridORRetriever(BaseRetriever):
    """Hybrid retriever based on the OR of two retrievers. Based on https://gpt-
    index.readthedocs.io/en/latest/examples/retrievers/bm25_retriever.html#custom-retriever-
    implementation.

    Args:
        retriever1 (`BaseRetriever`):
            First retriever to use. It must extend `llama_index.retrievers.BaseRetriever`.
        retriever2 (`BaseRetriever`):
            Second retriever to use. It must extend `llama_index.retrievers.BaseRetriever`.
    """

    def __init__(self, retriever1: BaseRetriever, retriever2: BaseRetriever):
        self.retriever1 = retriever1
        self.retriever2 = retriever2

    def _retrieve(self, query: QueryType, **kwargs):
        """Override `_retrieve` from `BaseRetriever`."""
        nodes2 = self.retriever2.retrieve(query, **kwargs)
        nodes1 = self.retriever1.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in nodes2 + nodes1:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
