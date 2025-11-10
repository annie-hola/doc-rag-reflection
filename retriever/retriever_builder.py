from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.constants import HYBRID_RETRIEVER_WEIGHTS, VECTOR_SEARCH_K


def build_hybrid_retriever(chunks, embeddings):
    """Build a hybrid retriever using BM25 and vector-based retrieval."""
    # Create Chroma vector store
    vector_store = Chroma.from_documents(chunks, embeddings)
    # Create BM25 retriever
    bm25 = BM25Retriever.from_documents(chunks)
    # Create vector-based retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_K})
    # Combine retrievers into a hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=HYBRID_RETRIEVER_WEIGHTS
    )
    return hybrid_retriever
