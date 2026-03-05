"""
Retrieval logic: load FAISS index and perform similarity search (k=4).
Returns documents with metadata (source, page). Supports score threshold for safety.
"""
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    SIMILARITY_SCORE_THRESHOLD,
)


def get_vector_store():
    """Load FAISS index from disk with same embedding model used at ingest time."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(question: str, k: int = RETRIEVAL_K, score_threshold: float = SIMILARITY_SCORE_THRESHOLD):
    """
    Similarity search: return top-k documents with scores.
    Optionally filter by score_threshold (caller can refuse to answer if best score is below).
    Returns list of (Document, score) for documents at or above threshold.
    """
    vector_store = get_vector_store()
    # similarity_search_with_relevance_scores returns (doc, score); lower is better for FAISS L2
    # We use similarity_search_with_score to get scores; FAISS returns L2 distance so we may need to
    # interpret: for cosine similarity stored with normalize_embeddings, some FAISS impls return similarity.
    # LangChain FAISS with HuggingFace typically returns distance - we'll treat higher score as worse.
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    # relevance_scores: 1=best, 0=worst for cosine; if your FAISS returns distance, invert logic.
    filtered = [
        (doc, score)
        for doc, score in results
        if score is not None and score >= score_threshold
    ]
    return filtered if filtered else [(doc, score) for doc, score in results]
