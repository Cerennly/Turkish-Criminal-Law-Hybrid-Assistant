"""
RAG chain: orchestrate retriever + generator with safety mechanisms.
- Refuse to answer if retrieval score below threshold (optional strict mode).
- Handle empty results gracefully.
- Return answer plus source refs (document, page).
"""
from app.rag.retriever import retrieve
from app.rag.generator import build_chain
from app.config import SIMILARITY_SCORE_THRESHOLD

NO_ANSWER_MSG = "Bu bilgi yüklenen belgelerde yer almamaktadır."


def run_rag(question: str, refuse_below_threshold: bool = True):
    """
    Run full RAG: retrieve top-k docs, build context, generate answer.
    If refuse_below_threshold and best score < SIMILARITY_SCORE_THRESHOLD, return NO_ANSWER_MSG and no sources.
    Returns (answer: str, sources: list[dict] with keys document, page).
    """
    retrieved = retrieve(question)
    if not retrieved:
        return NO_ANSWER_MSG, []

    docs_with_scores = retrieved
    best_score = docs_with_scores[0][1] if docs_with_scores else 0.0
    # If using distance instead of similarity, check logic: for L2 distance, lower is better.
    # HuggingFaceEmbeddings + FAISS often uses similarity (higher = better). Adjust if your FAISS returns distance.
    if refuse_below_threshold and best_score < SIMILARITY_SCORE_THRESHOLD:
        return NO_ANSWER_MSG, []

    documents = [doc for doc, _ in docs_with_scores]
    context = "\n\n---\n\n".join(doc.page_content for doc in documents)
    chain = build_chain()
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    seen = set()
    for doc in documents:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"document": source, "page": page})

    return answer.strip(), sources
