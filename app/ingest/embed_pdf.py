"""
Ingestion pipeline: load PDFs, split text, embed with bge-m3, and persist FAISS index.
Run this script once (or when documents change) to build the vector store.
"""
import sys
from pathlib import Path

# Add app to path when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)


def load_pdfs_from_data_dir() -> list:
    """
    Load all PDF files from data/ directory.
    Preserves source filename and page number in document metadata.
    """
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}. Place your 5 Turkish Criminal Law PDFs there.")
    for pdf_path in sorted(pdf_files):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for doc in pages:
            doc.metadata["source"] = pdf_path.name
            doc.metadata["page"] = doc.metadata.get("page", 0)
        documents.extend(pages)
    return documents


def split_documents(documents: list) -> list:
    """Split documents using RecursiveCharacterTextSplitter (1200 chars, 200 overlap)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_and_save_vector_store():
    """
    Full pipeline: load PDFs -> split -> embed with bge-m3 -> save FAISS to disk.
    """
    print("Loading PDFs from", DATA_DIR)
    raw_docs = load_pdfs_from_data_dir()
    print(f"Loaded {len(raw_docs)} pages from {len(list(DATA_DIR.glob('*.pdf')))} PDF(s).")
    split_docs = split_documents(raw_docs)
    print(f"Split into {len(split_docs)} chunks.")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Building FAISS index (this may take a few minutes)...")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Vector store saved to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_and_save_vector_store()
