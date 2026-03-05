from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from pdf_loader import load_pdfs

# 1️⃣ PDF’leri yükle
docs = load_pdfs()

# 2️⃣ Text chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = []
for doc in docs:
    chunks = splitter.split_text(doc['text'])
    for chunk in chunks:
        all_chunks.append({"file": doc['file'], "text": chunk})

# 3️⃣ Embedding modeli
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# 4️⃣ Chroma DB oluştur
db = Chroma(collection_name="legal_docs", embedding_function=embed_model)

# 5️⃣ Chunk’ları DB’ye ekle
texts = [c['text'] for c in all_chunks]
metadatas = [{"file": c["file"]} for c in all_chunks]

db.add_texts(texts=texts, metadatas=metadatas)

print(f"Added {len(texts)} chunks to Chroma DB")
