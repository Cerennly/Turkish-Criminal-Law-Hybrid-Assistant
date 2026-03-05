# Legal AI – Turkish Criminal Law (RAG)

Production-oriented local-first Legal AI assistant for Turkish Criminal Law. It ingests PDFs, uses semantic search (FAISS + bge-m3), and answers questions **only** from the provided documents to reduce hallucination.

## Features

- **RAG**: Ingest 5 Turkish Criminal Law PDFs → chunk (1200/200) → embed (bge-m3) → FAISS index.
- **Strict grounding**: System prompt instructs the model to answer only from context; otherwise respond: *"Bu bilgi yüklenen belgelerde yer almamaktadır."*
- **Safety**: Refuse to answer when retrieval score is below a threshold; log all queries.
- **API**: `POST /ask` with `question` → `answer` + `sources` (document, page).
- **Web UI**: SaaS-style arayüz; tarayıcıdan soru sorup cevap + kaynakları görebilirsiniz.

## Tech Stack

- Python 3.11, FastAPI, Uvicorn  
- LangChain (retriever, chain, prompts)  
- Ollama (local LLM)  
- FAISS (vector store), bge-m3 (embeddings)  
- Pydantic (schemas)

## Project Structure

```
legal_ai/
├── app/
│   ├── main.py          # FastAPI app, /, /ask, web UI
│   ├── static/
│   │   └── index.html   # SaaS-style web arayüzü
│   ├── config.py        # Paths, chunk size, k, threshold, Ollama settings
│   ├── rag/
│   │   ├── retriever.py # FAISS load, similarity search k=4
│   │   ├── generator.py # Ollama + strict prompt, temp=0.1, top_p=0.9
│   │   └── chain.py     # RAG orchestration, threshold check, source extraction
│   ├── ingest/
│   │   └── embed_pdf.py # Load PDFs, split, embed, save FAISS
│   └── schemas.py       # AskRequest, AskResponse, SourceRef
├── api/
│   └── ask.js           # Vercel serverless: proxy to BACKEND_URL/ask
├── public/              # (optional) static copy for reference
├── index.html           # Web arayüzü (Vercel’de / olarak sunulur)
├── vercel.json          # Vercel config
├── .vercelignore        # Vercel deploy’da hariç tutulacaklar
├── data/                # Place your 5 PDFs here
├── vector_store/        # FAISS index (created by ingest)
├── logs/                # queries.log
├── requirements.txt
└── README.md
```

## Run Instructions

### 1. Environment

```bash
cd legal_ai
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama (local LLM)

Install [Ollama](https://ollama.ai) and run a model (e.g. Llama 3.2 or a model that handles Turkish well):

```bash
ollama pull llama3.2
ollama serve
```

Set `OLLAMA_MODEL` in `app/config.py` to match (e.g. `llama3.2`).

### 3. Ingest PDFs

Put your 5 Turkish Criminal Law PDFs in `legal_ai/data/`:

```bash
# Example: copy PDFs into data/
# cp /path/to/tck.pdf data/
# cp /path/to/cmk.pdf data/
# ...
```

From the **legal_ai** directory, run the ingestion script (first run downloads bge-m3 and may take a few minutes):

```bash
cd legal_ai
python -m app.ingest.embed_pdf
```

This builds the FAISS index under `vector_store/`.

### 4. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Web arayüzü (herkes denesin):** http://localhost:8000  
- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### 5. Deploy on Vercel (linke tıklayıp herkes denesin)

Arayüzü Vercel’e deploy edip herkese tek link ile açabilirsiniz. RAG backend (Python + Ollama/FAISS) Vercel’de çalışmaz; backend’i **Railway** veya **Render**’da çalıştırıp Vercel’e bağlarsınız.

**Adım 1 – Backend’i Railway veya Render’da çalıştırın**

- **Railway:** [railway.app](https://railway.app) → New Project → Deploy from GitHub (legal_ai repo). Root Directory: `legal_ai`. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. PDF’leri ve vector store’u oluşturmak için bir kez ingestion çalıştırmanız gerekir (Railway’de run command veya local’de ingest edip vector_store’u commit/upload).
- **Render:** [render.com](https://render.com) → New Web Service → repo seçin, Root Directory: `legal_ai`, Build: `pip install -r requirements.txt`, Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

Backend’in public URL’ini kopyalayın (örn. `https://legal-ai-xxx.railway.app`).

**Adım 2 – Vercel’e deploy**

1. [vercel.com](https://vercel.com) → Add New Project → GitHub repo’yu seçin.
2. **Root Directory:** `legal_ai` yapın (veya sadece legal_ai klasörünü deploy edin).
3. **Environment Variables:** `BACKEND_URL` = backend URL’iniz (örn. `https://legal-ai-xxx.railway.app`), sonra Save.
4. Deploy’a tıklayın.

Deploy bitince Vercel size bir link verir (örn. `https://legal-ai-xxx.vercel.app`). Bu linke giren herkes arayüzü kullanabilir; sorular Vercel’deki `/api/ask` proxy üzerinden backend’inize gider.

**Not:** Sadece arayüzü Vercel’e atıp `BACKEND_URL` vermezseniz, “Backend yapılandırılmadı” uyarısı çıkar. Backend’i mutlaka Railway/Render (veya başka bir host) üzerinde çalıştırıp `BACKEND_URL` olarak eklemeniz gerekir.

### 6. Herkese link (ngrok – yerel backend)

Yerel bilgisayarınızda uvicorn çalışırken [ngrok](https://ngrok.com) ile geçici public link:

```bash
ngrok http 8000
```

Çıkan adresi paylaşın.

### 7. Example request

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"TCK 81 nedir?\"}"
```

Example response:

```json
{
  "answer": "...",
  "sources": [
    { "document": "tck.pdf", "page": 15 }
  ]
}
```

## Configuration

Edit `app/config.py`:

- **Ingestion**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`
- **Retrieval**: `RETRIEVAL_K` (default 4), `SIMILARITY_SCORE_THRESHOLD`
- **Ollama**: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `TEMPERATURE`, `TOP_P`

## Safety Mechanisms

- **Score threshold**: If the best retrieval relevance score is below `SIMILARITY_SCORE_THRESHOLD`, the API responds with *"Bu bilgi yüklenen belgelerde yer almamaktadır."* and no sources.
- **Query logging**: Every request is logged to `logs/queries.log`.
- **Empty retrieval**: Handled in the RAG chain; same refusal message and empty sources.

## License

Use for educational / internal purposes. Verify legal advice with qualified professionals.
