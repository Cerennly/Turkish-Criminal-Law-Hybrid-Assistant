import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. API ve Model Ayarları
os.environ["TAVILY_API_KEY"] = "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"

# Model takılmalarını önlemek için sıkı ayarlar
llm = OllamaLLM(
    model="llama3.2:1b", 
    temperature=0.1, 
    repeat_penalty=1.3 # 1.2'den 1.3'e çıkardım ki "beingin" demesin
)
embedding = OllamaEmbeddings(model="mxbai-embed-large")

# 2. Veritabanı Yükleme
persist_directory = "./chroma_db"
if os.path.exists(persist_directory):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})
else:
    print("Hata: chroma_db bulunamadı!")
    exit()

# 3. Araçları Hazırla (Manuel Kullanım İçin)
search_tool = TavilySearchResults(k=3)

def hukuk_asistani(soru):
    print(f"\n> İşleniyor: {soru}")
    
    # ADIM 1: Önce PDF belgelerine bak (RAG)
    print("> PDF belgeleri taranıyor...")
    docs = retriever.invoke(soru)
    pdf_context = "\n\n".join(d.page_content for d in docs)
    
    # ADIM 2: PDF'de bilgi var mı kontrol et
    # Eğer PDF çok kısa veya boş geldiyse internete sor
    if len(pdf_context.strip()) < 50:
        print("> PDF'de yeterli bilgi bulunamadı. İnternete soruluyor...")
        internet_sonuclari = search_tool.invoke(soru)
        baglam = f"İnternet Kaynakları: {internet_sonuclari}"
    else:
        print("> PDF'den bilgi alındı.")
        baglam = f"PDF Kayıtları: {pdf_context}"

    # ADIM 3: Final Cevabı Oluştur
    prompt = f"""Sen bir Türk hukukçusun. Sadece Türkçe cevap vereceksin. 
    İngilizce kelime kullanma. Türkçe karakterleri düzgün kullan.
    
    Soru: {soru}
    Bilgi: {baglam}
    
    Cevap (Kısa ve Maddeler Halinde):"""
    
    return llm.invoke(prompt)

# 4. ÇALIŞTIRMA
try:
    print("\n--- HUKUK ASİSTANI (GÜVENLİ MOD) BAŞLATILDI ---")
    query = "TCK 142 maddesine göre nitelikli hırsızlık suçunun cezası nedir?"
    
    response = hukuk_asistani(query)
    
    print("\n--- CEVAP ---")
    print(response)
    
except Exception as e:
    print(f"Bir hata oluştu: {e}")