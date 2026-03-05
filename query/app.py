import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
#from langchain_tavily import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults

# --- AYARLAR VE API ---
os.environ["TAVILY_API_KEY"] = "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"

st.set_page_config(page_title="Hukuk AI Asistanı", page_icon="⚖️")
st.title("⚖️ Türk Ceza Hukuku Hibrit Asistanı")
st.caption("PDF Belgeleri + Canlı İnternet Araması")

# --- SİSTEMİ YÜKLE (MODEL VE VERİTABANI) ---
@st.cache_resource
def load_system():
    # 1. Modeller (Takılmaları önlemek için ayarlandı)
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    llm = OllamaLLM(
        model="llama3.2:3b", 
        temperature=0.1,
        repeat_penalty=1.3 # Kelime tekrarını ve saçmalamayı önler
    )
    
    # 2. Vektör Veritabanı
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = db.as_retriever(search_kwargs={"k": 3})
    else:
        st.error("Veritabanı bulunamadı! Lütfen önce PDF'leri sisteme yükleyin.")
        st.stop()
    
    # 3. İnternet Arama Aracı
    search_tool = TavilySearchResults(k=3)
    #search_tool = TavilySearchResults(max_results=3) # 'k' yerine 'max_results' kullanmak daha günceldir
    
    return llm, retriever, search_tool

llm, retriever, search_tool = load_system()

# --- SOHBET ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt_input := st.chat_input("Hukuki sorunuzu yazın..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Yapay Zeka Yanıt Süreci
    with st.chat_message("assistant"):
        # 1. Adım: PDF Kontrolü
        with st.status("🔍 Bilgi kaynakları taranıyor...", expanded=True) as status:
            st.write("📂 PDF belgeleri inceleniyor...")
            docs = retriever.invoke(prompt_input)
            pdf_context = "\n\n".join(d.page_content for d in docs)
            
            # Eğer PDF'te bilgi yoksa internete git
            if len(pdf_context.strip()) < 50:
                st.write("🌐 PDF'de bulunamadı, internette araştırılıyor...")
                internet_results = search_tool.invoke(prompt_input)
                final_context = f"İnternet Kaynakları: {internet_results}"
            else:
                st.write("✅ PDF belgelerinde ilgili maddeler bulundu.")
                final_context = f"PDF Kayıtları: {pdf_context}"
            
            status.update(label="✅ Analiz tamamlandı!", state="complete", expanded=False)

        # 2. Adım: Llama ile Yanıt Oluşturma
        with st.spinner("Hukuki yanıt formüle ediliyor..."):
            master_prompt = f"""Sen profesyonel bir Türk Hukuk uzmanısın. 
            Aşağıdaki bilgileri kullanarak soruyu net, kısa ve tamamen Türkçe olarak cevapla.
            Asla İngilizce kelime kullanma.

            Bilgi Kaynağı: {final_context}
            Soru: {prompt_input}
            Cevap:"""
            
            full_response = llm.invoke(master_prompt)
            st.markdown(full_response)
    
    # Yanıtı geçmişe ekle
    st.session_state.messages.append({"role": "assistant", "content": full_response})