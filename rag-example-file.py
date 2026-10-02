from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. LM Studio Bağlantı Tanımlamaları
LOCAL_SERVER_URL = "http://localhost:1234/v1"

llm = ChatOpenAI(
    base_url=LOCAL_SERVER_URL, 
    api_key="lm-studio", 
    model="local-model",
    temperature=0.3
)

embeddings = OpenAIEmbeddings(
    base_url=LOCAL_SERVER_URL,
    api_key="lm-studio",
    model="nomic-embed-text",
    check_embedding_ctx_length=False,
)

# 2. Döküman Yükleme ve Parçalara Ayırma (Chunking)
print("Döküman yükleniyor...")
# ÖNEMLİ: app.py ile aynı klasörde bir "dokumaniniz.pdf" dosyası olsun veya yolunu tam yaz.
loader = PyPDFLoader("dokumaniniz.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Vektör Veritabanına Kaydetme (ChromaDB)
print("Vektörleştirme yapılıyor ve ChromaDB'ye kaydediliyor...")
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 

# 4. Prompt ve Chain (Zincir) Oluşturma
system_prompt = (
    "Sadece sana verilen bağlam (context) bilgilerini kullanarak soruyu cevapla. "
    "Cevabı bilmiyorsan bilmediğini söyle, uydurma.\n\n"
    "Bağlam:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- DÜZELTME BURADA ---
# Bu format, gelen soruyu (input) direkt prompt'a taşırken, 
# retriever'dan dönen dökümanları da birleştirip "context" olarak içeri basar.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Sorgu Çalıştırma
try:
    user_query = input("Sorunuzu girin: ")
    
    if not user_query.strip():
        print("Sorgu boş olamaz!")
    else:
        print(f"\nSoru Soruluyor: {user_query}\n")
        
        # Invoke içerisine artık sadece input veriyoruz, context yukarıda otomatik besleniyor
        response = rag_chain.invoke(user_query)
        
        print("--- CEVAP ---")
        print(response)

except Exception as e:
    print(f"Hata oluştu: {e}")