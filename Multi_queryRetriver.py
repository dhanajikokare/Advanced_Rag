from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever


# -----------------------------
# 1. Load document
# -----------------------------

loader = PyPDFLoader("company_policy_1.pdf")
documents = loader.load()


# -----------------------------
# 2. Split
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# -----------------------------
# 3. Embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# -----------------------------
# 4. Vector Store
# -----------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="multi_query_demo"
)


# -----------------------------
# 5. Base Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# 6. LLM
# -----------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# -----------------------------
# 7. Multi Query Retriever
# -----------------------------

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# -----------------------------
# 8. Search
# -----------------------------

query = "What is the leave policy?"

results = multi_query_retriever.invoke(query)


# -----------------------------
# 9. Display
# -----------------------------

for i, doc in enumerate(results, 1):

    print(f"\n===== RESULT {i} =====")
    print(doc.page_content)
    print("Metadata:", doc.metadata)