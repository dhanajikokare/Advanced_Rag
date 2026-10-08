from langchain_community.retrievers import BM25Retriever

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

loader = PyPDFLoader("company_policy.pdf")

documents = loader.load()

for doc in documents:
    doc.metadata["department"] = "HR"
    doc.metadata["document_type"] = "policy"
    doc.metadata["year"] = 2026
    doc.metadata["version"] = "v2"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# 3. Create embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy"
)



bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 3

query = "What is the leave policy?"

results = bm25_retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"\n===== RESULT {i} =====")
    print(doc.page_content)
    print("Metadata:", doc.metadata)