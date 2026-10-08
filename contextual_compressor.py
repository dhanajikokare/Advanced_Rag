from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langchain_classic.retrievers import (
    ContextualCompressionRetriever
)

from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)


# vectorstore.delete_collection()

# -----------------------------------
# 1. Load document
# -----------------------------------

loader = PyPDFLoader("company_policy_1.pdf")
documents = loader.load()


# -----------------------------------
# 2. Chunking
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# -----------------------------------
# 3. Embedding model
# -----------------------------------

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# -----------------------------------
# 4. Vector store
# -----------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy_compression"
)


#vectorstore.delete_collection()

# -----------------------------------
# 5. Base retriever
# -----------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# -----------------------------------
# 6. Normal retrieval
# -----------------------------------

query = "When does an employee become eligible for health insurance?"

docs = retriever.invoke(query)

print("\n===== BEFORE COMPRESSION =====")

for i, doc in enumerate(docs, 1):

    print(f"\n--- Document {i} ---")
    print(doc.page_content)


# -----------------------------------
# 7. LLM
# -----------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# -----------------------------------
# 8. Compressor
# -----------------------------------

compressor = LLMChainExtractor.from_llm(llm)


# -----------------------------------
# 9. Compression retriever
# -----------------------------------

compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)


# -----------------------------------
# 10. Compressed retrieval
# -----------------------------------

compressed_docs = compression_retriever.invoke(query)

print("\n===== AFTER COMPRESSION =====")

for i, doc in enumerate(compressed_docs, 1):

    print(f"\n--- Document {i} ---")
    print(doc.page_content)