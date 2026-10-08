from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# 1. Load PDF
loader = PyPDFLoader("company_policy.pdf")
documents = loader.load()

print("Pages:", len(documents))


# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))


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


# 5. Ask a question
query = "How many sick leave days do employees receive?"


# 6. Retrieve relevant chunks
results = vectorstore.similarity_search(
    query,
    k=3
)


# 7. Display results
for i, doc in enumerate(results, start=1):

    print(f"\n===== RESULT {i} =====")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)