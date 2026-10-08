from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# 1. Load document
loader = TextLoader("company_policy.pdf")
documents = loader.load()


# 2. Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# 3. Create embedding model
embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# 4. Store chunks in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy"
)


# 5. Vector retriever
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# 6. BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3


# 7. Combine both
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever
    ],
    weights=[
        0.3,   # BM25
        0.7    # Vector search
    ]
)


# 8. Search
query = "What is the employee leave policy?"

results = hybrid_retriever.invoke(query)


# 9. Display results
for i, doc in enumerate(results, 1):

    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)