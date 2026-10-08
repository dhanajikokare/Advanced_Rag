from langchain_community.document_loaders import TextLoader, PyPDFLoader

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------
# 1. Load document
# --------------------------------

loader = PyPDFLoader("company_policy_1.pdf")

documents = loader.load()


# --------------------------------
# 2. Embeddings
# --------------------------------

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# --------------------------------
# 3. Vector store
# --------------------------------

vectorstore = Chroma(
    collection_name="parent_child_demo",
    embedding_function=embeddings
)


# --------------------------------
# 4. Parent splitter
# --------------------------------

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)


# --------------------------------
# 5. Child splitter
# --------------------------------

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)


# --------------------------------
# 6. Parent document store
# --------------------------------

store = InMemoryStore()


# --------------------------------
# 7. Parent retriever
# --------------------------------

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)


# --------------------------------
# 8. Add documents
# --------------------------------

retriever.add_documents(documents)


# --------------------------------
# 9. Search
# --------------------------------

query = "What is the annual leave policy?"

results = retriever.invoke(query)


# --------------------------------
# 10. Display
# --------------------------------

for i, doc in enumerate(results, 1):

    print(f"\n===== RESULT {i} =====")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)