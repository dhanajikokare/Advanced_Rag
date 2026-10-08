from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever


# -----------------------------------
# 1. Create sample data
# -----------------------------------

documents = [

    Document(
        page_content="Employees receive 12 annual leave days.",
        metadata={
            "department": "HR",
            "year": 2025
        }
    ),

    Document(
        page_content="Employees receive 10 sick leave days.",
        metadata={
            "department": "HR",
            "year": 2026
        }
    ),

    Document(
        page_content="Employees must change passwords every 90 days.",
        metadata={
            "department": "IT",
            "year": 2025
        }
    ),

    Document(
        page_content="Employees can claim travel expenses.",
        metadata={
            "department": "Finance",
            "year": 2025
        }
    )
]


# -----------------------------------
# 2. Embedding model
# -----------------------------------

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)


# -----------------------------------
# 3. Vector database
# -----------------------------------

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="self_query_demo"
)


# -----------------------------------
# 4. LLM
# -----------------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# -----------------------------------
# 5. Describe metadata
# -----------------------------------

metadata_field_info = [

    AttributeInfo(
        name="department",
        description="Department such as HR, IT or Finance",
        type="string"
    ),

    AttributeInfo(
        name="year",
        description="Year the policy was published",
        type="integer"
    )
]


# -----------------------------------
# 6. Create Self Query Retriever
# -----------------------------------

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Company employee policies",
    metadata_field_info=metadata_field_info
)


# -----------------------------------
# 7. User query
# -----------------------------------

query = "Show me HR policies from 2025"


# -----------------------------------
# 8. Search
# -----------------------------------

results = retriever.invoke(query)


# -----------------------------------
# 9. Display results
# -----------------------------------

for i, doc in enumerate(results, 1):

    print(f"\nResult {i}")

    print("Content:")
    print(doc.page_content)

    print("Metadata:")
    print(doc.metadata)