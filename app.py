# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("company_policy.pdf")

# documents = loader.load()

# # print(type(documents))
# # print(len(documents))
# # print(documents[0].page_content) #access the content of the first page
# print(documents[0].metadata) #access the metadata of the first page


# ------------------Day 2-------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

print("Documents:", len(documents))
print("Chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]): #checking the first 5 chunks
    print(f"\n--- CHUNK {i} ---")
    print(chunk.page_content)
    print("METADATA:", chunk.metadata)


for i, chunk in enumerate(chunks[:10]): #checking the first 10 chunks size
    print(
        f"Chunk {i}:",
        len(chunk.page_content),
        "characters"
    )



#from langchain_text_splitters import RecursiveCharacterTextSplitter

small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=25
)

medium_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

large_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

small_chunks = small_splitter.split_documents(documents)
medium_chunks = medium_splitter.split_documents(documents)
large_chunks = large_splitter.split_documents(documents)

# print("Small chunks:", len(small_chunks))
# print("Medium chunks:", len(medium_chunks))
# print("Large chunks:", len(large_chunks))

#print(chunks[0].metadata)

text=[chunk.page_content for chunk in chunks]

embeddings = OllamaEmbeddings(
    model="embeddinggemma")

vectors = embeddings.embed_documents(text)

# print("Vectors:", len(vectors))
# print(vectors[0]) #checking the first vector

print("TEXT:")
print(chunks[0].page_content)

print("\nVECTOR:")
print(vectors[0][:10])

print("\nDIMENSION:")
print(len(vectors[0]))