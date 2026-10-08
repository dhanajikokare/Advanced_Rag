# from langchain_ollama import OllamaEmbeddings
# from ollama import embeddings
# embeddings=OllamaEmbeddings(
#     model="embeddinggemma"
# )

# #text = "Employees receive 12 sick leave days."

# texts = [
#     "Employees receive 12 sick leave days.",
#     "Employees can work from home 3 days per week.",
#     "The company headquarters are located in Pune."
# ]

# vectors = embeddings.embed_documents(texts)

# # print(vectors)
# # print("Dimensions:", len(vectors))

# sentences = [
#     "Employees receive 12 sick leave days.",
#     "How many sick leaves are available?",
#     "The company headquarters are located in Pune.",
#     "What is the location of the company office?"
# ]

# sentence_vectors = embeddings.embed_documents(sentences)

# from sklearn.metrics.pairwise import cosine_similarity

# scores=cosine_similarity(sentence_vectors)
# print("Cosine Similarity Scores:", scores)



# # print(sentence_vectors)
# # print("Dimensions:", len(sentence_vectors))

from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = OllamaEmbeddings(
    model="embeddinggemma"
)

texts = [
    "Employees receive 12 sick leave days.",
    "How many sick leaves are available?",
    "The company headquarters are located in Pune.",
    "Where is the company office located?"
]

vectors = embeddings.embed_documents(texts)

print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))

scores = cosine_similarity(vectors)

print("\nSimilarity Matrix:")
print(scores)