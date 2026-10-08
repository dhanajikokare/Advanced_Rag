from sklearn.metrics.pairwise import cosine_similarity

sentence_a = [[1, 2]]
sentence_b = [[2, 4]]
sentence_c = [[2, 0]]

print("A vs B:",
      cosine_similarity(sentence_a, sentence_b))

print("A vs C:",
      cosine_similarity(sentence_a, sentence_c))