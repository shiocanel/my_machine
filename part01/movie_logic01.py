import numpy as np

user_preference = np.array([0.9, 0.1, 0.8])

movie_a = np.array([0.8, 0.0, 0.9])
movie_b = np.array([0.1, 0.9, 0.1])
movie_c = np.array([0.5, 0.5, 0.2])

def calculate_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sim_a = calculate_similarity(user_preference, movie_a)
sim_b = calculate_similarity(user_preference, movie_b)
sim_c = calculate_similarity(user_preference, movie_c)

print(f"Similarity with Movie A: {sim_a:.4f}")
print(f"Similarity with Movie B: {sim_b:.4f}")
print(f"Similarity with Movie C: {sim_c:.4f}")