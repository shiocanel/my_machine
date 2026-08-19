import torch

user_prefernce = torch.tensor([0.9, 0.0, 0.8])

movie_a = torch.tensor([0.8, 0.0, 0.9])
movie_b = torch.tensor([0.1, 0.9, 0.1])
movie_c = torch.tensor([0.5, 0.5, 0.2])

def calculate_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

sim_a = calculate_similarity(user_prefernce, movie_a)
sim_b = calculate_similarity(user_prefernce, movie_b)
sim_c = calculate_similarity(user_prefernce, movie_c)

print(f"Similarity with Movie A: {sim_a.item():.4f}")
print(f"Similarity with Movie B: {sim_b.item():.4f}")
print(f"Similarity with Movie C: {sim_c.item():.4f}")