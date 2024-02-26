from typing import List, Tuple
from deeplake.core.dataset import Dataset as DeepLakeDataset
import torch
import numpy as np
import time
from tqdm import tqdm


distance_metric_map = {
    "l2": lambda a, b: np.linalg.norm(a - b, axis=1, ord=2),
    "l1": lambda a, b: np.linalg.norm(a - b, axis=1, ord=1),
    "max": lambda a, b: np.linalg.norm(a - b, axis=1, ord=np.inf),
    "cos": lambda a, b: np.dot(a, b.T)
    / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)),
}


def batch_cosine_similarity(query, embeddings, batch_size=100000):
    """Calculate cosine similarity in batches to reduce memory usage."""
    num_embeddings = embeddings.shape[0]
    cos_similarities = np.zeros(num_embeddings)
    for i in range(0, num_embeddings, batch_size):
        batch = embeddings[i:i + batch_size]
        cos_similarities[i:i + batch_size] = np.dot(batch, query.T) / (
                np.linalg.norm(query) * np.linalg.norm(batch, axis=1))
    return cos_similarities


def batch_inner_product(query, embeddings, batch_size=500000):
    """Calculate inner product in batches using PyTorch and fp16, with batch conversion."""
    # Convert query to PyTorch tensor of fp16 outside the loop since it's reused
    query = torch.tensor(query, dtype=torch.float32).unsqueeze(1)  # Ensure query is 2D for mm

    num_embeddings = embeddings.shape[0]
    # Initialize the result array in numpy to save on memory
    inner_products = np.zeros(num_embeddings, dtype=np.float32)

    for i in range(0, num_embeddings, batch_size):
        # Convert each batch to PyTorch tensor of fp16 within the loop
        batch = torch.tensor(embeddings[i:i + batch_size], dtype=torch.float32)
        # Perform matrix multiplication and convert the result to numpy in fp32
        inner_products[i:i + batch_size] = torch.mm(batch, query).squeeze().numpy()

    return inner_products


def search(
    deeplake_dataset: DeepLakeDataset,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    distance_metric: str = "l2",
    k: int = 4,
) -> Tuple[DeepLakeDataset, List]:
    if embeddings.shape[0] == 0:
        return deeplake_dataset[0:0], []

    if len(query_embedding.shape) > 1:
        query_embedding = query_embedding[0]

    # Use the appropriate distance calculation
    if distance_metric == "cos":
        distances = batch_cosine_similarity(query_embedding, embeddings)
    elif distance_metric == "inner":
        distances = batch_inner_product(query_embedding, embeddings)
    else:
        distances = distance_metric_map[distance_metric](query_embedding, embeddings)
    
    if distance_metric in ["cos", "inner"]:  # Assuming higher values are better for both
        nearest_indices = np.argsort(distances)[::-1][:k]
    else:
        nearest_indices = np.argsort(distances)[:k]

    return (
        deeplake_dataset[nearest_indices.tolist()],
        distances[nearest_indices].tolist(),
    )
