import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1, 1], or 0.0 if either vector has zero magnitude.
    """
    #magnitude of vectors
    norm_a = np.linalg.norm(a) 
    norm_b = np.linalg.norm(b)

    if norm_a==0 or norm_b==0:
        return 0.0
    
    dot_product = np.dot(a,b)
    return float(dot_product/(norm_a * norm_b))
