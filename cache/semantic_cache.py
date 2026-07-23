import faiss
import numpy as np

questions = []
answers = []

# use inner-product index for cosine similarity after normalizing vectors
dimension = 384

index = faiss.IndexFlatIP(dimension)

def _ensure_2d(vec):
    a = np.array(vec, dtype="float32")
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a

def _normalize(a):
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms

def store_semantic(question, embedding, answer):
    """Store a question, its embedding and answer into the in-memory FAISS index."""

    vec = _ensure_2d(embedding)

    # normalize for cosine similarity with IndexFlatIP
    vec = _normalize(vec)

    questions.append(question)
    answers.append(answer)

    index.add(vec)

def search_semantic(query_embedding, k=1):
    """Search the FAISS index and return similarity scores and indices.

    Returns (scores, indices) where higher `scores` are more similar (cosine).
    """

    vec = _ensure_2d(query_embedding)
    vec = _normalize(vec)

    scores, indices = index.search(vec, k=k)

    return scores, indices

def reset():
    """Clear all cached entries and reset the FAISS index."""
    global index
    questions.clear()
    answers.clear()
    index = faiss.IndexFlatIP(dimension)