from sentence_transformers import SentenceTransformer
import numpy as np

# model produces 384-dimensional embeddings for "all-MiniLM-L6-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    """Return a single 1-D float32 embedding for `text`.

    The SentenceTransformer `encode` with a list input returns a 2-D array
    shaped (1, dim). This helper returns the first row as a 1-D float32
    numpy array for downstream consumers.
    """

    emb = model.encode([text])

    emb = np.array(emb, dtype="float32")

    # emb may be shape (1, dim) — return the single vector as 1-D
    return emb.reshape(-1)