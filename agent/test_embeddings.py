from sentence_transformers import SentenceTransformer

print("Loading model (downloads ~80MB first time)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Test semantic similarity
sentences = [
    "overdue payment with broken promises",
    "late invoice, customer didn't pay as agreed",
    "the weather is sunny today"
]

embeddings = model.encode(sentences)
print(f"\nEmbedding shape: {embeddings.shape}")  # should be (3, 384)

# Quick cosine check
import numpy as np
def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nSimilarity between the two payment sentences: {cos(embeddings[0], embeddings[1]):.3f}")
print(f"Similarity between payment and weather:        {cos(embeddings[0], embeddings[2]):.3f}")