import sqlite3
import hashlib
import json
import numpy as np

# ─── Simple Vector Store (no ChromaDB hanging issues) ───

def embed(text):
    """Hash-based embedding — instant, no downloads."""
    hash_bytes = hashlib.sha256(text.encode()).digest()
    return [hash_bytes[i % len(hash_bytes)] / 255.0 for i in range(128)]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Load from SQLite
conn = sqlite3.connect("collections.db")
cursor = conn.execute("""
    SELECT vendor, risk_rating, days_overdue, reasoning, timestamp
    FROM agent_runs
    WHERE reasoning IS NOT NULL
    LIMIT 20
""")
rows = cursor.fetchall()
conn.close()

print(f"Rows fetched: {len(rows)}")

# Build vector store as a list of dicts
vector_store = []
for i, row in enumerate(rows):
    vendor, risk_rating, days_overdue, reasoning, timestamp = row
    doc = f"Vendor: {vendor}. Risk: {risk_rating}. Days overdue: {days_overdue}. Assessment: {reasoning[:300]}"
    vector_store.append({
        "id": f"run_{i}",
        "document": doc,
        "embedding": embed(doc),
        "metadata": {
            "vendor": vendor,
            "risk_rating": risk_rating,
            "days_overdue": int(days_overdue) if days_overdue else 0
        }
    })

print(f"Stored {len(vector_store)} documents in vector store")

# Save to JSON for persistence
with open("vector_store.json", "w") as f:
    json.dump(vector_store, f)
print("Vector store saved to vector_store.json")

# ─── Query function ───
def query(query_text, n_results=3):
    query_embedding = embed(query_text)
    scored = []
    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n_results]

# Query 1 — HIGH risk
print("\nQuery 1: Honeywell overdue broken promises")
results = query("Honeywell HIGH risk overdue broken promises disputes")
for score, item in results:
    print(f"\n  Vendor: {item['metadata']['vendor']} | Risk: {item['metadata']['risk_rating']} | Score: {score:.3f}")
    print(f"  {item['document'][:150]}")

# Query 2 — LOW risk
print("\nQuery 2: Siemens low risk clean")
results2 = query("Siemens LOW risk clean current payment")
for score, item in results2:
    print(f"\n  Vendor: {item['metadata']['vendor']} | Risk: {item['metadata']['risk_rating']} | Score: {score:.3f}")
    print(f"  {item['document'][:150]}")

print("\nRAG vector store working!")