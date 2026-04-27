import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
JSON_FILE  = "sshi_metadata_try1.json"
INDEX_FILE = "embeddings_index.npz"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # supports Arabic

# =========================
# 1) Load Dataset
# =========================
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

words   = [item["wordAr"] for item in data]
ids     = [item["id"]     for item in data]

# =========================
# 2) Load Model
# =========================
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

# =========================
# 3) Build or Load Index
# =========================
if os.path.exists(INDEX_FILE):
    print("Loading saved index...")
    saved = np.load(INDEX_FILE, allow_pickle=True)
    embeddings = saved["embeddings"]
    print(f"Index loaded — {len(embeddings)} words.")
else:
    print(f"Building index for {len(words)} words (first time only)...")
    embeddings = model.encode(words, show_progress_bar=True, batch_size=64)
    np.savez(INDEX_FILE, embeddings=embeddings)
    print("Index saved to", INDEX_FILE)


# =========================
# 4) Search Function
# =========================
def find_closest(query, top_k=5):
    query_vec = model.encode([query])[0]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    scores = np.dot(embeddings, query_vec) / (norms + 1e-9)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            "id":    ids[i],
            "word":  words[i],
            "score": round(float(scores[i]), 4)
        })

    return results


# =========================
# 5) Test Loop
# =========================
print("\n" + "=" * 50)
print("Embeddings Search — Arabic Sign Language")
print("Type 'exit' to quit")
print("=" * 50)

while True:
    try:
        query = input("\n>> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break

    if not query:
        continue

    if query.lower() == "exit":
        print("Exiting...")
        break

    results = find_closest(query, top_k=5)

    print(f"\nTop matches for '{query}':")
    for r in results:
        print(f"  ID: {r['id']:5} | Score: {r['score']:.4f} | Word: {r['word']}")