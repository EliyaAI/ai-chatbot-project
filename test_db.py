import faiss
import pickle

index = faiss.read_index("vector_db/index.faiss")

print("Index vectors:", index.ntotal)

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Chunks:", len(chunks))

print(chunks[0][:300])
