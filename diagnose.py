import faiss
import pickle

index = faiss.read_index("vector_db/index.faiss")
print("Index total vectors:", index.ntotal)
print("Index dimension:", index.d)

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Number of chunks:", len(chunks))
print("First chunk preview:", chunks[0][:200] if chunks else "NO CHUNKS")