# build_index.py

text = read_pdf(...)
chunks = chunk_text(text)

embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

faiss.write_index(
    index,
    "vector_db/index.faiss"
)

with open(
    "vector_db/chunks.pkl",
    "wb"
) as f:
    pickle.dump(chunks, f)