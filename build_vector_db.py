from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

PDF_PATH = r"documents/CV.pdf"

reader = PdfReader(PDF_PATH)

text = ""
for page in reader.pages:
    if page.extract_text():
        text += page.extract_text()

print("TEXT LENGTH:", len(text))
print(text[:1000])

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Chunks:", len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))


os.makedirs("vector_db", exist_ok=True)

faiss.write_index(index, "vector_db/index.faiss")

with open("vector_db/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("DONE")