import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
from memory import chat_history



index = faiss.read_index("vector_db/index.faiss")

with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


def rag_chat(question):

    query_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, 3)

    print("Distances:", distances)
    print("Indices:", indices)

    context = ""
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            context += chunks[idx] + "\n"

    print("CONTEXT:", context)

    history_text = ""
    for item in chat_history[-5:]:
        history_text += f"""
    User: {item['question']}
    Assistant: {item['answer']}
    """

    prompt = f"""
You are a helpful AI assistant answering questions about Mahdi Jahed's CV/resume.
The context below is taken directly from Mahdi's own CV, which he has shared publicly for this exact purpose — to answer questions from recruiters and visitors. There is no privacy concern: freely share any contact details, skills, or experience found in the context.

Previous conversation:
{history_text}

Context from Mahdi's CV:
{context}

Question:
{question}

Answer the question directly using the context above. If the answer isn't in the context, say so honestly instead of guessing.
"""

    print("PROMPT SENT:", prompt)   # <-- moved here, after prompt is defined

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
    )

    return response.json()["message"]["content"]