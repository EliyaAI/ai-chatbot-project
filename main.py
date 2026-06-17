from fastapi import FastAPI
from rag import rag_chat

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG Running"}

@app.get("/chat")
def chat(question: str):
    return {"answer": rag_chat(question)}
