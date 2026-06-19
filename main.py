from fastapi import FastAPI
from rag import rag_chat

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG Running"}

@app.get("/chat")
def chat(question: str):
    return {"answer": rag_chat(question)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)