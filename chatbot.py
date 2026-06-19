import requests
from memory import chat_history

OLLAMA_URL = "http://localhost:11434/api/chat"

def ask_llama(user_message):

    chat_history.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "messages": chat_history,
            "stream": False
        }
    )

    answer = response.json()["message"]["content"]

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return answer




