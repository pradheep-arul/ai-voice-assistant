import queue

import ollama

LLM_MODEL = "gemma3n:e4b"  # Ollama model to use
q = queue.Queue()


def get_llm_response(chat_history: queue.Queue) -> str:
    """Generate response using Ollama."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI Voice assistant. Your goal is to generate the test response. Your output will be converted to audio so don't include emojis or special characters in your answers. Respond in few words, no more than 20 words.",
        }
    ]

    print("Chat History:", list(chat_history.queue))

    for role, content in list(chat_history.queue):
        messages.append({"role": role.lower(), "content": content})

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]
