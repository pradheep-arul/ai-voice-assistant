import queue

import numpy as np
from fastrtc import get_stt_model, get_tts_model

from .llm import get_llm_response

chat_history = queue.Queue(maxsize=10)


tts_client = get_tts_model()
stt_client = get_stt_model()


def add_to_chat_history(role: str, content: str):
    """Add a message to the chat history."""
    if chat_history.full():
        chat_history.get()  # Remove the oldest message if full
    chat_history.put((role, content))


def process(audio: tuple[int, np.ndarray]):
    transcript = stt_client.stt(audio)
    print(f"You said: {transcript}")
    if transcript.strip() == "":
        return "No speech detected. Please try again."

    add_to_chat_history("User", transcript)
    response_text = get_llm_response(chat_history)

    print(f"Assistent Response: {response_text}")
    add_to_chat_history("Assistant", response_text)

    for audio_chunk in tts_client.stream_tts_sync(response_text):
        yield audio_chunk
