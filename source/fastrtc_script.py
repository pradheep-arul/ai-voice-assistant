import numpy as np
import ollama
from fastrtc import (AlgoOptions, ReplyOnPause, SileroVadOptions, Stream,
                     get_stt_model, get_tts_model)

LLM_MODEL = "gemma3n:e4b"  # Ollama model to use


tts_client = get_tts_model()
stt_client = get_stt_model()


def generate_response(prompt):
    """Generate response using Ollama."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI Voice assistant. Your goal is to generate the test response. Your output will be converted to audio so don't include emojis or special characters in your answers. Respond in few words, no more than 20 words.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]


def echo(audio: tuple[int, np.ndarray]):
    transcript = stt_client.stt(audio)
    print(f"You said: {transcript}")
    if transcript.strip() == "":
        return "No speech detected. Please try again."

    response_text = generate_response(transcript)
    print(f"Assistent Response: {response_text}")
    for audio_chunk in tts_client.stream_tts_sync(response_text):
        yield audio_chunk


def startup():
    for chunk in tts_client.stream_tts_sync(
        "Welcome to AI Voice Assistant! How can I help you?"
    ):
        yield chunk


stream = Stream(
    handler=ReplyOnPause(
        echo,
        can_interrupt=False,
        startup_fn=startup,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,
            started_talking_threshold=0.2,
            speech_threshold=0.1,
        ),
        model_options=SileroVadOptions(
            threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100
        ),
    ),
    modality="audio",
    mode="send-receive",
    ui_args={"title": "AI Voice Assistant"},
)
stream.ui.launch()
