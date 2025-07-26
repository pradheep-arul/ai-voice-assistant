from fastrtc import get_stt_model, get_tts_model

tts_client = get_tts_model()
stt_client = get_stt_model()


def startup():
    for chunk in tts_client.stream_tts_sync(
        "Welcome to AI Voice Assistant! How can I help you?"
    ):
        yield chunk
