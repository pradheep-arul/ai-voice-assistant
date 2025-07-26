from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    SileroVadOptions,
    Stream,
    get_stt_model,
    get_tts_model,
)
from utils.processing import process
from utils.start_up import startup

tts_client = get_tts_model()
stt_client = get_stt_model()


stream = Stream(
    handler=ReplyOnPause(
        process,
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
