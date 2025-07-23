import pyaudio
import wave
import whisper
import ollama
from openai import OpenAI
import os
import tempfile

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Adjust as needed

def record_audio():
    """Record audio from the microphone and save to a temporary WAV file."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording... Speak now.")
    frames = []
    
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wf = wave.open(temp_wav.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return temp_wav.name

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

def generate_response(prompt):
    """Generate response using Ollama."""
    response = ollama.generate(model="gemma3:4b", prompt=prompt, think=False)
    return response['response']

def text_to_speech(text):
    """Convert text to speech and save to a temporary MP3 file."""
    client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_sky+af_bella",
            input=text
        ) as response:
            response.stream_to_file(temp_mp3.name)
        return temp_mp3.name

def play_audio(audio_file):
    """Play audio file using platform's default audio player."""
    if os.name == 'nt':  # Windows
        os.system(f"start {audio_file}")
    else:  # macOS/Linux
        os.system(f"aplay {audio_file}" if os.name == 'posix' and 'linux' in os.uname().sysname.lower() else f"afplay {audio_file}")

def main():
    try:
        # Step 1: Record audio
        audio_file = record_audio()
        
        # Step 2: Transcribe audio to text
        print("Transcribing...")
        prompt = transcribe_audio(audio_file)
        print(f"You said: {prompt}")
        
        # Step 3: Generate response with Ollama
        print("Generating response...")
        response_text = generate_response(prompt)
        print(f"Response: {response_text}")
        
        # Step 4: Convert response to speech
        print("Converting to speech...")
        speech_file = text_to_speech(response_text)
        
        # Step 5: Play the speech
        print("Playing response...")
        play_audio(speech_file)
        
        # Clean up temporary files
        os.remove(audio_file)
        os.remove(speech_file)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    while True:
        main()