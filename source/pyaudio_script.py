import pyaudio
import wave
import whisper
import ollama
from openai import OpenAI
import os
import tempfile
import struct

# Constants for Ollama and Whisper models
LLM_MODEL = "gemma3n:e4b"  # Ollama model to use
WHISPER_MODEL = "small.en"  # Whisper model to use for transcription    
# ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo']

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
BUFFER_RECORD_SECONDS = 1  # Adjust as needed

def record_audio():
    """Record audio from the microphone and save to a temporary WAV file."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording... Speak now.")
    frames = []
    
    has_not_spoken_yet = True
    contains_data = True
    while contains_data or has_not_spoken_yet:
        current_buffer_content = []
        for _ in range(0, int(RATE / CHUNK * BUFFER_RECORD_SECONDS)):
            data = stream.read(CHUNK)
            # print("Recording chunk 2..." + str(len(data)))
            # print("Recording chunk 2..." + str(data[:20]))
            
            # print("Max Data " + str(max(data)))
            frames.append(data)
            current_buffer_content.append(data)
                
        # Check if average amplitude of the current buffer is above a threshold
        # Flatten all samples in the buffer
        all_samples = []
        for chunk in current_buffer_content:
            # Unpack chunk into 16-bit signed integers
            samples = struct.unpack('<' + 'h' * (len(chunk) // 2), chunk)
            all_samples.extend(samples)
        # Calculate average amplitude
        average_amplitude = sum(abs(s) for s in all_samples) / len(all_samples) if all_samples else 0
        # print(f"Average amplitude: {average_amplitude}")
        if average_amplitude < 100:  # Adjust threshold as needed
            contains_data = False
        else:
            has_not_spoken_yet = False
            contains_data = True
    
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
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_file)
    return result["text"]

def generate_response(prompt):
    """Generate response using Ollama."""
    messages=[
        {
            'role': 'system',
            'content': 'You are an AI Voice assistant. Give your responses in a short, concise and conversational manner.'
        },
        {
            'role': 'user',
            'content': prompt,
        },
    ]

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response['message']['content']

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
        
        if prompt.strip() == "":
            print("No speech detected. Please try again.")
            return
        
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