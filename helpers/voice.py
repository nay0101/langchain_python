from pathlib import Path
from openai import OpenAI
from .config import Config
from pydub import AudioSegment


openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)


def speech_to_text(audio_file_path):
    path = Path(audio_file_path).resolve()
    chunk_length = 10 * 60 * 1000
    audio = AudioSegment.from_file(path)
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i : i + chunk_length]
        chunks.append(chunk)

    transcriptions = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = Path(f"./audio/chunk{i}_chunks.mp3").resolve()
        chunk.export(chunk_file_path, format="mp3")
        with open(chunk_file_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="en",
                response_format="verbose_json",
            )
        transcriptions.append(transcript.text)
    return "".join(transcriptions)


def text_to_speech(text, response_audio_file_path):
    max_length = 4096
    chunks = [text[i : i + max_length] for i in range(0, len(text), max_length)]
    audio_files = []
    for i, chunk in enumerate(chunks):
        try:
            response = openai_client.audio.speech.create(
                model="tts-1", voice="alloy", input=chunk
            )
            speech_file_path = Path(f"audio/chunk_{i}.mp3").resolve()
            response.stream_to_file(speech_file_path)
            audio_files.append(AudioSegment.from_mp3(speech_file_path))
        except Exception as e:
            print(f"Error in text-to-speech conversion for chunk {i}: {e}")
            return None
    # Combine audio files into one
    combined = AudioSegment.empty()
    for audio in audio_files:
        combined += audio

    # Export combined audio to a file
    combined_file_path = Path(response_audio_file_path)
    combined.export(combined_file_path, format="mp3")
    return combined_file_path
