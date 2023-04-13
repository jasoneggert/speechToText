import os
from dotenv import load_dotenv
import argparse
import openai
from pydub import AudioSegment, silence

# Load the .env file
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Transcribe audio and generate summary using OpenAI Whisper and GPT-3')
parser.add_argument('file_path', type=str, help='Path to the audio file')
args = parser.parse_args()

# Access the OpenAI credentials
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the audio file path
audio_file_path = args.file_path

# Load the audio file using PyDub
audio = AudioSegment.from_file(audio_file_path)

# Set silence threshold in dB (lower values indicate more silence)
silence_threshold = -30

# Remove silence from the audio
audio_without_silence = silence.detect_silence(
    audio, silence_thresh=silence_threshold)

# Set chunk duration in milliseconds (5 minutes = 300,000 milliseconds)
chunk_duration = 300000

# Split the audio without silence into chunks
chunks = []
for i in range(0, len(audio_without_silence), chunk_duration):
    chunk = audio_without_silence[i:i + chunk_duration]
    chunks.append(chunk)

# Process each chunk and transcribe using Whisper API
transcript = ""
for i, chunk in enumerate(chunks):
    # Save the chunk as a temporary file
    chunk_file_path = f"chunk_{i + 1}.wav"
    chunk.export(chunk_file_path, format="wav")

    # Transcribe the chunk using Whisper API
    with open(chunk_file_path, "rb") as chunk_file:
        chunk_transcript = openai.Audio.transcribe("whisper-1", chunk_file)
        transcript += chunk_transcript

    # Clean up temporary file
    os.remove(chunk_file_path)

print("Transcript:\n", transcript)

# Generate summary using GPT-3
summary = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Generate a summary of the key points from the audio transcription:\n{transcript}"}
    ],
    max_tokens=1000
)
if summary.choices and summary.choices[0].text:
    print("Summary:\n", summary.choices[0].text)
else:
    print("Failed to generate summary.")
