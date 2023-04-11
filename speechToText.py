import os
from dotenv import load_dotenv
import argparse
import openai

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

# Use the credentials in your code
audio_file_path = args.file_path
with open(audio_file_path, "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print("Transcript:\n", transcript)

    # Generate summary using GPT-3
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You help condense transcriptions of meetings into a brief summary of key points."},
            {"role": "user", "content": f"Generate a summary of the key points from the audio transcription:\n{transcript}"}
        ],  # Added a comma here
        max_tokens=1000
    )
    print("Summary:\n", summary)
