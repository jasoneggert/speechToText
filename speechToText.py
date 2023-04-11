import os
from dotenv import load_dotenv
import argparse
import openai

# Load the .env file
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Transcribe audio using OpenAI Whisper API')
parser.add_argument('file_path', type=str, help='Path to the audio file')
args = parser.parse_args()

# Access the OpenAI credentials
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use the credentials in your code
audio_file_path = args.file_path
with open(audio_file_path, "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
