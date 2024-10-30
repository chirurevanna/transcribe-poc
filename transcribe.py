from openai import OpenAI
import os
from pydub import AudioSegment
import time

start_time = time.time()
print(os.environ.get("OPENAI_API_KEY"))

client = OpenAI(
    api_key="")

# Load your original audio file
audio = AudioSegment.from_mp3("audio/abc.mp3")

# Determine the length of the audio file in milliseconds
audio_length = len(audio)

# Initialize variables for processing
chunk_length = 60000  # 1 minute in milliseconds
start = 0

# Open the output text file in append mode
with open("output/abc_english.txt", "a") as text_file:
    while start < audio_length:
        # Extract a 1-minute chunk from the audio file
        end = min(start + chunk_length, audio_length)
        segment = audio[start:end]

        # Export this clip to a new file
        segment_file_name = f"audio/segment_{start//1000}-{end//1000}.mp3"
        segment.export(segment_file_name, format="mp3")

        # Process the audio segment with the API
        with open(segment_file_name, "rb") as audio_file:
            translation = client.audio.translations.create(
                model="whisper-1",
                file=audio_file
            )

        # Append the translated text to the output file
        lines = translation.text.split('. ')
        for line in lines:
            text_file.write(line.strip() + '.\n')

        # Move to the next segment
        start += chunk_length

end_time = time.time()

# Calculate the runtime in seconds
runtime_seconds = end_time - start_time

# Convert the runtime to minutes
runtime_minutes = runtime_seconds / 60

print(f"Runtime of the program: {runtime_minutes:.2f} minutes")
