import csv
import os
from openai import OpenAI
from pathlib import Path

# Initialize OpenAI client
# Make sure to set your API key as an environment variable: OPENAI_API_KEY
client = OpenAI()

def generate_tts_audio():
    """
    Reads FAQ answers from CSV and generates text-to-speech audio files.
    """

    # Define paths
    csv_file = Path("data/faq.csv")
    audio_folder = Path("audio")

    # Create audio folder if it doesn't exist
    audio_folder.mkdir(exist_ok=True)

    print(f"Reading FAQ data from {csv_file}")
    print(f"Saving audio files to {audio_folder}")

    try:
        # Generate speech using OpenAI TTS API
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input="Hello! I'm here to answer some questions you have about the placement year at Loughborough University."
        )

        # Save audio file
        audio_filename = f"audio_15.mp3"
        audio_path = audio_folder / audio_filename

        response.stream_to_file(audio_path)

        print(f"✓ Saved {audio_filename}")

    except Exception as e:
        print("✗ Error generating audio for FAQ ")
    
    '''

    # Read CSV file
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            faq_id = row['id']
            answer = row['answer']
            if (int(faq_id) == 19):

                # Skip empty answers
                if not answer.strip():
                    print(f"Skipping FAQ {faq_id} - empty answer")
                    continue

                print(f"Generating audio for FAQ {faq_id}...")

                try:
                    # Generate speech using OpenAI TTS API
                    response = client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
                        input=answer
                    )

                    # Save audio file
                    audio_filename = f"audio_{faq_id}.mp3"
                    audio_path = audio_folder / audio_filename

                    response.stream_to_file(audio_path)

                    print(f"✓ Saved {audio_filename}")

                except Exception as e:
                    print(f"✗ Error generating audio for FAQ {faq_id}: {str(e)}")
                    continue
    '''

    print("\nTTS generation completed!")

if __name__ == "__main__":
    generate_tts_audio()


