import argparse
import base64

import requests
from dotenv import load_dotenv

from src.cli.recording.recording_client import RecordingClient
from src.cli.tts import tts_and_play

load_dotenv(".env.local")


def transcribe_audio(enable_tts=False):
    """Transcribe audio using the recording client."""
    client = RecordingClient()
    try:
        while True:
            input("Press Enter to start recording...")
            client.get_next_batch()
            input("Press Enter to send the next batch...")
            batch = client.get_next_batch()
            data = {
                "b64_pcm": base64.b64encode(batch).decode("ascii"),
                "duration": len(batch) / 32000,
            }
            response = requests.post(url="http://127.0.0.1:5000/transcribe", json=data)
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                continue
            transcription = response.json().get("transcription", "")
            print(f"Transcription: {transcription}")
            if enable_tts and transcription:
                tts_and_play(transcription)
    except KeyboardInterrupt:
        print("\nStopping transcription.")


def main():
    parser = argparse.ArgumentParser(
        description="CLI application for the-curator project."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe audio using the recording client."
    )
    transcribe_parser.add_argument(
        "--enable-tts",
        action="store_true",
        help="Enable TTS playback for transcriptions.",
    )
    transcribe_parser.set_defaults(
        func=lambda args: transcribe_audio(enable_tts=args.enable_tts)
    )
    args = parser.parse_args()
    args.func(args)
