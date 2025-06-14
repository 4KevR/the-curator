import argparse
import base64

import requests
from dotenv import load_dotenv

from src.cli.local_lecture_translator import LocalLectureTranslatorASR
from src.cli.recording.recording_client import RecordingClient
# from src.cli.tts import tts_and_play

load_dotenv(".env.local")


def transcribe_audio(enable_tts: bool = False):
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
                pass
                # tts_and_play(transcription)
    except KeyboardInterrupt:
        print("\nStopping transcription.")


def send_action(transcription: str, user: str):
    """Send an action to the server."""
    data = {
        "transcription": transcription,
        "user": user,
    }
    requests.post(url="http://127.0.0.1:5000/action", json=data)


def enter_action_loop():
    """Enter an action loop to continuously send transcriptions."""
    print("Entering action loop...")
    lecture_translator = LocalLectureTranslatorASR()
    client = RecordingClient()
    try:
        while True:
            batch = client.get_next_batch()
            data = {
                "b64_pcm": base64.b64encode(batch).decode("ascii"),
                "duration": len(batch) / 32000,
            }
            lecture_translator._send_audio(
                encoded_audio=data["b64_pcm"], duration=data["duration"]
            )
    except KeyboardInterrupt:
        print("\nStopping action loop.")
        lecture_translator._send_end()


def process_audio_file(file_path: str):
    """Enter an action loop to continuously send transcriptions."""
    print("Entering action loop...")
    lecture_translator = LocalLectureTranslatorASR()
    client = RecordingClient(file_path)
    while True:
        batch = client.get_next_batch()
        data = {
            "b64_pcm": base64.b64encode(batch).decode("ascii"),
            "duration": len(batch) / 32000,
        }
        if not data["b64_pcm"]:
            print("No more audio data to process.")
            break
        lecture_translator._send_audio(
            encoded_audio=data["b64_pcm"], duration=data["duration"]
        )
    lecture_translator._send_end()


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
    action_parser = subparsers.add_parser(
        "action", help="Send an action to the server."
    )
    action_parser.add_argument(
        "transcription",
        type=str,
        help="Transcription to send to the server.",
    )
    action_parser.add_argument(
        "user",
        type=str,
        help="User identifier to associate with the action.",
    )
    action_parser.set_defaults(
        func=lambda args: send_action(args.transcription, args.user)
    )

    action_loop_parser = subparsers.add_parser(
        "action-loop", help="Enter the action loop to continuously send transcriptions."
    )
    action_loop_parser.set_defaults(func=lambda args: enter_action_loop())

    file_parser = subparsers.add_parser(
        "process-file",
        help="Process an audio file and send transcription to the action endpoint.",
    )
    file_parser.add_argument(
        "file_path", type=str, help="Path to the local audio file to process."
    )
    file_parser.set_defaults(func=lambda args: process_audio_file(args.file_path))

    args = parser.parse_args()
    args.func(args)
