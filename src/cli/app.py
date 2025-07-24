import argparse
import base64

import requests
from dotenv import load_dotenv

from src.cli.cli_socket import main as websocket_main
from src.cli.recording.recording_client import RecordingClient
from src.cli.tts import tts_and_play

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
                tts_and_play(transcription)
    except KeyboardInterrupt:
        print("\nStopping transcription.")


def send_action(transcription: str, user: str):
    """Send an action to the server."""
    data = {
        "transcription": transcription,
        "user": user,
    }
    res = requests.post(url="http://127.0.0.1:5000/action", json=data)
    if res.status_code != 200:
        print(f"Error sending action: {res.status_code} - {res.text}")
    else:
        print(f"Action sent successfully: {res.json()}")


def process_audio_file(file_path: str, user: str):
    """Process an audio file, transcribe each batch, and send actions to the server."""
    print("Processing audio file and sending actions...")
    client = RecordingClient(file_path)
    batch = b""
    while True:
        batch_to_add = client.get_next_batch()
        if not batch_to_add:
            print("No more audio data to process.")
            break
        batch += batch_to_add
    data = {
        "b64_pcm": base64.b64encode(batch).decode("ascii"),
        "duration": len(batch) / 16000,
    }
    response = requests.post(url="http://127.0.0.1:5000/transcribe", json=data)
    if response.status_code != 200:
        print(f"Transcription error: {response.status_code} - {response.text}")
        return
    transcription = response.json().get("transcription", "")
    print(f"Transcription: {transcription}")
    if transcription:
        action_data = {
            "transcription": transcription,
            "user": user,
        }
        action_response = requests.post(url="http://127.0.0.1:5000/action", json=action_data)
        if action_response.status_code != 200:
            print(f"Action error: {action_response.status_code} - {action_response.text}")
        else:
            print(f"Action response: {action_response.json()}")


def main():
    parser = argparse.ArgumentParser(description="CLI application for the-curator project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio using the recording client.")
    transcribe_parser.add_argument(
        "--enable-tts",
        action="store_true",
        help="Enable TTS playback for transcriptions.",
    )
    transcribe_parser.set_defaults(func=lambda args: transcribe_audio(enable_tts=args.enable_tts))
    action_parser = subparsers.add_parser("action", help="Send an action to the server.")
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
    action_parser.set_defaults(func=lambda args: send_action(args.transcription, args.user))

    file_parser = subparsers.add_parser(
        "process-file",
        help="Process an audio file and send transcription to the action endpoint.",
    )
    file_parser.add_argument("file_path", type=str, help="Path to the local audio file to process.")
    file_parser.add_argument(
        "user",
        type=str,
        help="User identifier to associate with the action.",
    )
    file_parser.set_defaults(func=lambda args: process_audio_file(args.file_path, args.user))

    websocket_parser = subparsers.add_parser(
        "websocket-mode",
        help="Interact with the backend using websocket mode (Socket.IO).",
    )
    websocket_parser.set_defaults(func=lambda args: websocket_main())

    args = parser.parse_args()
    args.func(args)
