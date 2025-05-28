import base64
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.adapter.cloud_lecture_translator import CloudLectureTranslatorASR
from src.cli.recording.recording_client import RecordingClient

if __name__ == "__main__":
    load_dotenv(".env")
    load_dotenv(".env.local")
    print(os.getenv("AUDIO_DEVICE"))
    asr = CloudLectureTranslatorASR()
    recording_client = RecordingClient()

    for _ in range(3):
        input("Press Enter to indicate you're done:")

        batch = recording_client.get_next_batch()
        encoded_batch = base64.b64encode(batch).decode("ascii")
        final_text = asr.transcribe(encoded_batch, len(batch) / 32000)

        if final_text:
            print("Content recognized:", final_text)
        else:
            print("No content recognized.")
