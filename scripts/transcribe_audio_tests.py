import json
import os
import sys
import time

from dotenv import load_dotenv

# Load environment variables from .env and .env.db
load_dotenv(".env")
load_dotenv(".env.db")

# Add the project root to the sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backend.modules.asr.cloud_lecture_translator import CloudLectureTranslatorASR  # noqa E402
from src.backend.modules.evaluation.load_test_data.load_test_data import load_test_data  # noqa E402


def transcribe_audio_tests(audio_dir: str, tests_json_path: str) -> dict:
    """
    Transcribe audio files for tests and return a dictionary of transcriptions.
    """
    transcriptions = {}
    asr = CloudLectureTranslatorASR()

    tests_data = load_test_data(tests_json_path)
    # interaction_tests = tests_data.interaction[:]
    interaction_tests = []
    question_answering_tests = tests_data.question_answering[:10]

    for test in interaction_tests + question_answering_tests:
        if hasattr(test, "sound_file_names") and test.sound_file_names:
            for sound_file_base_name in test.sound_file_names:
                file_path = os.path.join(audio_dir, f"{sound_file_base_name}.wav")
                tag = sound_file_base_name  # Use the base name as the tag

                if os.path.exists(file_path):
                    print(f"Transcribing {file_path}...", end="")
                    try:
                        max_retries = 3
                        for attempt in range(max_retries):
                            transcription = asr.transcribe_wav_file(file_path)
                            if transcription != "":
                                break
                            if attempt < max_retries - 1:
                                print(f" (empty, retrying in 5s, attempt {attempt + 2}/{max_retries})", end="")
                                time.sleep(5)
                        if transcription != "":
                            transcriptions[tag] = transcription
                            print(f"  -> {transcription}")
                        else:
                            print(f"  -> No transcription available after {max_retries} attempts.")
                    except Exception as e:
                        print(f"Error transcribing {file_path}: {e}")
                        transcriptions[tag] = f"ERROR: {e}"
                else:
                    print(f"Warning: Audio file not found: {file_path}")
    return transcriptions


if __name__ == "__main__":
    audio_data_path = "data/recording_data/combined"
    tests_json_path = "tests/data/tests.json"
    save_transcriptions_path = "transcriptions_cache_new.json"

    # Ensure the audio data path exists
    if not os.path.isdir(audio_data_path):
        print(f"Error: Audio data directory not found at {audio_data_path}")
        sys.exit(1)

    # Ensure tests.json exists
    if not os.path.exists(tests_json_path):
        print(f"Error: tests.json not found at {tests_json_path}")
        sys.exit(1)

    all_transcriptions = transcribe_audio_tests(audio_data_path, tests_json_path)

    print("\n--- All Transcriptions ---")
    print(json.dumps(all_transcriptions, indent=2))

    # Save this dictionary to a file for caching
    with open(save_transcriptions_path, "w") as f:
        json.dump(all_transcriptions, f, indent=2)
    print(f"\nTranscriptions saved to {save_transcriptions_path}")
