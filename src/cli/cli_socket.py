import base64
import os
import tempfile
import threading
import wave

import socketio

from src.backend.modules.recording.recording_client import RecordingClient

ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_VIOLET = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_RESET = "\033[0m"
ANSI_CLEAR = "\033[2J\033[H"

sio = socketio.Client()
srs_actions = []
current_actions = []
action_event = threading.Event()


def clear_terminal():
    print(f"{ANSI_CLEAR}", end="")


def print_top(user):
    # Heading
    left = f"{ANSI_VIOLET}the-curator{ANSI_RESET} | User: {ANSI_GREEN}{user}{ANSI_RESET}"
    print(left)
    print("-" * len(left))
    # SRS actions box below heading
    box_title = f"{ANSI_VIOLET}SRS Actions (last 6){ANSI_RESET}"
    print(box_title)
    if srs_actions:
        for action in srs_actions[-6:]:
            print(f"  {ANSI_GREEN}- {action}{ANSI_RESET}")
    else:
        print(f"  {ANSI_YELLOW}(none){ANSI_RESET}")
    print()


def print_question(msg):
    print(f"{ANSI_CYAN}{msg}{ANSI_RESET}")


def print_progress(msg):
    print(f"{ANSI_YELLOW}[Progress]{ANSI_RESET} {msg}")


def print_srs_action(msg):
    print(f"{ANSI_BOLD}[SRS Action]{ANSI_RESET} {msg}")


def print_result(result):
    print(f"{ANSI_BOLD}[Result]{ANSI_RESET} {result}")


def print_error(msg):
    print(f"{ANSI_RED}[Error]{ANSI_RESET} {msg}")


@sio.on("action_progress")
def on_action_progress(data):
    msg = data.get("message", "")
    is_srs_action = data.get("is_srs_action", False)
    if is_srs_action:
        current_actions.append(msg)
        print_srs_action(msg)
    else:
        print_progress(msg)


@sio.on("action_result")
def on_action_result(data):
    result_data = data.get("result", {})
    task_finish_message = result_data.get("task_finish_message", None)
    if task_finish_message:
        print_result(task_finish_message)
    quesion_answer = result_data.get("question_answer", None)
    if quesion_answer:
        print_result(quesion_answer)
    # Add any pending SRS actions to the persistent list
    srs_actions.extend(current_actions)
    current_actions.clear()
    action_event.set()


@sio.on("action_error")
def on_action_error(data):
    print_error(data.get("error", ""))
    sio.disconnect()
    action_event.set()


def websocket_mode(user, mode, value=None):
    action_event.clear()
    if mode == "text":
        sio.emit("submit_action", {"user": user, "transcription": value})
    elif mode == "file":
        with open(value, "rb") as f:
            file_b64 = base64.b64encode(f.read()).decode("ascii")
        sio.emit("submit_action_file", {"user": user, "file_b64": file_b64})
    elif mode == "mic":
        client = RecordingClient()
        input(f"{ANSI_CYAN}Press Enter to start recording...{ANSI_RESET}")
        client.get_next_batch()
        input(f"{ANSI_CYAN}Press Enter to send the next batch...{ANSI_RESET}")
        batch = client.get_next_batch()
        if not batch:
            print_error("No audio data recorded.")
            sio.disconnect()
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(batch)
            tmp_wav_path = tmp_wav.name
        with open(tmp_wav_path, "rb") as f:
            file_b64 = base64.b64encode(f.read()).decode("ascii")
        print(f"{ANSI_YELLOW}Sending audio file...{ANSI_RESET}")
        sio.emit("submit_action_file", {"user": user, "file_b64": file_b64})
        os.remove(tmp_wav_path)
    action_event.wait()


def main():
    server_url = "http://127.0.0.1:5000"
    sio.connect(server_url)
    clear_terminal()
    print_question("Enter your name:")
    user = input(f"{ANSI_GREEN}> {ANSI_RESET}").strip()
    while True:
        clear_terminal()
        print_top(user)
        print_question("Choose interaction type:")
        options = ["text", "file", "mic", "end"]
        for idx, opt in enumerate(options, 1):
            print(f"  {ANSI_BOLD}{idx}{ANSI_RESET}. {opt}")
        print()
        mode_input = input(f"{ANSI_CYAN}Select option [1-{len(options)}]: {ANSI_RESET}").strip()
        if mode_input.isdigit() and 1 <= int(mode_input) <= len(options):
            mode = options[int(mode_input) - 1]
        else:
            print_error("Invalid option. Please choose a valid number.")
            input(f"{ANSI_CYAN}Press Enter to continue...{ANSI_RESET}")
            continue
        if mode == "end":
            print(f"{ANSI_VIOLET}Goodbye!{ANSI_RESET}")
            sio.disconnect()
            break
        elif mode == "text":
            clear_terminal()
            print_top(user)
            print_question("Enter your message:")
            text = input(f"{ANSI_GREEN}> {ANSI_RESET}")
            clear_terminal()
            print_top(user)
            print_question("Enter your message:")
            print(f"{ANSI_GREEN}> {text}{ANSI_RESET}")
            websocket_mode(user, "text", text)
        elif mode == "file":
            clear_terminal()
            print_top(user)
            print_question("Select an audio file or enter a custom path:")
            # List .wav files in current directory
            base_path = "./data/recording_data/combined"
            if not os.path.exists(base_path):
                print_error(f"Directory {base_path} does not exist.")
                input(f"{ANSI_CYAN}Press Enter to continue...{ANSI_RESET}")
                continue
            wav_files = [
                os.path.join(base_path, f)
                for f in os.listdir(base_path)
                if f.lower().endswith(".wav") and os.path.isfile(os.path.join(base_path, f))
            ]
            for idx, fname in enumerate(wav_files, 1):
                print(f"  {ANSI_BOLD}{idx}{ANSI_RESET}. {fname}")
            file_choice = input(f"{ANSI_CYAN}Select file [1-{len(wav_files)}]: {ANSI_RESET}").strip()
            if file_choice.isdigit():
                file_choice = int(file_choice)
                if 1 <= file_choice <= len(wav_files):
                    file_path = wav_files[file_choice - 1]
                else:
                    print_error("Invalid selection.")
                    input(f"{ANSI_CYAN}Press Enter to continue...{ANSI_RESET}")
                    continue
            else:
                print_error("Invalid input.")
                input(f"{ANSI_CYAN}Press Enter to continue...{ANSI_RESET}")
                continue
            if not os.path.isfile(file_path):
                print_error("File does not exist.")
                input(f"{ANSI_CYAN}Press Enter to continue...{ANSI_RESET}")
                continue
            clear_terminal()
            print_top(user)
            print_question("Selected file:")
            print(f"{ANSI_GREEN}> {file_path}{ANSI_RESET}")
            websocket_mode(user, "file", file_path)
        elif mode == "mic":
            clear_terminal()
            print_top(user)
            print_question("Microphone input selected.")
            websocket_mode(user, "mic")
        input(f"{ANSI_CYAN}Press Enter to continue...{ANSI_RESET}")
    sio.disconnect()


if __name__ == "__main__":
    main()
