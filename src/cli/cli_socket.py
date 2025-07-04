import base64
import os
import re
import tempfile
import threading
import wave
from enum import Enum

import socketio

from src.shared.recording.recording_client import RecordingClient

from .tts import tts_and_play


class ANSI(Enum):
    """ANSI escape codes for terminal formatting."""

    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    VIOLET = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    CLEAR = "\033[2J\033[H"
    UNDERLINE = "\033[4m"


class SocketAction(Enum):
    """Enum for different socket action types."""

    TEXT = "text"
    FILE = "file"
    MIC_WHISPER = "mic (whisper)"
    MIC_LT = "mic (LT)"
    END = "end"


sio = socketio.Client()
srs_actions = []
current_actions = []
action_event = threading.Event()
sentence_complete_event = threading.Event()


class TerminalManager:
    def __init__(self):
        self.print_and_execute_user_selection()

    def print_heading(self):
        # Heading
        TerminalPrinter.print_headline(self.user)
        # SRS actions box below heading
        TerminalPrinter.print_title("SRS Actions (last 6)")
        if srs_actions:
            for action in srs_actions[-6:]:
                TerminalPrinter.print_past_srs_action(action)
        else:
            TerminalPrinter.print_none_srs_action()
        TerminalPrinter.print_empty_line()

    def print_and_execute_user_selection(self):
        TerminalPrinter.clear()
        self.user = TerminalPrinter.print_user_input_request("Enter your name")

    def print_and_execute_path_selection_screen(self, selection_query: str, path: str) -> str:
        while True:
            TerminalPrinter.clear()
            self.print_heading()
            TerminalPrinter.print_question(selection_query)
            if not os.path.exists(path):
                TerminalPrinter.print_error(f"Directory {path} does not exist.")
                path = TerminalPrinter.print_user_input_request("Please enter a valid recording path:")
                continue
            wav_files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith(".wav") and os.path.isfile(os.path.join(path, f))
            ]
            file_choice = TerminalPrinter.print_list_selection_block(wav_files)
            if file_choice.isdigit():
                file_choice = int(file_choice)
                if 1 <= file_choice <= len(wav_files):
                    file_path = wav_files[file_choice - 1]
                else:
                    TerminalPrinter.print_error("Invalid selection.")
                    continue
            else:
                file_path = file_choice.strip()
            if not os.path.isfile(file_path):
                TerminalPrinter.print_error("File does not exist.")
                continue
            TerminalPrinter.clear()
            self.print_heading()
            TerminalPrinter.print_client_action(f"Selected file: {file_path}")
            break
        return file_path

    def print_and_execute_selection_screen(self, selection_query: str, options: Enum) -> Enum:
        while True:
            TerminalPrinter.clear()
            self.print_heading()
            TerminalPrinter.print_question(selection_query)
            mode_input = TerminalPrinter.print_enum_selection_block(options)
            if mode_input.isdigit() and 1 <= int(mode_input) <= len(options):
                return list(options)[int(mode_input) - 1]
            else:
                TerminalPrinter.print_error("Invalid option. Please choose a valid number.")

    def print_whisper_screen(self):
        TerminalPrinter.clear()
        self.print_heading()
        TerminalPrinter.print_client_action("Microphone input (whisper) selected.")

    def print_lt_screen(self):
        TerminalPrinter.clear()
        self.print_heading()
        TerminalPrinter.print_client_action("Microphone input (LT) selected.")

    def print_goodbye(self):
        TerminalPrinter.print_title("Goodbye!")

    def _reset_view(self):
        TerminalPrinter.clear()
        self.print_heading()

    def execute_text_input(self) -> str:
        self._reset_view()
        return TerminalPrinter.print_user_input_request("Enter your message")


class TerminalPrinter:
    """
    TerminalPrinter provides static methods for printing messages in different styles.
    It uses ANSI escape codes for color and formatting.
    """

    @staticmethod
    def print_headline(user: str):
        left = f"{ANSI.VIOLET.value}the-curator{ANSI.RESET.value} | User: {ANSI.GREEN.value}{user}{ANSI.RESET.value}"
        print(left)
        visible_length = len(re.sub(r"\x1b\[[0-9;]*m", "", left))
        print("-" * visible_length)

    @staticmethod
    def print_title(title: str):
        print(f"{ANSI.VIOLET.value}{title}{ANSI.RESET.value}")

    @staticmethod
    def print_past_srs_action(action: str):
        print(f"  {ANSI.GREEN.value}- {action}{ANSI.RESET.value}")

    @staticmethod
    def print_empty_line():
        print()

    @staticmethod
    def print_none_srs_action():
        print(f"  {ANSI.YELLOW.value}(none){ANSI.RESET.value}")

    @staticmethod
    def print_question(msg: str, newline: bool = True):
        print(f"{ANSI.CYAN.value}{msg}{ANSI.RESET.value}", end="\n" if newline else "")

    @staticmethod
    def print_client_action(msg: str):
        print(f"{ANSI.YELLOW.value}[Client]{ANSI.RESET.value} {msg}")

    @staticmethod
    def print_progress(msg: str):
        print(f"{ANSI.YELLOW.value}[Progress]{ANSI.RESET.value} {msg}")

    @staticmethod
    def print_srs_action(msg: str):
        print(f"{ANSI.BOLD.value}[SRS Action]{ANSI.RESET.value} {msg}")

    @staticmethod
    def print_result(result: str):
        print(f"{ANSI.BOLD.value}[Result]{ANSI.RESET.value} {result}")

    @staticmethod
    def print_transcription_start():
        print(f"{ANSI.YELLOW.value}[Transcription]{ANSI.RESET.value} ", end="")

    @staticmethod
    def print_transcription_words(msg: str):
        print(f"{msg}", end="")

    @staticmethod
    def print_transcription_end():
        print(f"{ANSI.RESET.value}")

    @staticmethod
    def print_error(msg: str):
        print(f"{ANSI.RED.value}[Error]{ANSI.RESET.value} {msg}")
        TerminalPrinter.wait_for_enter()

    @staticmethod
    def wait_for_enter():
        TerminalPrinter.print_user_enter_request("Press Enter to continue...")

    @staticmethod
    def print_user_enter_request(msg: str):
        input(f"{ANSI.CYAN.value}{msg}{ANSI.RESET.value}")

    def print_user_input_request(msg: str) -> str:
        TerminalPrinter.print_question(msg, newline=False)
        return input(f"{ANSI.CYAN.value} > {ANSI.RESET.value}").strip()

    @staticmethod
    def print_enum_selection_block(options: Enum) -> str:
        for idx, opt in enumerate(options, 1):
            print(f"  {ANSI.BOLD.value}{idx}{ANSI.RESET.value}. {opt.value}")
        print()
        return input(f"{ANSI.CYAN.value}Select option [1-{len(options)}]: {ANSI.RESET.value}").strip()

    @staticmethod
    def print_list_selection_block(options: list[str]) -> str:
        for idx, opt in enumerate(options, 1):
            print(f"  {ANSI.BOLD.value}{idx}{ANSI.RESET.value}. {opt}")
        print()
        return input(f"{ANSI.CYAN.value}Select option [1-{len(options)}]: {ANSI.RESET.value}").strip()

    @staticmethod
    def clear():
        print(f"{ANSI.CLEAR.value}", end="")


@sio.on("action_progress")
def on_action_progress(data):
    msg = data.get("message", "")
    is_srs_action = data.get("is_srs_action", False)
    if is_srs_action:
        current_actions.append(msg)
        TerminalPrinter.print_srs_action(msg)
    else:
        TerminalPrinter.print_progress(msg)


@sio.on("action_result")
def on_action_result(data):
    result_data = data.get("result", {})
    task_finish_message = result_data.get("task_finish_message", None)
    if task_finish_message:
        TerminalPrinter.print_result(task_finish_message)
    quesion_answer = result_data.get("question_answer", None)
    if quesion_answer:
        TerminalPrinter.print_result(quesion_answer)
        tts_and_play(quesion_answer)
    # Add any pending SRS actions to the persistent list
    srs_actions.extend(current_actions)
    current_actions.clear()
    action_event.set()


@sio.on("streamed_sentence_part")
def on_streamed_sentence_part(data):
    part = data.get("part", "")
    TerminalPrinter.print_transcription_words(part)


@sio.on("received_complete_sentence")
def on_received_complete_sentence(data):
    sentence = data.get("sentence", "")
    sentence_complete_event.set()
    TerminalPrinter.print_transcription_end()
    TerminalPrinter.print_progress(f"Processing sentence: {sentence}")


@sio.on("action_error")
def on_action_error(data):
    TerminalPrinter.print_error(data.get("error", ""))
    sio.disconnect()
    action_event.set()


@sio.on("acknowledged_stream_start")
def on_acknowledged_stream_start(data):
    user = data.get("user")
    batch = b""
    client = RecordingClient()
    while True:
        if sentence_complete_event.is_set():
            sentence_complete_event.clear()
            break
        batch += client.get_next_batch()
        duration = len(batch) / 32000
        if not batch:
            continue
        if duration < 1:
            continue
        processed_batch = base64.b64encode(batch).decode("ascii")
        sio.emit("submit_stream_batch", {"user": user, "b64_pcm": processed_batch, "duration": duration})
        batch = b""


def action_processor(user: str, mode: SocketAction, value=None):
    action_event.clear()
    if mode == SocketAction.TEXT:
        sio.emit("submit_action", {"user": user, "transcription": value})
    elif mode == SocketAction.FILE:
        with open(value, "rb") as f:
            file_b64 = base64.b64encode(f.read()).decode("ascii")
        sio.emit("submit_action_file", {"user": user, "file_b64": file_b64})
    elif mode == SocketAction.MIC_WHISPER:
        client = RecordingClient()
        TerminalPrinter.print_user_enter_request("Press Enter to start recording...")
        client.get_next_batch()
        TerminalPrinter.print_user_enter_request("Press Enter to send the next batch...")
        batch = client.get_next_batch()
        if not batch:
            TerminalPrinter.print_error("No audio data recorded.")
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
        TerminalPrinter.print_client_action("Sending audio file for transcription...")
        sio.emit("submit_action_file", {"user": user, "file_b64": file_b64})
        os.remove(tmp_wav_path)
    elif mode == SocketAction.MIC_LT:
        TerminalPrinter.print_client_action("Listening for audio input...")
        TerminalPrinter.print_transcription_start()
        sio.emit("start_audio_streaming", {"user": user})
    action_event.wait()
    TerminalPrinter.wait_for_enter()


def main():
    server_url = "http://127.0.0.1:5000"
    sio.connect(server_url)
    terminal_manager = TerminalManager()
    while True:
        try:
            mode = terminal_manager.print_and_execute_selection_screen("Choose interaction type:", SocketAction)
            if mode == SocketAction.END:
                terminal_manager.print_goodbye()
                break
            elif mode == SocketAction.TEXT:
                text = terminal_manager.execute_text_input()
                action_processor(terminal_manager.user, mode, text)
            elif mode == SocketAction.FILE:
                file_path = terminal_manager.print_and_execute_path_selection_screen(
                    "Select an audio file or enter a custom path:", "./data/recording_data/combined"
                )
                action_processor(terminal_manager.user, mode, file_path)
            elif mode == SocketAction.MIC_WHISPER:
                terminal_manager.print_whisper_screen()
                action_processor(terminal_manager.user, mode)
            elif mode == SocketAction.MIC_LT:
                terminal_manager.print_lt_screen()
                action_processor(terminal_manager.user, mode)
        except KeyboardInterrupt:
            terminal_manager.print_goodbye()
            break
    sio.disconnect()


if __name__ == "__main__":
    main()
