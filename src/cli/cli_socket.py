import base64
import os
import tempfile
import threading
import wave
from enum import Enum

import socketio

from src.cli.cli_print import TerminalManager, TerminalPrinter
from src.cli.recording.recording_client import RecordingClient
from src.cli.tts import tts_and_play


class SocketAction(Enum):
    """Enum for different socket action types."""

    TEXT = "text"
    FILE = "file"
    MIC_WHISPER = "mic (whisper)"
    MIC_LT = "mic (LT)"
    NEW_CONVERSATION = "new conversation"
    CHANGE_USER = "change user"
    END = "end"


sio = socketio.Client()
srs_actions = []
current_actions = []
action_event = threading.Event()
sentence_complete_event = threading.Event()
continue_single_cycle = True


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
    task_finish_message = data.get("task_finish_message", None)
    if task_finish_message:
        TerminalPrinter.print_result(task_finish_message)
    quesion_answer = data.get("question_answer", None)
    if quesion_answer:
        TerminalPrinter.print_result(quesion_answer)
        tts_and_play(quesion_answer)
    # Add any pending SRS actions to the persistent list
    srs_actions.extend(current_actions)
    current_actions.clear()
    action_event.set()


@sio.on("action_single_result")
def on_action_single_result(data):
    task_finish_message = data.get("task_finish_message", None)
    TerminalPrinter.print_result(task_finish_message)
    global continue_single_cycle
    continue_single_cycle = True
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
    action_event.set()


@sio.on("acknowledged_stream_start")
def on_acknowledged_stream_start(data):
    TerminalPrinter.print_client_action("Audio transcription stream started.")
    TerminalPrinter.print_transcription_start()
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
        TerminalPrinter.print_client_action("Waiting for stream initialization...")
        sio.emit("start_audio_streaming", {"user": user})


def main():
    server_url = "http://127.0.0.1:5000"
    sio.connect(server_url)
    terminal_manager = TerminalManager()
    exit = False
    global srs_actions
    while not exit:
        try:
            mode = terminal_manager.print_and_execute_selection_screen(
                "Choose interaction type:", SocketAction, srs_actions
            )
            action_event.clear()
            global continue_single_cycle
            continue_single_cycle = True
            reset_view = True
            skip_enter = False
            while continue_single_cycle:
                continue_single_cycle = False
                if mode == SocketAction.END:
                    terminal_manager.print_goodbye()
                    exit = True
                    break
                elif mode == SocketAction.TEXT:
                    text = terminal_manager.execute_text_input(srs_actions, reset_view)
                    action_processor(terminal_manager.user, mode, text)
                elif mode == SocketAction.FILE:
                    file_path = terminal_manager.print_and_execute_path_selection_screen(
                        "Select an audio file or enter a custom path:",
                        "./data/recording_data/combined",
                        srs_actions,
                        reset_view,
                    )
                    action_processor(terminal_manager.user, mode, file_path)
                elif mode == SocketAction.MIC_WHISPER:
                    terminal_manager.print_whisper_screen(srs_actions, reset_view)
                    action_processor(terminal_manager.user, mode)
                elif mode == SocketAction.MIC_LT:
                    terminal_manager.print_lt_screen(srs_actions, reset_view)
                    action_processor(terminal_manager.user, mode)
                elif mode == SocketAction.NEW_CONVERSATION:
                    TerminalPrinter.print_client_action("Starting a new conversation...")
                    sio.emit("new_conversation", {"user": terminal_manager.user})
                    srs_actions = []
                    action_event.set()
                elif mode == SocketAction.CHANGE_USER:
                    sio.emit("new_conversation", {"user": terminal_manager.user})
                    terminal_manager.print_and_execute_user_selection()
                    srs_actions = []
                    action_event.set()
                    skip_enter = True
                action_event.wait()
                if continue_single_cycle:
                    action_event.clear()
                    reset_view = False
            if not exit and not skip_enter:
                TerminalPrinter.wait_for_enter()
        except KeyboardInterrupt:
            terminal_manager.print_goodbye()
            break
    sio.disconnect()


if __name__ == "__main__":
    main()
