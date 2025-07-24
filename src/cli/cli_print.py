import os
import re
from enum import Enum


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


class TerminalManager:
    def __init__(self):
        self.print_and_execute_user_selection()

    def print_heading(self, srs_actions: list[str] = None):
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

    def print_and_execute_path_selection_screen(
        self, selection_query: str, path: str, srs_actions: list[str] = None, reset_view: bool = True
    ) -> str:
        while True:
            if reset_view:
                self._reset_view(srs_actions=srs_actions)
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
            if reset_view:
                self._reset_view(srs_actions=srs_actions)
            TerminalPrinter.print_client_action(f"Selected file: {file_path}")
            break
        return file_path

    def print_and_execute_selection_screen(
        self, selection_query: str, options: Enum, srs_actions: list[str] = None
    ) -> Enum:
        while True:
            self._reset_view(srs_actions=srs_actions)
            TerminalPrinter.print_question(selection_query)
            mode_input = TerminalPrinter.print_enum_selection_block(options)
            if mode_input.isdigit() and 1 <= int(mode_input) <= len(options):
                return list(options)[int(mode_input) - 1]
            else:
                TerminalPrinter.print_error("Invalid option. Please choose a valid number.")

    def print_whisper_screen(self, srs_actions: list[str] = None, reset_view: bool = True):
        if reset_view:
            self._reset_view(srs_actions=srs_actions)
        TerminalPrinter.print_client_action("Microphone input (whisper) selected.")

    def print_lt_screen(self, srs_actions: list[str] = None, reset_view: bool = True):
        if reset_view:
            self._reset_view(srs_actions=srs_actions)
        TerminalPrinter.print_client_action("Microphone input (LT) selected.")

    def print_goodbye(self):
        TerminalPrinter.print_title("Goodbye!")

    def _reset_view(self, srs_actions: list[str] = None):
        TerminalPrinter.clear()
        self.print_heading(srs_actions=srs_actions)

    def execute_text_input(self, srs_actions: list[str] = None, reset_view: bool = True) -> str:
        if reset_view:
            self._reset_view(srs_actions=srs_actions)
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
