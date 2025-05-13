import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Tuple

import pandas as pd
import sounddevice as sd
import soundfile as sf
import json
import numpy as np
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from functional import seq

from poetry.console.commands import self


class PromptRecorderApp:
    # prompts must be list of tuples (id, prompt)
    def __init__(self, root, prompts: List[Tuple[str, str]]):
        self.root = root
        self.root.title("Prompt Recorder")

        self.prompt_index = 0
        self.recording = None
        self.fs = 44100
        self.recording_data = None
        self.filename = None
        self.record_start_frame = None
        self.record_stop_frame = None

        now = datetime.now(ZoneInfo("Europe/Berlin")).strftime(f"%Y-%m-%d %H:%M:%S %z")
        out_dir_name = f"recording session {now}"
        os.makedirs(out_dir_name, exist_ok=True)
        self.out_directory = f"{out_dir_name}/"

        self.prompts = prompts
        self.setup_gui()
        self.bind_shortcuts()

    def setup_gui(self):
        self.prompt_label = tk.Label(self.root, text="", wraplength=400, font=("Arial", 14))
        self.prompt_label.pack(pady=10)

        self.mic_label = tk.Label(self.root, text="Select Microphone:")
        self.mic_label.pack()
        self.device_list = sd.query_devices()
        self.input_devices = [
            (i, d['name']) for i, d in enumerate(self.device_list) if d['max_input_channels'] > 0
        ]
        self.device_names = [name for i, name in self.input_devices]
        self.device_index_map = {name: i for i, name in self.input_devices}

        print("Available input devices:")
        for i, name in self.input_devices:
            print(f"  [{i}] {name}")

        self.mic_combo = ttk.Combobox(self.root, values=self.device_names, state="readonly")
        self.mic_combo.pack(pady=5)
        self.mic_combo.current(0)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="Start Recording (A)", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop Recording (S)", command=self.stop_recording)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.listen_btn = tk.Button(btn_frame, text="Listen to Recording (D)", command=self.listen_recording)
        self.listen_btn.grid(row=0, column=2, padx=5)

        self.submit_btn = tk.Button(btn_frame, text="Submit (Space)", command=self.submit_recording)
        self.submit_btn.grid(row=0, column=3, padx=5)

        self.skip_btn = tk.Button(btn_frame, text="Skip (Backspace)", command=self.skip_prompt)
        self.skip_btn.grid(row=0, column=4, padx=5)

        self.display_prompt()

    def bind_shortcuts(self):
        self.root.bind('<a>', lambda event: self.start_recording())
        self.root.bind('<s>', lambda event: self.stop_recording())
        self.root.bind('<d>', lambda event: self.listen_recording())
        self.root.bind('<space>', lambda event: self.submit_recording())
        self.root.bind('<BackSpace>', lambda event: self.skip_prompt())

    def display_prompt(self):
        if self.prompt_index < len(self.prompts):
            self.prompt = self.prompts[self.prompt_index]
            self.prompt_label.config(text=self.prompt[1])
            self.filename = f"{self.prompt[0]}.wav"
            self.recording_data = None
        else:
            self.prompt_label.config(text="All prompts recorded.")
            self.disable_buttons()

    def disable_buttons(self):
        for btn in [self.start_btn, self.stop_btn, self.listen_btn, self.submit_btn, self.skip_btn]:
            btn.config(state=tk.DISABLED)

    def start_recording(self):
        selected_device = self.device_index_map[self.mic_combo.get()]
        if self.recording_data is not None:
            self.recording_data = None
        self.recording = sd.rec(int(60 * self.fs), samplerate=self.fs, channels=1, device=selected_device)
        print("Recording started...")

    def stop_recording(self):
        if self.recording is not None:
            sd.stop()
            self.recording_data = self.recording[:np.flatnonzero(self.recording)[-1]].copy()
            self.recording = None
            print("Recording stopped.")

    def listen_recording(self):
        if self.recording_data is not None:
            sd.play(self.recording_data, self.fs)
            sd.wait()
        else:
            messagebox.showinfo("Info", "No recording to play.")

    def submit_recording(self):
        if self.recording_data is not None:
            sf.write(self.out_directory + self.filename, self.recording_data, self.fs)
            print(f"Recording saved to {self.filename}")
            self.prompt_index += 1
            self.display_prompt()
        else:
            messagebox.showwarning("Warning", "No recording to submit.")

    def skip_prompt(self):
        self.prompt_index += 1
        self.display_prompt()


def get_prompts_from_tests():
    json_object = json.load(open("../tests/tests.json"))
    prompts_test_single_turn = [
        (it["name"] + f"_{q_id}", q)
        for it in json_object["tests"]
        for (q_id, q) in enumerate(it["queries"][0])
        if len(it["queries"]) == 1
    ]
    prompts_test_multi_turn = len([
        1
        for it in json_object["tests"]
        if len(it["queries"]) != 1
    ])
    prompts_test_question_answering_single = [
        (it["name"] + f"_{q_id}", q)
        for subject in json_object["question_answering"]
        for it in json_object["question_answering"][subject]
        for (q_id, q) in enumerate(it["queries"][0])
        if len(it["queries"]) == 1
    ]
    prompts_test_question_answering_multi_turn = len([
        1
        for subject in json_object["question_answering"]
        for it in json_object["question_answering"][subject]
        if len(it["queries"]) != 1
    ])

    print(f"There are {prompts_test_multi_turn} multi-turn test prompts and {prompts_test_question_answering_multi_turn} multi-turn question answering prompts – you have to record them manually!")

    return prompts_test_single_turn + prompts_test_question_answering_single


if __name__ == "__main__":
    single_prompts = get_prompts_from_tests()
    single_prompts_sample = pd.Series(single_prompts).sample(50, random_state=2308421)

    print("\n".join(str(it) for it in single_prompts_sample))

    root = tk.Tk()
    app = PromptRecorderApp(root, list(single_prompts_sample))
    root.mainloop()
