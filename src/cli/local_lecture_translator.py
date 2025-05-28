import base64
import json
import logging
import os
import sys
import time
import wave
from queue import Queue
from threading import Thread
from urllib.parse import urljoin

import requests
from sseclient import SSEClient

logger = logging.getLogger(__name__)


class LocalLectureTranslatorASR:
    def __init__(self):
        self.__token = os.getenv("LECTURE_TRANSLATOR_TOKEN")
        self.url = os.getenv("LECTURE_TRANSLATOR_URL")
        self.api = "webapi"
        self.text_queue = Queue()
        self._run_session()
        self._send_start()

    def _terminate_session(self):
        self._send_end()
        self.session_thread.join()
        self.session_keepalive_thread.join()

    def _run_session(self):
        session_id, stream_id = self._set_graph()
        self.session_url = urljoin(
            self.url, f"{self.api}/{session_id}/{stream_id}/append"
        )

        self.session_thread = Thread(target=self._read_text)
        self.session_thread.daemon = True
        self.session_thread.start()

        time.sleep(1)

        self.session_keepalive_thread = Thread(target=self._send_keepalive)
        self.session_keepalive_thread.daemon = True
        self.session_keepalive_thread.start()

        logging.debug("Requesting worker informations")
        data = {"controll": "INFORMATION"}
        res = requests.post(
            self.session_url,
            json=json.dumps(data),
            cookies={"_forward_auth": self.__token},
        )
        res.raise_for_status()

    def _set_graph(self):
        logging.debug("Requesting default graph for ASR")
        d = {}
        res = requests.post(
            self.url + "/" + self.api + "/get_default_asr",
            json=json.dumps(d),
            cookies={"_forward_auth": self.__token},
        )
        if res.status_code != 200:
            if res.status_code == 401:
                logging.debug(
                    "You are not authorized. "
                    """Either authenticate with --url https://$username:$password@$server
                      or with --token $token where you get the token from """
                    + self.url
                    + "/gettoken",
                )
            else:
                logging.debug(f"Status: {res.status_code}, Text: {res.text}")
                logging.debug("ERROR in requesting default graph for ASR")
            sys.exit(1)
        self.session_id, self.stream_id = res.text.split()

        logging.debug("Setting properties")
        graph = json.loads(
            requests.post(
                self.url + "/" + self.api + "/" + self.session_id + "/getgraph",
                cookies={"_forward_auth": self.__token},
            ).text,
        )
        logging.debug(f"Graph: {graph}")

        return self.session_id, self.stream_id

    def _send_start(self):
        logging.debug("Start sending audio")

        data = {"controll": "START"}
        res = requests.post(
            self.session_url,
            json=json.dumps(data),
            cookies={"_forward_auth": self.__token},
        )
        res.raise_for_status()

    def _send_audio(self, encoded_audio: str, duration: float):
        s = time.time()
        e = s + duration
        data = {
            "b64_enc_pcm_s16le": encoded_audio,
            "start": s,
            "end": e,
        }
        res = requests.post(
            self.session_url,
            json=json.dumps(data),
            cookies={"_forward_auth": self.__token},
        )
        res.raise_for_status()

    def _send_end(self):
        logging.debug("Sending END.")
        data = {"controll": "END"}
        res = requests.post(
            self.session_url,
            json=json.dumps(data),
            cookies={"_forward_auth": self.__token},
        )
        res.raise_for_status()

    def _send_keepalive(self):
        while True:
            data = {"markup": "command"}
            command = {"function": "keep_alive", "parameter": {}}
            data["seq"] = json.dumps(command)
            res = requests.post(
                self.session_url,
                json=json.dumps(data),
                cookies={"_forward_auth": self.__token},
            )
            res.raise_for_status()
            time.sleep(30)

    def _read_text(self):
        logging.debug("Starting SSEClient")
        messages = SSEClient(
            self.url + "/" + self.api + "/stream?channel=" + self.session_id
        )

        for msg in messages:
            if len(msg.data) == 0:
                break
            try:
                data = json.loads(msg.data)
                if "markup" in data:
                    continue
                if "seq" in data:
                    logging.debug(f"Received data: {data}")
                    transcription = data["seq"].replace("<br><br>", "")
                    data = {
                        "transcription": transcription,
                        "user": "test_user",
                    }
                    logger.info(f"Sending {transcription} to server")
                    requests.post(url="http://127.0.0.1:5000/action", json=data)

            except json.decoder.JSONDecodeError:
                logging.debug(
                    """WARNING: json.decoder.JSONDecodeError(this may happen
                      when running tts system but no video generation)""",
                )
                continue

    def process_audio_file(self, file_path: str):
        """Process a local audio file and send its content to the lecture translator."""
        with wave.open(file_path, "rb") as audio_file:
            if audio_file.getframerate() != 16000 or audio_file.getnchannels() != 1:
                raise ValueError(
                    "Audio file must be mono and have a sample rate of 16kHz."
                )

            frames = audio_file.readframes(audio_file.getnframes())
            encoded_audio = base64.b64encode(frames).decode("ascii")
            duration = audio_file.getnframes() / audio_file.getframerate()

            self._send_audio(encoded_audio, duration)
