import logging
import queue
import sys
import threading
from typing import Optional

import numpy as np
import pyaudio

from src.backend.modules.recording.input_stream_adapter import BaseAdapter


class NoAudioDeviceException(Exception):
    pass


class PortaudioStream(BaseAdapter):
    def __init__(self, device_id: int = None, **kwargs) -> None:
        self.input_id: Optional[int] = device_id
        self._stream: Optional[pyaudio.Stream] = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self.chunk_size = 1024
        super().__init__(format=pyaudio.paInt16)
        self._pyaudio = pyaudio.PyAudio()
        if self.input_id is None:
            self.print_all_devices()
            raise NoAudioDeviceException
        self._stream = self._pyaudio.open(
            format=self.format,
            input_device_index=self.input_id,
            channels=self.channel_count,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        print("Format: ", self.format)
        print("Channels: ", self.channel_count)
        print("Rate: ", self.rate)
        print("Chunk size: ", self.chunk_size)
        print("Input device: ", self.input_id)
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()

    def _process_audio(self):
        while self.running:
            chunk = self._stream.read(self.chunk_size, exception_on_overflow=False)
            self.queue.put(chunk)

    def get_stream(self, **kwargs):
        return self._stream

    def read(self) -> bytes:
        combined_chunk = b""
        while not self.queue.empty():
            combined_chunk += self.queue.get()
        return combined_chunk

    def chunk_modify(self, chunk: bytes) -> bytes:
        if self.chosen_channel is not None and self.channel_count > 1:
            # filter out specific channel using numpy
            logging.info("Using numpy to filter out specific channel.")
            data = np.fromstring(chunk, dtype="int16").reshape((-1, self.channel_count))
            data = data[:, self.chosen_channel - 1]
            chunk = data.tostring()
        return chunk

    def cleanup(self) -> None:
        self.running = False
        self.thread.join()
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()

        if self._pyaudio is not None:
            self._pyaudio.terminate()

    def set_input(self, id: int) -> None:
        devices = self.get_audio_devices()
        try:
            devices[id]
            self.input_id = id
        except (ValueError, KeyError):
            self.print_all_devices()
            sys.exit(1)

    def get_audio_devices(self) -> dict[int, str]:
        devices = {}

        info = self._pyaudio.get_host_api_info_by_index(0)
        deviceCount = info.get("deviceCount")

        for i in range(0, deviceCount):
            if self._pyaudio.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels") > 0:
                devices[i] = self._pyaudio.get_device_info_by_host_api_device_index(0, i).get("name")
        return devices

    def print_all_devices(self) -> None:
        """
        Special command, prints all audio devices available
        """
        print("-- AUDIO devices:")
        devices = self.get_audio_devices()
        for key in devices:
            dev = devices[key]
            if isinstance(dev, bytes):
                dev = dev.decode("ascii", "replace")
            print(f"    id={key} - {dev}")

    def set_audio_channel_filter(self, channel: int) -> None:
        # actually chosing a specific channel is apparently impossible with portaudio,
        # so we record all channels instead and then filter out
        # the wanted channel with numpy
        channelCount = self._pyaudio.get_device_info_by_host_api_device_index(0, self.input_id).get("maxInputChannels")
        self.channel_count = channelCount
        self.chosen_channel = channel
