import os

from src.cli.recording.portautio_stream_adapter import PortaudioStream


class RecordingClient:
    def __init__(self):
        audio_env = os.getenv("AUDIO_DEVICE", "")
        audio_device = None if audio_env == "" else int(audio_env)
        self.stream_adapter = PortaudioStream(audio_device)

    def get_next_batch(self) -> bytes:
        chunk = self.stream_adapter.read()
        return chunk
