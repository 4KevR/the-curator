import os

from src.cli.recording.portaudio_stream_adapter import PortaudioStream
from src.shared.recording.ffmpeg_stream_adapter import FfmpegStream


class RecordingClient:
    def __init__(self, filename: str = None) -> None:
        audio_env = os.getenv("AUDIO_DEVICE", "")
        audio_device = None if audio_env == "" else int(audio_env)
        if not filename:
            self.stream_adapter = PortaudioStream(audio_device)
        else:
            self.stream_adapter = FfmpegStream(
                pre_input=None,
                post_input=None,
                volume=1.0,
                repeat_input=False,
                ffmpeg_speed=1.0,
            )
            self.stream_adapter.set_input(filename)

    def get_next_batch(self) -> bytes:
        chunk = self.stream_adapter.read()
        chunk = self.stream_adapter.chunk_modify(chunk)
        return chunk
