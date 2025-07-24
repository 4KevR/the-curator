from abc import ABC, abstractmethod


class AbstractASR(ABC):
    """Abstract class for ASR adapters."""

    @abstractmethod
    def transcribe(self, audio_chunk: str, duration: int) -> str:
        """Transcribe audio to text. Using the bcm_pcm format with additional base64 encoding."""

    @abstractmethod
    def transcribe_wav_file(self, audio_file_path: str) -> str:
        """Transcribe audio file in .wav format to text."""

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the ASR."""
