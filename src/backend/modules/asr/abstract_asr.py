from abc import ABC, abstractmethod


class AbstractASR(ABC):
    """Abstract class for ASR adapters."""

    @abstractmethod
    def transcribe(self, audio_chunk: str, duration: int) -> str:
        """Transcribe audio to text."""
        raise NotImplementedError
