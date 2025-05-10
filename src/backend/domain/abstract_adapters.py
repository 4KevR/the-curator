from abc import ABC, abstractmethod
from typing import Optional


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    @abstractmethod
    def generate(self, messages: list) -> str:
        """Generate text using the LLM."""
        raise NotImplementedError


class AbstractASR(ABC):
    """Abstract class for ASR adapters."""

    @abstractmethod
    def transcribe(self, audio_chunk: str, duration: int) -> str:
        """Transcribe audio to text."""
        raise NotImplementedError


class AbstractPDFReader(ABC):
    """Abstract class for PDF reader adapters."""

    @abstractmethod
    def read(
        self, file_path: str, page_range: Optional[tuple[int, int]] = None
    ) -> dict:
        """Read text from a PDF file."""
        raise NotImplementedError
