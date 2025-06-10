from abc import ABC, abstractmethod
from typing import Optional


class AbstractPDFReader(ABC):
    """Abstract class for PDF reader adapters."""

    @abstractmethod
    def read(
        self, file_path: str, page_range: Optional[tuple[int, int]] = None
    ) -> dict:
        """Read text from a PDF file."""
        raise NotImplementedError