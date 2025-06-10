from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    @abstractmethod
    def generate(self, messages: list) -> str:
        """Generate text using the LLM."""
        raise NotImplementedError