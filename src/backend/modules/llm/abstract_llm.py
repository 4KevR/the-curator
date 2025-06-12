from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Generate text using the LLM in OpenAI format. Example for messages:
        [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris"}]
        """
        raise NotImplementedError
