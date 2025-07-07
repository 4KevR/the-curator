from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text using the LLM in OpenAI format. Example for messages:
        [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris"}]
        """
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the LLM."""

    def generate_single(
        self, message: str, role: str = "user", temperature: float | None = None, max_tokens: int | None = None
    ) -> str:
        """Shorthand for generating a response from a single message."""
        return self.generate([{"role": role, "content": message}], temperature, max_tokens)
