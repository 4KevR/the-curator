from abc import ABC, abstractmethod

from src.backend.modules.llm.types import TokenUsage


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    def __init__(self):
        """Initialize the LLM client."""
        self.current_input_tokens_accumulation = 0
        self.current_output_tokens_accumulation = 0

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

    def get_and_reset_token_usage(self) -> TokenUsage:
        """Get and reset the token usage statistics."""
        token_usage = TokenUsage(
            prompt_tokens=self.current_input_tokens_accumulation,
            completion_tokens=self.current_output_tokens_accumulation,
        )
        self.current_input_tokens_accumulation = 0
        self.current_output_tokens_accumulation = 0
        return token_usage

    def generate_single(
        self, message: str, role: str = "user", temperature: float | None = None, max_tokens: int | None = None
    ) -> str:
        """Shorthand for generating a response from a single message."""
        return self.generate([{"role": role, "content": message}], temperature, max_tokens)
