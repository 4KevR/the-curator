import os

from huggingface_hub import InferenceClient
from overrides import overrides

from src.backend.modules.llm.abstract_llm import AbstractLLM


class KitLLM(AbstractLLM):
    def __init__(
        self,
        model: str,
        default_temperature: float,
        default_max_tokens: int,
    ):
        """Initialize the KitLLM client."""
        self.client = InferenceClient(model=os.getenv("LLM_URL"))
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.model = model

    @overrides
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self._format_messages_for_llama(messages)

        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        text_generation = self.client.text_generation(
            prompt=prompt, temperature=temperature, max_new_tokens=max_tokens, stop=["User:", "System:", "Assistant:"]
        )
        return text_generation

    @staticmethod
    def _format_messages_for_llama(messages: list) -> str:
        formatted_messages = []
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            formatted_messages.append(f"{role}: {content}")
        prompt = "\n".join(formatted_messages) + "\nAssistant:"
        return prompt

    def get_description(self) -> str:
        return (
            f"KIT HuggingFace {self.model} with default temperature {self.default_temperature} and "
            f"max tokens {self.default_max_tokens}"
        )
