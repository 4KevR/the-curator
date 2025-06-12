import os

from huggingface_hub import InferenceClient
from overrides import overrides

from src.backend.modules.llm.abstract_llm import AbstractLLM


class KitLLM(AbstractLLM):
    def __init__(self, temperature: float, max_tokens: int):
        self.client = InferenceClient(model=os.getenv("LLM_URL"))
        self.temperature = temperature
        self.max_tokens = max_tokens

    @overrides
    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self._format_messages_for_llama(messages)
        text_generation = self.client.text_generation(
            prompt=prompt,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,
            stop=["User:", "System:", "Assistant:"],
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
