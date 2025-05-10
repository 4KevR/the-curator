import os

from huggingface_hub import InferenceClient

from src.backend.domain.abstract_adapters import AbstractLLM


class KitLLM(AbstractLLM):
    def __init__(self):
        self.client = InferenceClient(model=os.getenv("LLM_URL"))

    def generate(self, messages: list) -> str:
        prompt = self._format_messages_for_llama(messages)
        return self.client.text_generation(
            prompt=prompt,
            temperature=0.7,
            max_new_tokens=2048,
            stop=["User:", "System:", "Assistant:"],
        )

    def _format_messages_for_llama(self, messages: list) -> str:
        formatted_messages = []
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            formatted_messages.append(f"{role}: {content}")
        return "\n".join(formatted_messages)
