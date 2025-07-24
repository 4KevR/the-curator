import os

import requests
from transformers import AutoTokenizer

from src.backend.modules.llm.abstract_llm import AbstractLLM


class KitLLMReq(AbstractLLM):
    def __init__(
        self,
        llm_url: str,
        default_temperature: float,
        default_max_tokens: int,
    ):
        """Initialize the KitLLM client."""
        super().__init__()
        self.llm_url = llm_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, token=os.getenv("HUGGING_FACE_TOKEN"), cache_dir="./model_cache"
        )

    @staticmethod
    def _format_llama_chat(messages):
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>"
        return formatted

    def generate(
        self, messages: list[dict[str, str]], temperature: float | None = None, max_tokens: int | None = None
    ) -> str:
        prompt = self._format_llama_chat(messages)

        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        payload = {
            "inputs": self._format_llama_chat(messages),
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
        }

        response = requests.post(self.llm_url, json=payload)
        result: str = response.json()["generated_text"]

        self.current_input_tokens_accumulation += len(self.tokenizer(prompt).input_ids)
        self.current_output_tokens_accumulation += len(self.tokenizer(result).input_ids)

        result = result.lstrip().replace("assistant", "").lstrip()
        return result

    def get_description(self) -> str:
        return (
            f"KIT Request meta-llama-3.1-8b-instruct with default temperature {self.default_temperature} and "
            f"max tokens {self.default_max_tokens}"
        )
