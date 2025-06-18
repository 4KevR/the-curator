import requests

from src.backend.modules.llm.abstract_llm import AbstractLLM


class KitLLMReq(AbstractLLM):
    def __init__(
        self,
        llm_url: str,
        default_temperature: float,
        default_max_tokens: int,
    ):
        """Initialize the LLM Studio client."""
        self.llm_url = llm_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

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
        payload = {
            "inputs": self._format_llama_chat(messages),
            "parameters": {"max_new_tokens": 1000, "temperature": 0.05},
        }

        response = requests.post(self.llm_url, json=payload)
        result = response.json()["generated_text"]

        result = result.replace("assistant\n\n", "")
        return result

    def get_description(self) -> str:
        return (
            f"KIT Request meta-llama-3.1-8b-instruct with default temperature {self.default_temperature} and "
            f"max tokens {self.default_max_tokens}"
        )
