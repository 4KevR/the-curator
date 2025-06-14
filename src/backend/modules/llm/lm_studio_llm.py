# Meta-Llama-3.1-8B-Instruct
# The above model is deployed locally in LM Studio.
# First check the port usage, the default port is 1234.
# Then start the server using "lms server start"

from openai import OpenAI
from overrides import overrides

from src.backend.modules.llm.abstract_llm import AbstractLLM


class LMStudioLLM(AbstractLLM):
    """Adapter for LLM Studio."""

    def __init__(self, model: str, default_temperature: float, default_max_tokens: int, add_no_think: bool = False):
        """Initialize the LLM Studio client."""
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.model = model
        self.add_no_think = add_no_think

    @overrides
    def generate(
        self, messages: list[dict[str, str]], temperature: float | None = None, max_tokens: int | None = None
    ) -> str:
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        if self.add_no_think:
            messages[-1]["content"] = messages[-1]["content"] + "\n\\no_think"

        # This works, be quiet
        # noinspection PyTypeChecker
        return (
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            .choices[0]
            .message.content
        )
