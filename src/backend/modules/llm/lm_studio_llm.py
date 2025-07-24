# Meta-Llama-3.1-8B-Instruct
# The above model is deployed locally in LM Studio.
# First check the port usage, the default port is 1234.
# Then start the server using "lms server start"
import os

from openai import OpenAI
from overrides import overrides

from src.backend.modules.helpers.string_util import remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM


class LMStudioLLM(AbstractLLM):
    """Adapter for LLM Studio."""

    def __init__(
        self,
        model: str,
        default_temperature: float,
        default_max_tokens: int,
        no_think: bool = False,
    ):
        """Initialize the LLM Studio client."""
        super().__init__()
        if int(os.getenv("IN_DOCKER", "0")):
            self.client = OpenAI(base_url="http://host.docker.internal:1234/v1", api_key="lm-studio")
        else:
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.model = model
        self.no_think = no_think

    @overrides
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        if self.no_think:
            messages = [dict(it) for it in messages]  # copy
            messages[-1]["content"] = messages[-1]["content"] + "\n\\no_think"

        # This works, be quiet
        # noinspection PyTypeChecker
        raw_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = raw_response.choices[0].message.content
        self.current_input_tokens_accumulation += raw_response.usage.prompt_tokens
        self.current_output_tokens_accumulation += raw_response.usage.completion_tokens
        if not self.no_think:
            return response

        response = remove_block(response, "think")
        return response

    def get_description(self) -> str:
        return (
            f"LMStudio {self.model} with default temperature {self.default_temperature} and "
            f'max tokens {self.default_max_tokens}{" with thinking disabled" if self.no_think else ""}'
        )
