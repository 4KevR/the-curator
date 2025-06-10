# Meta-Llama-3.1-8B-Instruct
# The above model is deployed locally in LM Studio.
# First check the port usage, the default port is 1234.
# Then start the server using "lms server start"

from openai import OpenAI

from src.backend.modules.llm.AbstractLLM import AbstractLLM


class LMStudioLLM(AbstractLLM):
    """Adapter for LLM Studio."""

    def __init__(self):
        """Initialize the LLM Studio client."""
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    def generate(self, messages: list) -> str:
        return (
            self.client.chat.completions.create(
                model="meta-llama-3.1-8b-instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            .choices[0]
            .message.content
        )
