import os

from huggingface_hub import InferenceClient
from overrides import overrides
from transformers import AutoTokenizer

from src.backend.modules.llm.abstract_llm import AbstractLLM


class KitLLM(AbstractLLM):
    def __init__(
        self,
        default_temperature: float,
        default_max_tokens: int,
    ):
        """Initialize the KitLLM client."""
        super().__init__()
        self.client = InferenceClient(model=os.getenv("LLM_URL"))
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, token=os.getenv("HUGGING_FACE_TOKEN"), cache_dir="./model_cache"
        )

    @overrides
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        text_generation = self.client.text_generation(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_tokens,
            details=True,
        )
        # Accumulate input tokens by tokenizing the prompt string using the local tokenizer.
        # This counts the tokens in the string sent to the inference client.
        self.current_input_tokens_accumulation += len(self.tokenizer(prompt).input_ids)
        self.current_output_tokens_accumulation += text_generation.details.generated_tokens
        return text_generation.generated_text

    def get_description(self) -> str:
        return (
            f"KIT HuggingFace {self.model} with default temperature {self.default_temperature} and "
            f"max tokens {self.default_max_tokens}"
        )
