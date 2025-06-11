from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


@dataclass
class LLM_conversation_role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class LLMConversation:
    messages: list[tuple[LLM_conversation_role, str]]


class AbstractLLM(ABC):
    """Abstract class for LLM adapters."""

    @abstractmethod
    def generate(self, messages: LLMConversation) -> str:
        """Generate text using the LLM."""
        raise NotImplementedError
