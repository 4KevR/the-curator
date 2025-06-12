import re
from enum import Enum

from src.backend.modules.llm.abstract_llm import AbstractLLM


class LLMRole(Enum):
    """Enum for the different roles of a message in a llm conversation: user, assistant, or system."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LLMCommunicator:
    """Class for managing a conversation between a user and an LLM."""
    __messages: list[dict[str, str]]
    __all_messages: list[dict[str, str]]
    __llm: AbstractLLM
    __visibility_block_beginning: int | None

    def __init__(self, llm: AbstractLLM):
        self.__llm = llm
        self.__messages = []
        self.__all_messages = []
        self.__visibility_block_beginning = None

    def _add_message(self, message: str, role="user"):
        """Add a new message to the conversation, without sending it to the LLM yet."""
        new_message = {"role": role, "content": message}
        self.messages.append(new_message)
        self.__all_messages.append(new_message)

    def set_system_prompt(self, message: str) -> None:
        """Set the system prompt. May only be called if there is no other message (the first message)."""
        if len(self.__all_messages) > 0:
            raise ValueError("System prompt can only be set as the first message.")
        self._add_message(message, role="system")

    @property
    def messages(self) -> list[dict[str, str]]:
        """Get the list of messages in the conversation. Includes the system prompt if it exists."""
        return list(self.__all_messages)

    def send_message(self, message: str) -> str:
        """Send a (user) message to the LLM and return the response."""
        self._add_message(message)
        response = self.__llm.generate(self.messages)
        self._add_message(response, role=LLMRole.ASSISTANT.value)
        return response

    def start_visibility_block(self):
        """
        Start a visibility block.
        All messages after can be removed from the conversation when calling end_visibility_block().
        If there already is a visibility block, this will overwrite it.
        """
        self.__visibility_block_beginning = len(self.messages)

    def end_visibility_block(self):
        """
        End a visibility block, removing all messages since start_visibility_block() was called.
        If there is no visibility block, calling this method has no effect.
        """
        if self.__visibility_block_beginning is None:
            return
        # cut all messages in the visibility block
        self.__messages = self.__messages[:self.__visibility_block_beginning]
        self.__visibility_block_beginning = None

    def pretty_print(self, skip_thinking=False):
        """
        Prints the conversation in a markdown-friendly format.
        """
        for (role, message) in self.messages:
            if skip_thinking:
                message = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL)
                message = re.sub("\n\n+", "\n", message)
                message = message.strip()

            print(f"## {role}\n\n{message}\n\n")
