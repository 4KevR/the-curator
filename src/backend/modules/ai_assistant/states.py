from abc import ABC, abstractmethod
from typing import Optional

from src.backend.modules.ai_assistant.history_manager import HistoryManager
from src.backend.modules.ai_assistant.progress_callback import ProgressCallback
from src.backend.modules.helpers.string_util import find_substring_in_llm_response_or_null, remove_block
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.llm_communicator import LLMCommunicator
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.abstract_srs import AbstractSRS


class AbstractActionState(ABC):

    @abstractmethod
    def act(self) -> Optional["AbstractActionState"]:
        """
        Returns
        * A (different) ActionState if the llm is now in another state.
        * A string to send to the llm and to send the response back to this state.
        * None if the task is completed.
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class ExceedingMaxAttemptsError(Exception):
    """
    Custom exception for exceeding the maximum number of attempts.
    """

    def __init__(self, state_name):
        super().__init__(f"Exceeding maximum number of attempts in state {state_name}.")


class StateAction(AbstractActionState):
    _prompt_template = """
You are an assistant of a flashcard management system. You assist a user in interacting in three ways:
1. Interacting with the flashcard system (creating/modifying/deleting cards/decks etc.),
2. Answering questions about the content of the flashcards or about the system itself (e.g., how many decks exist),
3. Entering and managing study sessions (e.g., starting to learn or review a deck).

The user gave the following prompt:

{user_input}

If you think the user wants you to **interact** with the flashcard system (e.g. creating, modifying, or deleting cards or decks), please answer "task".
If the user wants you to answer a question about the content of the flashcards or about the system itself, please answer "question".
If the user wants to **enter study mode**, such as learning or reviewing a specific deck, please answer "study".
Do not answer anything else.
"""
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        user_prompt: str,
        llm: AbstractLLM,
        srs: AbstractSRS,
        llama_index_executor: LlamaIndexExecutor,
        progress_callback: ProgressCallback,
        history_manager: HistoryManager,
    ):
        self.llm = llm
        self.llm_communicator = LLMCommunicator(llm)
        self.user_prompt = user_prompt
        self.srs = srs
        self.llama_index_executor = llama_index_executor
        self.progress_callback = progress_callback
        self.history_manager = history_manager

    def act(self) -> AbstractActionState | None:
        # believe me I hate that this is necessary, but else we get circular imports.
        from src.backend.modules.ai_assistant.learning_states import StateClassify, StateStartLearn
        from src.backend.modules.ai_assistant.question_states import StateClassifyQuestion
        from src.backend.modules.ai_assistant.task_states import StateRewriteTask

        if self.srs.study_mode:
            return StateClassify(self.user_prompt, self.llm, self.srs, self.progress_callback.handle)

        for attempt in range(self.MAX_ATTEMPTS):
            if attempt == 0:
                message = self._prompt_template.format(user_input=self.user_prompt)
            else:
                message = "Your answer must be either 'question', 'task' or 'study'."

            response = self.llm_communicator.send_message(message)

            response = remove_block(response, "think")
            response = response.replace('"', "").replace("'", "")
            resp = find_substring_in_llm_response_or_null(response, "question", "task", True)

            if resp is True:
                return StateClassifyQuestion(self.user_prompt, self.llm, self.srs, self.llama_index_executor)
            elif resp is False:
                return StateRewriteTask(
                    self.user_prompt,
                    self.llm,
                    self.srs,
                    self.llama_index_executor,
                    self.progress_callback,
                    self.history_manager,
                )
            elif resp is None:
                if "study" in response.lower():
                    return StateStartLearn(self.user_prompt, self.llm, self.srs, self.progress_callback.handle)

        raise ExceedingMaxAttemptsError(self.__class__.__name__)
