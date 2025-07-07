from dataclasses import dataclass
from typing import Callable, Optional

from src.backend.modules.ai_assistant.action_states import (
    AbstractActionState,
    StateAction,
    StateAnswer,
    StateFinishedTask,
)
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.llm.logging_llm import LoggingLLM
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.abstract_srs import AbstractSRS


@dataclass
class ExecutionResult:
    task_finish_message: str | None
    question_answer: str | None
    state_history: list[str]
    llm_history: list[list[tuple[str, str]]]


class StateManager:

    _current_state: AbstractActionState | None

    def __init__(
        self,
        task_llm: AbstractLLM,
        srs: AbstractSRS,
        llama_index_executor: LlamaIndexExecutor,
        progress_callback: Callable[[str, Optional[bool]], None] | None = None,
        max_states: int | None = None,
    ):
        self.logging_llm = LoggingLLM(task_llm)
        self._current_state = None
        self.state_history: list[str] = []
        self.srs = srs
        self.llama_index_executor = llama_index_executor
        self.progress_callback = progress_callback
        self.max_states = max_states

    def run(self, user_prompt: str, log_states: bool = False) -> ExecutionResult:

        self._current_state = StateAction(user_prompt, self.logging_llm, self.srs, self.llama_index_executor)
        self.state_history.append(str(self._current_state))

        n_states = 1

        while True:
            if log_states:
                print(f"Current state: {self._current_state}")
            if self.progress_callback:
                self.progress_callback(f"Current State: {str(self._current_state)}")

            next_state = self._current_state.act(self.progress_callback)
            if next_state is None:
                # now we know that we are in one of the end states.
                # Only end states are: StateFinishedTask and StateAnswer.
                answer = self._current_state.answer if isinstance(self._current_state, StateAnswer) else None
                task_msg = self._current_state.message if isinstance(self._current_state, StateFinishedTask) else None
                return ExecutionResult(task_msg, answer, self.state_history, self.logging_llm.get_log())

            if self.max_states is not None and n_states + 1 > self.max_states:
                return ExecutionResult("Reached max states.", None, self.state_history, self.logging_llm.get_log())

            n_states += 1
            self._current_state = next_state
            self.state_history.append(str(self._current_state))
