from src.backend.modules.ai_assistant.history_manager import HistoryManager
from src.backend.modules.ai_assistant.progress_callback import ProgressCallback
from src.backend.modules.ai_assistant.state_manager import ExecutionResult, StateManager
from src.backend.modules.ai_assistant.task_states import StateFinishedDueToMissingInformation
from src.backend.modules.llm.abstract_llm import AbstractLLM
from src.backend.modules.search.llama_index import LlamaIndexExecutor
from src.backend.modules.srs.abstract_srs import AbstractSRS


class ConversationManager:
    def __init__(
        self,
        task_llm: AbstractLLM,
        srs: AbstractSRS,
        llama_index_executor: LlamaIndexExecutor,
        progress_callback: ProgressCallback,
        max_states: int | None = None,
    ):
        self.history_manager = HistoryManager()
        self.state_manager = StateManager(
            task_llm=task_llm,
            srs=srs,
            llama_index_executor=llama_index_executor,
            progress_callback=progress_callback,
            history_manager=self.history_manager,
            max_states=max_states,
        )

    def process_query(
        self,
        query: str,
    ) -> ExecutionResult:
        idx_to_start = max(
            [
                i
                for i in range(len(self.history_manager.latest_execution_result))
                if type(self.history_manager.latest_execution_result[i].finish_state)
                is not StateFinishedDueToMissingInformation
            ],
            default=0,
        )
        query_to_send = " - ".join(self.history_manager.latest_queries[idx_to_start + 1 :] + [query])

        result = self.state_manager.run(query_to_send)
        self.history_manager.latest_queries.append(query)
        self.history_manager.latest_execution_result.append(result)
        return result
